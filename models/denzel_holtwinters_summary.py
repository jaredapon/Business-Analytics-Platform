#!/usr/bin/env python3
# denzel_holtwinters_summary_aligned.py
#
# Summary of Holt–Winters results for TOP_N bundles, aligned with the in-depth script:
#  - Resolve products by EXACT product_name
#  - Build one COMMON_FC_INDEX from the latest actual among bundle/A/B
#  - No asfreq() padding (sparse series allowed, like the in-depth file)
#  - Elasticity: apply power model to RAW bundle forecast, THEN clip
#  - Cannibalization uses non-negative baseline bundle units
#  - Revenue math on the COMMON_FC_INDEX

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# USER CONFIGURATION
# =========================
# File paths
RULES_PATH   = 'mba_meal/association_rules.csv'
FACT_PATH    = 'etl_dimensions/fact_transaction_dimension.csv'
PRODUCT_PATH = 'etl_dimensions/current_product_dimension.csv'

# Top-N MBA rules to summarize (use the literal first TOP_N rows)
TOP_N = 15

# Time-series settings
AGG_FREQ = 'QE'            # 'QE' Quarter End; ('MS' for Month Start, etc.)
SEASONAL_PERIODS = 4       # 4 for quarterly, 12 for monthly, etc.
HORIZON = 4                # number of future periods to forecast

# Holt–Winters smoothing (set OPTIMIZED=True to ignore the fixed alphas)
OPTIMIZED = False
HW_ALPHA = 0.2
HW_BETA  = 0.2
HW_GAMMA = 0.2

# Pricing / discount (to mirror the in-depth file, set FORCE_NEW_PRICE to that same value)
DISCOUNT_PCT   = 0.10        # used only when FORCE_NEW_PRICE is None
FORCE_NEW_PRICE = None       # e.g., 392.4 (to match your in-depth run)

# Output
OUTPUT_CSV = 'mba_meal/bundle_holtwinters_summary.csv'

# =========================
# Utils
# =========================
def cap(s: str, n: int = 50) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[:n-1] + "…"

def require_columns(df: pd.DataFrame, cols: list[str], df_name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"Error: {df_name} is missing required columns: {missing}")
        sys.exit(1)

def resolve_ids_by_exact_names(rule_row: pd.Series, product_df: pd.DataFrame) -> tuple[str | None, str | None]:
    """
    Aligns with the in-depth script: resolve by EXACT product_name match only.
    """
    nameA = rule_row.get('antecedents_names', None)
    nameB = rule_row.get('consequents_names', None)
    if nameA is None or nameB is None:
        return None, None
    a_match = product_df.loc[product_df['product_name'] == str(nameA)]
    b_match = product_df.loc[product_df['product_name'] == str(nameB)]
    if a_match.empty or b_match.empty:
        return None, None
    return str(a_match['product_id'].iloc[0]), str(b_match['product_id'].iloc[0])

def build_ts_receipt_presence(lines: pd.DataFrame, freq: str) -> pd.Series:
    """
    Collapse line items to a receipt-level presence time series at freq.
    Counts the number of receipts containing the item.
    (No asfreq padding — matches the in-depth script.)
    """
    if lines.empty:
        return pd.Series(dtype=float)
    rec = lines.groupby('Receipt No').agg(Date=('Date', 'first'))
    rec['Date'] = pd.to_datetime(rec['Date'], errors='coerce')
    ts = rec.groupby(pd.Grouper(key='Date', freq=freq)).size().astype(float)
    ts.index = pd.to_datetime(ts.index)
    return ts

def demand_points_for_pair(fact_df: pd.DataFrame, id_a: str, id_b: str, freq: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build price-quantity pairs for the bundle using receipts that contain BOTH items.
    Also return the bundle baseline time series (count of such receipts per period).
    """
    sub = fact_df[fact_df['Product ID'].astype(str).isin([str(id_a), str(id_b)])]
    receipt_products = sub.groupby('Receipt No')['Product ID'].apply(lambda s: set(s.astype(str)))
    both_receipts_idx = receipt_products[receipt_products.apply(lambda s: (str(id_a) in s) and (str(id_b) in s))].index
    working_df = fact_df[fact_df['Receipt No'].isin(both_receipts_idx)]

    # Demand by price point
    receipt_summary = working_df.groupby('Receipt No').agg(
        Combined_Price=('Line Total', 'sum'),
        Date=('Date', 'first')
    )
    price_qty = (receipt_summary
                 .groupby('Combined_Price')
                 .agg(Num_Transactions=('Date', 'count'))
                 .reset_index()
                 .sort_values('Combined_Price'))

    # Baseline bundle series
    receipt_summary['Date'] = pd.to_datetime(receipt_summary['Date'], errors='coerce')
    bundle_ts = receipt_summary.groupby(pd.Grouper(key='Date', freq=freq)).size().astype(float)
    bundle_ts.index = pd.to_datetime(bundle_ts.index)

    return price_qty, bundle_ts

def constant_elasticity(price_qty_df: pd.DataFrame) -> tuple[float, float]:
    """
    Log–log regression: log(Q) = intercept + elasticity * log(P)
    Returns (elasticity, intercept). If insufficient data, returns (0.0, 0.0).
    """
    df = price_qty_df[(price_qty_df['Combined_Price'] > 0) &
                      (price_qty_df['Num_Transactions'] > 0)].copy()
    if len(df) < 2:
        return 0.0, 0.0
    df['log_P'] = np.log(df['Combined_Price'])
    df['log_Q'] = np.log(df['Num_Transactions'])
    reg = LinearRegression().fit(df[['log_P']], df['log_Q'])
    return float(reg.coef_[0]), float(reg.intercept_)

def _last_index_or_min(ts: pd.Series) -> pd.Timestamp:
    return ts.index[-1] if (ts is not None and not ts.empty) else pd.Timestamp.min

def _next_step_offset(freq: str):
    if freq.upper().startswith('Q'):
        return pd.offsets.QuarterEnd()
    elif freq.upper().startswith('M'):
        return pd.offsets.MonthEnd()
    else:
        return pd.tseries.frequencies.to_offset(freq)

def build_common_fc_index(a_ts: pd.Series, b_ts: pd.Series, bundle_ts: pd.Series,
                          freq: str, horizon: int) -> pd.DatetimeIndex:
    step = _next_step_offset(freq)
    latest_actual = max([_last_index_or_min(bundle_ts),
                         _last_index_or_min(a_ts),
                         _last_index_or_min(b_ts)])
    return pd.date_range(start=latest_actual + step, periods=horizon, freq=freq)

def fit_and_forecast_to_index(series: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """
    Fit Holt–Winters to 'series' and forecast exactly onto 'idx'.
    Return RAW forecast (may contain negatives). We will clip later, after elasticity, to
    mirror the in-depth script.
    """
    if series is None or series.empty:
        return pd.Series(0.0, index=idx, dtype=float)

    # Ensure datetime index (no asfreq padding)
    series = series.copy()
    series.index = pd.to_datetime(series.index)

    model = ExponentialSmoothing(
        series, trend='add', seasonal='add',
        seasonal_periods=SEASONAL_PERIODS, initialization_method="estimated"
    )
    fitted = (model.fit(optimized=True) if OPTIMIZED else
              model.fit(smoothing_level=HW_ALPHA,
                        smoothing_trend=HW_BETA,
                        smoothing_seasonal=HW_GAMMA,
                        optimized=False))
    fc = fitted.forecast(len(idx))
    fc.index = idx
    return fc  # RAW (can be < 0)

def safe_price(product_df: pd.DataFrame, product_id: str) -> float:
    row = product_df.loc[product_df['product_id'].astype(str) == str(product_id)]
    return float(row['Price'].iloc[0]) if not row.empty else 0.0

def safe_name(product_df: pd.DataFrame, product_id: str) -> str:
    row = product_df.loc[product_df['product_id'].astype(str) == str(product_id)]
    return str(row['product_name'].iloc[0]) if not row.empty else f"(id:{product_id})"

# =========================
# Core per-bundle computation
# =========================
def compute_bundle_summary_aligned(product_df: pd.DataFrame,
                                   fact_df: pd.DataFrame,
                                   id_a: str, id_b: str,
                                   freq: str,
                                   horizon: int) -> dict:
    # Lines for each product
    a_lines = fact_df[fact_df['Product ID'].astype(str) == str(id_a)]
    b_lines = fact_df[fact_df['Product ID'].astype(str) == str(id_b)]

    # Demand points + bundle baseline series
    price_qty, bundle_ts = demand_points_for_pair(fact_df, id_a, id_b, freq)
    elasticity, _intercept = constant_elasticity(price_qty)

    # Individual histories
    a_ts = build_ts_receipt_presence(a_lines, freq)
    b_ts = build_ts_receipt_presence(b_lines, freq)

    series_len = int(max(len(bundle_ts), len(a_ts), len(b_ts)))

    # Common forecast index (exactly like the in-depth script)
    COMMON_FC_INDEX = build_common_fc_index(a_ts, b_ts, bundle_ts, freq, horizon)

    # Forecast RAW (may be negative)
    bundle_fc_raw = fit_and_forecast_to_index(bundle_ts, COMMON_FC_INDEX)
    a_fc_all_raw  = fit_and_forecast_to_index(a_ts, COMMON_FC_INDEX)
    b_fc_all_raw  = fit_and_forecast_to_index(b_ts, COMMON_FC_INDEX)

    # Clamp non-negative for downstream math (same as in-depth)
    bundle_fc = bundle_fc_raw.clip(lower=0)
    a_fc_all  = a_fc_all_raw.clip(lower=0)
    b_fc_all  = b_fc_all_raw.clip(lower=0)

    # Prices
    price_a = safe_price(product_df, id_a)
    price_b = safe_price(product_df, id_b)
    current_price = price_a + price_b

    # New price (forced or by discount)
    if FORCE_NEW_PRICE is not None:
        new_price = float(FORCE_NEW_PRICE)
    else:
        new_price = current_price * (1.0 - float(DISCOUNT_PCT))

    # Demand multiplier (power model), then adjust RAW bundle → then clip
    if current_price > 0 and new_price > 0:
        demand_multiplier = (new_price / current_price) ** elasticity
    else:
        demand_multiplier = 1.0
    bundle_fc_adj = (bundle_fc_raw * demand_multiplier).clip(lower=0)

    # Cannibalization (use non-negative baseline bundle units)
    cannibalization_units = bundle_fc
    a_fc_after = (a_fc_all - cannibalization_units).clip(lower=0)
    b_fc_after = (b_fc_all - cannibalization_units).clip(lower=0)

    # Revenue (per COMMON_FC_INDEX)
    rev_a_before = (a_fc_all * price_a).sum()
    rev_b_before = (b_fc_all * price_b).sum()
    rev_before_total = float(rev_a_before + rev_b_before)

    rev_a_after = (a_fc_after * price_a).sum()
    rev_b_after = (b_fc_after * price_b).sum()
    rev_bundle_after = (bundle_fc_adj * new_price).sum()
    rev_after_total = float(rev_a_after + rev_b_after + rev_bundle_after)

    impact_abs = float(rev_after_total - rev_before_total)

    # Break-even (if negative impact)
    break_even_units = int(np.ceil((-impact_abs) / new_price)) if (impact_abs < 0 and new_price > 0) else 0

    # Names (uncapped for CSV)
    name_a = safe_name(product_df, id_a)
    name_b = safe_name(product_df, id_b)

    return {
        'product_id_1': str(id_a),
        'product_id_2': str(id_b),
        'product name 1': name_a,
        'product name 2': name_b,
        'Price': round(current_price, 2),
        'Discounted_Price': round(new_price, 2),
        'Revenue_Before': round(rev_before_total, 2),
        'Revenue_After': round(rev_after_total, 2),
        'Revenue_Impact': round(impact_abs, 2),
        'BreakEven_Bundle_Units': break_even_units,
        'Series_Length': series_len,
        'Forecast_Horizon': int(horizon),
    }

# =========================
# Main
# =========================
def main():
    # Load data
    try:
        rules_df = pd.read_csv(RULES_PATH)
        fact_df = pd.read_csv(FACT_PATH)
        product_df = pd.read_csv(PRODUCT_PATH)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Column sanity
    require_columns(rules_df, ['antecedents_names', 'consequents_names'], 'RULES file')
    require_columns(fact_df,  ['Product ID', 'Receipt No', 'Line Total', 'Date'], 'FACT file')
    require_columns(product_df, ['product_id', 'product_name', 'Price'], 'PRODUCT file')

    # Header
    forced = FORCE_NEW_PRICE is not None
    disc_str = f"forced price={FORCE_NEW_PRICE:.2f}" if forced else f"{DISCOUNT_PCT*100:.1f}%"
    print(f"=== Bundle Forecast Summary (TOP {TOP_N} MBA rules; product_id bundles) ===")
    print(f"Discount: {disc_str} | Freq: {AGG_FREQ} | Horizon: {HORIZON} | Seasonal periods: {SEASONAL_PERIODS}")

    rows = []
    n = min(TOP_N, len(rules_df))
    for i in range(n):
        rule = rules_df.iloc[i]
        id_a, id_b = resolve_ids_by_exact_names(rule, product_df)
        if not id_a or not id_b:
            # Align with in-depth behavior: if names can't be resolved, skip
            continue
        try:
            row = compute_bundle_summary_aligned(product_df, fact_df, id_a, id_b,
                                                 AGG_FREQ, HORIZON)
            rows.append(row)
        except Exception as e:
            print(f"[WARN] Skipping rule row {i} ({id_a}, {id_b}) due to error: {e}")

    if not rows:
        print("No bundles resolved; nothing to summarize.")
        sys.exit(0)

    # Exact column order
    out_cols = [
        'product_id_1','product_id_2','product name 1','product name 2',
        'Price','Discounted_Price','Revenue_Before','Revenue_After',
        'Revenue_Impact','BreakEven_Bundle_Units','Series_Length','Forecast_Horizon'
    ]
    out_df = pd.DataFrame(rows)[out_cols]

    # Console table with capped names
    print_cols = [
        'product name 1','product name 2','Price','Discounted_Price',
        'Revenue_Before','Revenue_After','Revenue_Impact','BreakEven_Bundle_Units'
    ]
    printable = out_df.copy()
    printable['product name 1'] = printable['product name 1'].map(lambda s: cap(s, 50))
    printable['product name 2'] = printable['product name 2'].map(lambda s: cap(s, 50))
    print("\n--- Summary (names capped to 50 chars) ---")
    print(printable[print_cols].to_string(index=False))

    # Export CSV
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"\nCSV saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
