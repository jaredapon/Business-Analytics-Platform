import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
import sys

# =========================
# USER CONFIGURATION
# =========================
# File paths
RULES_PATH   = 'mba_meal/association_rules.csv'
FACT_PATH    = 'etl_dimensions/fact_transaction_dimension.csv'
PRODUCT_PATH = 'etl_dimensions/current_product_dimension.csv'

# Bundle selection (row in association_rules.csv to analyze)
BUNDLE_ROW = 1

# Time-series settings
AGG_FREQ = 'QE'            # 'QE' = Quarter End; (e.g., 'MS' for Month Start)
SEASONAL_PERIODS = 4       # 4 quarters in a year (use 12 for monthly)
HORIZON = 4                # number of future periods to forecast

# Holt-Winters smoothing (set OPTIMIZED=True to ignore the fixed alphas)
OPTIMIZED = False
HW_ALPHA = 0.2
HW_BETA  = 0.2
HW_GAMMA = 0.2

# Price scenario
NEW_PRICE = 419.4         # promotional bundle price you want to test

# =========================
# LOAD BUNDLE FROM RULES
# =========================
try:
    rules_df = pd.read_csv(RULES_PATH)
    if 'antecedents_names' not in rules_df.columns or 'consequents_names' not in rules_df.columns:
        print("Error: Required columns not found in association_rules.csv")
        sys.exit()

    if not (0 <= BUNDLE_ROW < len(rules_df)):
        print(f"Error: BUNDLE_ROW {BUNDLE_ROW} out of range (0..{len(rules_df)-1}).")
        sys.exit()

    bundle = rules_df.iloc[BUNDLE_ROW]
    product_a_name = bundle['antecedents_names']
    product_b_name = bundle['consequents_names']
    print(f"Analyzing bundle: '{product_a_name}' and '{product_b_name}'")

except FileNotFoundError:
    print(f"Error: '{RULES_PATH}' not found.")
    sys.exit()
except Exception as e:
    print(f"Error reading association rules: {e}")
    sys.exit()

# =========================
# LOAD FACTS & PRODUCTS
# =========================
try:
    fact_df = pd.read_csv(FACT_PATH)
    product_df = pd.read_csv(PRODUCT_PATH)
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    sys.exit()

# Map product names -> IDs
product_a_match = product_df[product_df['product_name'] == product_a_name]
product_b_match = product_df[product_df['product_name'] == product_b_name]
if product_a_match.empty or product_b_match.empty:
    print(f"Could not find product IDs for '{product_a_name}' or '{product_b_name}'.")
    sys.exit()

product_a_id = product_a_match['product_id'].iloc[0]
product_b_id = product_b_match['product_id'].iloc[0]

# =========================
# BUNDLE DEMAND BY PRICE POINT (all receipts that contain BOTH)
# =========================
pair_transactions = fact_df[fact_df['Product ID'].isin([product_a_id, product_b_id])]
receipt_products = pair_transactions.groupby('Receipt No')['Product ID'].apply(set)
receipts_with_both = receipt_products[
    receipt_products.apply(lambda s: (product_a_id in s) and (product_b_id in s))
].index

# Use all receipts that contain both items (no "only pair" filter)
working_df = fact_df[fact_df['Receipt No'].isin(receipts_with_both)]

receipt_summary = working_df.groupby('Receipt No').agg(
    Combined_Price=('Line Total', 'sum'),
    Date=('Date', 'first')
)

demand_summary = (receipt_summary
                  .groupby('Combined_Price')
                  .agg(Num_Transactions=('Date', 'count'))
                  .reset_index()
                  .sort_values('Combined_Price'))

print(f"\nDemand Analysis for {product_a_name} + {product_b_name} bundle:")
print(demand_summary)

# =========================
# TIME SERIES (AGG_FREQ)
# =========================
receipt_summary['Date'] = pd.to_datetime(receipt_summary['Date'], errors='coerce')
bundle_sales_ts = receipt_summary.groupby(pd.Grouper(key='Date', freq=AGG_FREQ)).size()

# Individual product histories (ALL receipts)
def build_ts_all(lines: pd.DataFrame) -> pd.Series:
    if lines.empty:
        return pd.Series(dtype=float)
    rec = lines.groupby('Receipt No').agg(Date=('Date', 'first'))
    rec['Date'] = pd.to_datetime(rec['Date'], errors='coerce')
    return rec.groupby(pd.Grouper(key='Date', freq=AGG_FREQ)).size()

a_lines_all = fact_df[fact_df['Product ID'] == product_a_id]
b_lines_all = fact_df[fact_df['Product ID'] == product_b_id]

a_ts_all = build_ts_all(a_lines_all)
b_ts_all = build_ts_all(b_lines_all)

if bundle_sales_ts.empty and (a_ts_all.empty and b_ts_all.empty):
    print("No sales found for forecasting.")
    sys.exit()

# =========================
# CONSTANT ELASTICITY (LOG-LOG) FOR BUNDLE
# =========================
log_df = demand_summary[(demand_summary['Combined_Price'] > 0) &
                        (demand_summary['Num_Transactions'] > 0)].copy()
log_df['log_Price'] = np.log(log_df['Combined_Price'])
log_df['log_Quantity'] = np.log(log_df['Num_Transactions'])

if len(log_df) >= 2:
    X = log_df[['log_Price']]
    y = log_df['log_Quantity']
    reg = LinearRegression().fit(X, y)
    elasticity = reg.coef_[0]
    intercept = reg.intercept_
else:
    print("\nNot enough positive (price, quantity) points for log-log regression.")
    elasticity = 0.0
    intercept = 0.0

print(f"\nEstimated Price Elasticity of Demand (constant elasticity): {elasticity:.3f}")
print(f"Regression equation: log_Quantity = {intercept:.3f} + ({elasticity:.3f}) * log_Price")

# =========================
# PRICE SETTING SCENARIO (BUNDLE) — POWER FORMULA
# =========================
current_price_a = float(product_a_match['Price'].iloc[0])
current_price_b = float(product_b_match['Price'].iloc[0])
current_price = current_price_a + current_price_b
new_price = float(NEW_PRICE)

print(f"\n--- PRICE setting FORECAST ---")
print(f"Current price: Php {current_price:.2f}")
print(f"New promotional price: Php {new_price:.2f}")
print(f"Price change: {((new_price - current_price) / current_price * 100):.1f}%")

# Power model multiplier
if current_price <= 0 or new_price <= 0:
    print("Warning: Non-positive price encountered; cannot apply power formula. Falling back to no change.")
    demand_multiplier = 1.0
else:
    price_ratio = new_price / current_price
    demand_multiplier = price_ratio ** elasticity

expected_demand_change_pct = (demand_multiplier - 1.0) * 100.0
print(f"Expected demand change in bundle units (power model): {expected_demand_change_pct:.1f}%")

# =========================
# COMMON FORECAST INDEX (IMPLEMENTATION A)
# =========================
# Determine the step (same as your plotting offset)
if AGG_FREQ.upper().startswith('Q'):
    step = pd.offsets.QuarterEnd()
elif AGG_FREQ.upper().startswith('M'):
    step = pd.offsets.MonthEnd()
else:
    step = pd.tseries.frequencies.to_offset(AGG_FREQ)

def last_index_or_min(ts: pd.Series):
    return ts.index[-1] if (ts is not None and not ts.empty) else pd.Timestamp.min

latest_actual = max([last_index_or_min(bundle_sales_ts),
                     last_index_or_min(a_ts_all),
                     last_index_or_min(b_ts_all)])

COMMON_FC_INDEX = pd.date_range(start=latest_actual + step, periods=HORIZON, freq=AGG_FREQ)

def fit_and_forecast_to_index(series: pd.Series, label: str, idx: pd.DatetimeIndex) -> pd.Series:
    """Fit Holt-Winters to 'series' and forecast exactly onto 'idx' (RAW, may be negative)."""
    if series is None or series.empty:
        print(f"\nNo sales found for {label}. Returning zeros on common index.")
        return pd.Series(0.0, index=idx)
    model = ExponentialSmoothing(
        series, trend='add', seasonal='add',
        seasonal_periods=SEASONAL_PERIODS, initialization_method="estimated"
    )
    if OPTIMIZED:
        fitted = model.fit(optimized=True)
    else:
        fitted = model.fit(
            smoothing_level=HW_ALPHA,
            smoothing_trend=HW_BETA,
            smoothing_seasonal=HW_GAMMA,
            optimized=False
        )
    fc = fitted.forecast(len(idx))
    fc.index = idx
    return fc

# Forecast RAW series (can be negative)
bundle_fc_raw     = fit_and_forecast_to_index(bundle_sales_ts, "Bundle (baseline)", COMMON_FC_INDEX)
a_fc_all_raw      = fit_and_forecast_to_index(a_ts_all, f"{product_a_name} (ALL)", COMMON_FC_INDEX)
b_fc_all_raw      = fit_and_forecast_to_index(b_ts_all, f"{product_b_name} (ALL)", COMMON_FC_INDEX)

# Apply non-negativity for plotting and all downstream math
bundle_fc     = bundle_fc_raw.clip(lower=0)
bundle_fc_adj = (bundle_fc_raw * demand_multiplier).clip(lower=0)  # adjust then clamp

a_fc_all = a_fc_all_raw.clip(lower=0)
b_fc_all = b_fc_all_raw.clip(lower=0)

# =========================
# CANNIBALIZATION MODEL (on COMMON_FC_INDEX)
#   Use non-negative baseline bundle units to avoid subtracting negatives.
#   After = baseline - baseline_bundle_units, floored at 0.
# =========================
cannibalization_units = bundle_fc  # non-negative
a_fc_after_aligned = (a_fc_all - cannibalization_units).clip(lower=0)
b_fc_after_aligned = (b_fc_all - cannibalization_units).clip(lower=0)

# =========================
# REVENUE IMPACT ANALYSIS (on COMMON_FC_INDEX)
#   Always use clamped (non-negative) series
# =========================
def revenue_forecast(series: pd.Series, price: float) -> pd.Series:
    # series is already clamped to non-negative above
    return series * price

price_a = float(product_a_match['Price'].iloc[0])
price_b = float(product_b_match['Price'].iloc[0])

rev_a_before = revenue_forecast(a_fc_all, price_a)
rev_b_before = revenue_forecast(b_fc_all, price_b)

rev_a_after  = revenue_forecast(a_fc_after_aligned, price_a)
rev_b_after  = revenue_forecast(b_fc_after_aligned, price_b)

# Bundle revenue:
rev_bundle_before = pd.Series(0.0, index=COMMON_FC_INDEX)         # no bundle pre-intro
rev_bundle_after  = bundle_fc_adj * new_price                      # promo bundle revenue (already clamped)

# Totals
indiv_rev_before = {"A": float(rev_a_before.sum()), "B": float(rev_b_before.sum())}
indiv_rev_after  = {"A": float(rev_a_after.sum()),  "B": float(rev_b_after.sum())}

overall_before = indiv_rev_before["A"] + indiv_rev_before["B"]
overall_after  = indiv_rev_after["A"] + indiv_rev_after["B"] + float(rev_bundle_after.sum())

# --- PRINT RESULTS ---
print("\n--- DETAILED REVENUE IMPACT ANALYSIS (Common index, power elasticity, non-negative forecasts) ---")
bundle_before_total = float(rev_bundle_before.sum())   # 0.0
bundle_after_total  = float(rev_bundle_after.sum())

print("Individual Revenue Forecast BEFORE Bundling/Cannibalization:")
print(f"  {product_a_name}: Php {indiv_rev_before['A']:.2f}")
print(f"  {product_b_name}: Php {indiv_rev_before['B']:.2f}")
print(f"  BUNDLE (status-quo price Php {current_price:.2f}): Php {bundle_before_total:.2f}")

print("\nIndividual Revenue Forecast AFTER Bundling/Cannibalization:")
print(f"  {product_a_name}: Php {indiv_rev_after['A']:.2f}")
print(f"  {product_b_name}: Php {indiv_rev_after['B']:.2f}")
print(f"  BUNDLE (promo price Php {new_price:.2f}): Php {bundle_after_total:.2f}")

print(f"\nOverall Revenue BEFORE (A + B): Php {overall_before:.2f}")
print(f"Overall Revenue AFTER (A_after + B_after + Bundle_after): Php {overall_after:.2f}")

impact_abs = overall_after - overall_before
impact_pct = (impact_abs / overall_before * 100.0) if overall_before != 0 else float('nan')
print(f"Revenue Impact: Php {impact_abs:.2f} ({impact_pct:.1f}%)")

# =========================
# BREAK-EVEN BUNDLE UNITS (to offset any overall revenue loss)
# =========================
overall_before_series = rev_a_before + rev_b_before
overall_after_series  = rev_a_after + rev_b_after + rev_bundle_after
revenue_gap_series = (overall_before_series - overall_after_series).clip(lower=0)  # gap can't be negative

if new_price <= 0:
    print("\nBreak-even analysis skipped: NEW_PRICE must be > 0.")
else:
    needed_bundles_series = (revenue_gap_series / new_price)
    needed_bundles_ceiling = np.ceil(needed_bundles_series)

    total_gap = float(revenue_gap_series.sum())
    total_needed_bundles = int(np.ceil(max(0.0, total_gap / new_price)))

    print("\n--- BREAK-EVEN (Incremental Bundle Units Required) ---")
    if impact_abs < 0:
        print(f"Overall revenue shortfall: Php {(-impact_abs):.2f}")
        print(f"Bundle price used for break-even: Php {new_price:.2f}")
        print(f"Total incremental bundle units required to break even: {total_needed_bundles:,}")
    else:
        print("No break-even needed (overall revenue is not negative).")
        print(f"Surplus: Php {impact_abs:.2f}")

    be_df = pd.DataFrame({
        'Period': COMMON_FC_INDEX,
        'Revenue_Gap': revenue_gap_series.values.round(2),
        'Bundle_Units_Required_Ceil': needed_bundles_ceiling.astype('Int64').values
    })
    print("\nPer-period incremental bundle units required (rounded up):")
    print(be_df.to_string(index=False))

# =========================
# FUNCTION: detect trend
#   Use clamped series to match what we plot.
# =========================
def detect_trend(series: pd.Series, label: str) -> str:
    """Determine trend direction (upward, downward, flat) from first vs last values."""
    if series is None or len(series) < 2:
        return f"{label}: No data"
    first_val, last_val = series.iloc[0], series.iloc[-1]
    if last_val > first_val * 1.05:   # >5% increase
        return f"{label}: Upward trend"
    elif last_val < first_val * 0.95: # >5% decrease
        return f"{label}: Downward trend"
    else:
        return f"{label}: Relatively flat trend"

# =========================
# TREND ANALYSIS (clamped)
# =========================
print("\n--- TREND ANALYSIS (clamped series) ---")
print(detect_trend(bundle_fc, "Baseline bundle forecast"))
print(detect_trend(bundle_fc_adj, "Adjusted bundle forecast (power model)"))
print(detect_trend(a_fc_all, f"{product_a_name} baseline forecast"))
print(detect_trend(a_fc_after_aligned, f"{product_a_name} after cannibalization"))
print(detect_trend(b_fc_all, f"{product_b_name} baseline forecast"))
print(detect_trend(b_fc_after_aligned, f"{product_b_name} after cannibalization"))

# =========================
# PLOTS (ALL IN ONE FRAME) — all forecasts on COMMON_FC_INDEX
#   Plot clamped series so negatives appear as 0 but the seasonal shape is preserved above zero.
# =========================
fig, axes = plt.subplots(3, 1, figsize=(14, 16), sharex=False)
fig.subplots_adjust(hspace=0.2)  # set vertical spacing explicitly

# 1) Bundle subplot (historical + forecasts on common index)
axes[0].plot(bundle_sales_ts.index, bundle_sales_ts.values,
             label=f'Historical ({AGG_FREQ})', marker='o')
axes[0].plot(COMMON_FC_INDEX, bundle_fc.values,
             label='Baseline Bundle Forecast (units, status-quo price)', linestyle='--', marker='s')
axes[0].plot(COMMON_FC_INDEX, bundle_fc_adj.values,
             label=f'Adjusted Bundle Forecast at Php {new_price:.0f}', linestyle='--', marker='s')
# connector from last actual to first forecast (baseline)
if not bundle_sales_ts.empty:
    axes[0].plot([bundle_sales_ts.index[-1], COMMON_FC_INDEX[0]],
                 [bundle_sales_ts.values[-1], bundle_fc.values[0]],
                 color='gray', linestyle=':', linewidth=2)
axes[0].set_title(
    f'Bundle Forecast: {product_a_name} + {product_b_name}\n'
    f'Elasticity-adj demand change (power): {expected_demand_change_pct:.1f}% | '
    f'Old price: Php {current_price:.2f} → New price: Php {new_price:.2f}'
)
axes[0].set_ylabel('Bundle Sales (units)')
axes[0].grid(True)
axes[0].legend(loc='best')

# 2) Product A subplot (baseline vs after) on common index
if not a_ts_all.empty:
    axes[1].plot(a_ts_all.index, a_ts_all.values,
                 label=f'Historical {product_a_name} ({AGG_FREQ})', marker='o')
axes[1].plot(COMMON_FC_INDEX, a_fc_all.values,
             label='Baseline Forecast (pre-bundle, clamped ≥0)', linestyle='--', marker='s')
axes[1].plot(COMMON_FC_INDEX, a_fc_after_aligned.values,
             label='After Cannibalization (subtract baseline bundle units)', linestyle='--', marker='^')
if not a_ts_all.empty:
    axes[1].plot([a_ts_all.index[-1], COMMON_FC_INDEX[0]],
                 [a_ts_all.values[-1], a_fc_all.values[0]],
                 color='gray', linestyle=':', linewidth=2)
axes[1].set_title(f'Product A Forecasts: {product_a_name} — Baseline vs After Cannibalization')
axes[1].set_ylabel('Receipts with Product A')
axes[1].grid(True)
axes[1].legend(loc='best')

# 3) Product B subplot (baseline vs after) on common index
if not b_ts_all.empty:
    axes[2].plot(b_ts_all.index, b_ts_all.values,
                 label=f'Historical {product_b_name} ({AGG_FREQ})', marker='o')
axes[2].plot(COMMON_FC_INDEX, b_fc_all.values,
             label='Baseline Forecast (pre-bundle, clamped ≥0)', linestyle='--', marker='s')
axes[2].plot(COMMON_FC_INDEX, b_fc_after_aligned.values,
             label='After Cannibalization (subtract baseline bundle units)', linestyle='--', marker='^')
if not b_ts_all.empty:
    axes[2].plot([b_ts_all.index[-1], COMMON_FC_INDEX[0]],
                 [b_ts_all.values[-1], b_fc_all.values[0]],
                 color='gray', linestyle=':', linewidth=2)
axes[2].set_title(f'Product B Forecasts: {product_b_name} — Baseline vs After Cannibalization')
axes[2].set_xlabel('Period')
axes[2].set_ylabel('Receipts with Product B')
axes[2].grid(True)
axes[2].legend(loc='best')

plt.show()