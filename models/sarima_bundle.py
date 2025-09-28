import argparse
import sys
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

"""Seasonal ARIMA (SARIMA) for a 2‑product bundle.

Mirrors functionality of arima_bundle.py but allows seasonal components.

Bundle Demand Modes:
  occurrence : each qualifying receipt counts as 1 (default)
  min_pair   : per receipt min(qty A, qty B)
  sum_qty    : per receipt qty A + qty B (only co-present items)

Receipt Filtering:
  --only-pair : restrict to receipts containing *exactly* the two products
  default     : allow extra items on the receipt as long as both appear

Seasonality:
  Provide seasonal (P,D,Q,s) explicitly; grid search only over non-seasonal (p,q).
  Non-seasonal differencing d chosen automatically via ADF unless --force-d supplied.

Outputs:
  - In-sample error metrics (MAE, MSE, RMSE, MAPE excl zeros, MASE, AIC, BIC)
  - Forecast table for horizon
  - Interpretation block with qualitative diagnostics
  - Terminal only (no file export) to be consistent with other models
"""

# --------------------------- CLI --------------------------- #


def parse_args():
    p = argparse.ArgumentParser(
        description="SARIMA forecasting for a product bundle (pair).")
    # Data sourcing
    p.add_argument('--rules-path', default='mba_drinks/association_rules.csv',
                   help='Association rules CSV.')
    p.add_argument('--rules-set', choices=['drinks', 'foods'],
                   default=None, help='Shortcut: use mba_drinks or mba_foods rules.')
    p.add_argument('--use-row', type=int, default=0,
                   help='Row in rules file (antecedent + consequent first tokens).')
    p.add_argument('--fact-path', default='etl_dimensions/fact_transaction_dimension.csv',
                   help='Fact transactions CSV.')
    p.add_argument('--product-path', default='etl_dimensions/current_product_dimension.csv',
                   help='Product dimension CSV.')
    # Demand construction
    p.add_argument('--freq', default='ME',
                   help='Aggregation frequency (e.g. D, W, ME=month end, QE=quarter end).')
    p.add_argument('--demand-mode', choices=['occurrence', 'min_pair',
                   'sum_qty'], default='occurrence', help='Bundle demand definition.')
    p.add_argument('--only-pair', action='store_true',
                   help='Require receipt contains *only* the two products.')
    # Forecast config
    p.add_argument('--forecast-horizon', type=int, default=4,
                   help='Future periods to forecast.')
    # Non-seasonal search
    p.add_argument('--max-p', type=int, default=2)
    p.add_argument('--max-q', type=int, default=2)
    p.add_argument('--max-d', type=int, default=2)
    p.add_argument('--force-d', type=int, default=None,
                   help='Override automatic differencing.')
    # Seasonal order (fixed during grid search over p,q)
    p.add_argument('--seasonal-P', type=int, default=1)
    p.add_argument('--seasonal-D', type=int, default=0)
    p.add_argument('--seasonal-Q', type=int, default=1)
    p.add_argument('--seasonal-s', type=int, default=12,
                   help='Seasonal period (e.g. 7=weekly season on daily, 12=monthly, 4=quarterly).')
    # Warnings / thresholds
    p.add_argument('--min-length', type=int, default=8,
                   help='Warn if non-zero periods fewer than this.')
    # Misc
    p.add_argument('--plot', action='store_true')
    return p.parse_args()


# --------------------------- Resolution --------------------------- #


def resolve_token_to_product_id(token: str, product_df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    if token is None:
        return None, None
    t = str(token).strip()
    cols = set(product_df.columns)
    candidates = pd.DataFrame()
    method = None
    if 'product_id' in cols:
        subset = product_df[product_df['product_id'].astype(str) == t]
        if not subset.empty:
            candidates = subset
            method = 'product_id'
    if method is None and 'product_name' in cols:
        subset = product_df[product_df['product_name'].astype(str) == t]
        if not subset.empty:
            candidates = subset
            method = 'product_name'
    if method is None and 'parent_sku' in cols:
        subset = product_df[product_df['parent_sku'].astype(str) == t]
        if not subset.empty:
            candidates = subset
            method = 'parent_sku'
    if candidates.empty:
        return None, None
    if len(candidates) > 1 and 'Price' in candidates.columns:
        candidates = candidates.sort_values('Price', ascending=False)
    pid = str(candidates.iloc[0]['product_id']
              ) if 'product_id' in candidates.columns else None
    return pid, method


# --------------------------- Stationarity & Metrics --------------------------- #


def adf_stationarity_test(ts: pd.Series):
    try:
        res = adfuller(ts.dropna())
        return {'p_value': res[1]}
    except Exception:
        return None


def determine_d(series: pd.Series, max_d: int) -> int:
    for d in range(0, max_d + 1):
        tmp = series.copy()
        for _ in range(d):
            tmp = tmp.diff()
        if tmp.dropna().empty:
            continue
        res = adf_stationarity_test(tmp)
        if res and res['p_value'] < 0.05:
            print(f"[STATIONARITY] d={d} p={res['p_value']:.4f} (stationary)")
            return d
        if res:
            print(
                f"[STATIONARITY] d={d} p={res['p_value']:.4f} (not stationary)")
    print(f"[STATIONARITY] Using max differencing d={max_d}")
    return max_d


def mase(actual: pd.Series, forecast: pd.Series, m: int = 1) -> float:
    if len(actual) <= m + 1:
        return float('nan')
    denom = np.mean(np.abs(actual[m:] - actual[:-m]))
    if denom == 0 or np.isnan(denom):
        return float('nan')
    return np.mean(np.abs(actual - forecast)) / denom


def classify_mape(mape: float) -> str:
    if np.isnan(mape):
        return "MAPE not computed"
    if mape < 10:
        return "Excellent"
    if mape < 20:
        return "Good"
    if mape < 50:
        return "Moderate"
    return "High (>50%)"


def interpret_bundle(ts: pd.Series, mae: float, rmse: float, mape: float, mase_val: float,
                     non_zero: int, zero_excluded: int, mode: str, seasonal_order: tuple) -> str:
    lines = []
    mean_demand = ts.mean() if len(ts) else float('nan')
    rmse_ratio = rmse / \
        mean_demand if mean_demand and mean_demand != 0 else float('nan')
    lines.append(f"- Demand mode: {mode} (how bundle units defined).")
    lines.append(f"- Seasonal order: {seasonal_order} (P,D,Q,s).")
    lines.append(f"- MAPE classification: {classify_mape(mape)}")
    if not np.isnan(rmse_ratio):
        lines.append(
            f"- RMSE/mean demand = {rmse_ratio:.2f}; <0.5 often acceptable.")
    if not np.isnan(mase_val):
        if mase_val < 1:
            lines.append(
                f"- MASE {mase_val:.2f} (<1) better than naive seasonal benchmark.")
        else:
            lines.append(
                f"- MASE {mase_val:.2f} (>=1) no improvement vs naive; consider aggregating or different mode.")
    else:
        lines.append(
            "- MASE not available (insufficient variation or length).")
    if zero_excluded:
        lines.append(
            f"- {zero_excluded} zero periods excluded from MAPE (intermittent demand).")
    if non_zero < 8:
        lines.append(
            f"- Only {non_zero} non-zero periods → fragile seasonal inference.")
    lines.append("- Validate with hold-out or rolling-origin for robustness.")
    return "\n".join(lines)


# --------------------------- Bundle Demand Construction --------------------------- #


def build_bundle_timeseries(fact_df: pd.DataFrame, pid_a: str, pid_b: str, freq: str,
                            mode: str, only_pair: bool) -> pd.Series:
    df = fact_df.copy()
    if 'Date' not in df.columns:
        return pd.Series(dtype=float)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Product ID'] = df['Product ID'].astype(str)

    receipt_sets = df.groupby('Receipt No')['Product ID'].apply(
        lambda s: set(s.astype(str)))
    if only_pair:
        valid_receipts = receipt_sets[receipt_sets.apply(
            lambda s: s == {pid_a, pid_b})].index
    else:
        valid_receipts = receipt_sets[receipt_sets.apply(
            lambda s: pid_a in s and pid_b in s)].index

    if len(valid_receipts) == 0:
        return pd.Series(dtype=float)

    sub = df[df['Receipt No'].isin(valid_receipts)]

    if mode == 'occurrence':
        per_receipt = sub.groupby('Receipt No').agg(Date=('Date', 'first'))
        per_receipt['Bundle_Demand'] = 1
    else:
        pivot_qty = sub.pivot_table(
            index='Receipt No', columns='Product ID', values='Qty', aggfunc='sum', fill_value=0)
        for col in [pid_a, pid_b]:
            if col not in pivot_qty.columns:
                pivot_qty[col] = 0
        if mode == 'min_pair':
            bundle_units = pivot_qty[[pid_a, pid_b]].min(axis=1)
        elif mode == 'sum_qty':
            bundle_units = pivot_qty[[pid_a, pid_b]].sum(axis=1)
        else:
            raise ValueError('Unknown demand mode')
        first_dates = sub.groupby('Receipt No')['Date'].first()
        per_receipt = pd.DataFrame(
            {'Date': first_dates, 'Bundle_Demand': bundle_units})
        per_receipt = per_receipt[per_receipt['Bundle_Demand'] > 0]

    ts = per_receipt.groupby(pd.Grouper(key='Date', freq=freq))[
        'Bundle_Demand'].sum().astype(float)
    return ts


# --------------------------- Grid Search --------------------------- #


def grid_search_sarima(series: pd.Series, d: int, max_p: int, max_q: int, seasonal_order: tuple):
    best = None
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            try:
                model = SARIMAX(series, order=(p, d, q), seasonal_order=seasonal_order,
                                enforce_stationarity=False, enforce_invertibility=False)
                fit = model.fit(disp=False)
                aic = fit.aic
                print(
                    f"  Tried SARIMA({p},{d},{q})x{seasonal_order} AIC={aic:.2f}")
                if best is None or aic < best[1]:
                    best = ((p, d, q), aic, fit)
            except Exception as e:
                print(f"  Failed SARIMA({p},{d},{q}): {e}")
    if best is None:
        print('[ERROR] No SARIMA models fit successfully.')
        sys.exit(1)
    print(f"[GRID] Best order={best[0]} AIC={best[1]:.2f}")
    return best[0], best[2]


# --------------------------- Main --------------------------- #


def main():
    args = parse_args()

    # rules-set shortcut
    if getattr(args, 'rules_set', None):
        args.rules_path = f"mba_{args.rules_set}/association_rules.csv"
        print(
            f"[INFO] Using rules set '{args.rules_set}' -> {args.rules_path}")

    # Load data
    try:
        rules_df = pd.read_csv(args.rules_path)
        fact_df = pd.read_csv(args.fact_path)
        product_df = pd.read_csv(args.product_path)
    except Exception as e:
        print(f"[ERROR] Loading data: {e}")
        sys.exit(1)

    if args.use_row >= len(rules_df):
        print(
            f"[ERROR] use-row {args.use_row} out of range (rows={len(rules_df)})")
        sys.exit(1)

    row = rules_df.iloc[args.use_row]
    token_a = str(row['antecedents_names']).split(',')[0].strip()
    token_b = str(row['consequents_names']).split(',')[0].strip()
    print(f"[PAIR] Tokens: '{token_a}' + '{token_b}'")

    pid_a, method_a = resolve_token_to_product_id(token_a, product_df)
    pid_b, method_b = resolve_token_to_product_id(token_b, product_df)
    if not pid_a or not pid_b:
        print('[ERROR] Could not resolve both tokens to product_ids.')
        sys.exit(1)
    print(
        f"[RESOLVE] {token_a}->{pid_a} ({method_a}); {token_b}->{pid_b} ({method_b})")

    ts = build_bundle_timeseries(
        fact_df, pid_a, pid_b, args.freq, args.demand_mode, args.only_pair)
    if ts.empty:
        print('[INFO] Empty bundle series; aborting.')
        sys.exit(0)

    non_zero = (ts > 0).sum()
    if non_zero < args.min_length:
        print(
            f"[WARN] Only {non_zero} non-zero periods (< {args.min_length}); model may be unreliable.")

    # Differencing
    if args.force_d is not None:
        d = args.force_d
        print(f"[INFO] Using forced d={d}")
    else:
        d = determine_d(ts, args.max_d)

    seasonal_order = (args.seasonal_P, args.seasonal_D,
                      args.seasonal_Q, args.seasonal_s)
    order, fit = grid_search_sarima(
        ts, d, args.max_p, args.max_q, seasonal_order)
    print(
        f"\n[MODEL] Best SARIMA order={order} x {seasonal_order} AIC={fit.aic:.2f}")

    # In-sample
    try:
        insample = fit.predict(start=ts.index[0], end=ts.index[-1])
    except Exception:
        insample = fit.fittedvalues
    insample = insample.reindex(ts.index)

    mae = mean_absolute_error(ts, insample)
    mse = mean_squared_error(ts, insample)
    rmse = np.sqrt(mse)
    actual_nonzero = ts.replace(0, np.nan)
    if actual_nonzero.notna().any():
        mape = (np.abs((ts - insample)/actual_nonzero)).mean() * 100
        zero_excluded = (ts == 0).sum()
    else:
        mape = float('nan')
        zero_excluded = 0
    mase_val = mase(ts, insample, m=1)

    print("\n[IN-SAMPLE METRICS]")
    print(f"MAE:  {mae:.3f}")
    print(f"MSE:  {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAPE: {mape:.2f}%" if not np.isnan(mape) else 'MAPE: N/A')
    if zero_excluded:
        print(f"  (Excluded {zero_excluded} zero periods in MAPE)")
    print(f"MASE: {mase_val:.3f}" if not np.isnan(mase_val) else 'MASE: N/A')
    print(f"AIC:  {fit.aic:.2f}")
    print(f"BIC:  {fit.bic:.2f}")

    # Forecast
    steps = args.forecast_horizon
    forecast = fit.forecast(steps=steps)
    print(f"\n[FORECAST] Next {steps} periods:")
    print(forecast)

    # Summary tables
    print("\n[HISTORY (tail)]")
    hist_tail = pd.DataFrame(
        {'period': ts.index, 'bundle_demand': ts.values}).tail(12)
    print(hist_tail.to_string(index=False))
    print("\n[FORECAST TABLE]")
    fc_tbl = pd.DataFrame(
        {'period': forecast.index, 'forecast_bundle_demand': forecast.values})
    print(fc_tbl.to_string(index=False))
    print("\n[NOTE] No CSV export; copy needed values.")

    print("\n[INTERPRETATION]")
    try:
        print(interpret_bundle(ts, mae, rmse, mape, mase_val, non_zero,
              zero_excluded, args.demand_mode, seasonal_order))
    except Exception as e:
        print(f"Could not generate interpretation: {e}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(ts.index, ts.values, marker='o', label='History')
            plt.plot(forecast.index, forecast.values, marker='x',
                     linestyle='--', label='Forecast')
            plt.title(f'SARIMA Bundle ({args.demand_mode}) {pid_a}+{pid_b}')
            plt.xlabel('Period')
            plt.ylabel('Bundle Demand')
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"[PLOT ERROR] {e}")


if __name__ == '__main__':
    main()
