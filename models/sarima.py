import argparse
import sys
import warnings
from typing import Optional, Tuple

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

"""SARIMA Modeling Script
Matches style of arima.py but includes seasonal (P,D,Q,s) components.
Features:
  - Token resolution via association rules or explicit product token
  - Automatic differencing (non-seasonal d) via ADF unless forced
  - Grid search over (p,q) while keeping d fixed and seasonal order fixed
  - Terminal-only output (history tail, forecast table, interpretation)
  - Interpretation section summarizing fit quality
"""

# --------------------------- Helpers --------------------------- #


def parse_args():
    p = argparse.ArgumentParser(
        description="Seasonal ARIMA (SARIMA) modeling for a single product.")
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument('--from-rules', action='store_true',
                     help='Infer product from association rules antecedent row.')
    src.add_argument('--product-token', dest='product_token',
                     help='Explicit product token (product_id, product_name, parent_sku).')
    p.add_argument('--rules-path', default='mba_drinks/association_rules.csv',
                   help='Association rules CSV.')
    p.add_argument('--rules-set', choices=['drinks', 'foods'],
                   default=None, help='Shortcut: choose mba_drinks or mba_foods.')
    p.add_argument('--use-row', type=int, default=0,
                   help='Row in rules file when using --from-rules.')
    p.add_argument(
        '--fact-path', default='etl_dimensions/fact_transaction_dimension.csv')
    p.add_argument('--product-path',
                   default='etl_dimensions/current_product_dimension.csv')
    p.add_argument('--freq', default='QE',
                   help='Resample frequency (default quarterly end QE).')
    p.add_argument('--forecast-horizon', type=int, default=4,
                   help='Future periods to forecast.')
    # Non-seasonal search
    p.add_argument('--max-p', type=int, default=2)
    p.add_argument('--max-q', type=int, default=2)
    p.add_argument('--max-d', type=int, default=2)
    p.add_argument('--force-d', type=int, default=None,
                   help='Override automatic differencing.')
    # Seasonal order (fixed during grid search)
    p.add_argument('--seasonal-P', type=int, default=1)
    p.add_argument('--seasonal-D', type=int, default=1)
    p.add_argument('--seasonal-Q', type=int, default=1)
    p.add_argument('--seasonal-s', type=int, default=4,
                   help='Seasonal period (e.g., 4=quarterly, 12=monthly).')
    p.add_argument('--min-length', type=int, default=8,
                   help='Minimum non-zero periods warning threshold.')
    p.add_argument('--plot', action='store_true')
    return p.parse_args()


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


def select_product_token(args):
    if args.from_rules:
        try:
            rules_df = pd.read_csv(args.rules_path)
            if args.use_row >= len(rules_df):
                raise ValueError(
                    f"use-row {args.use_row} out of range (rows={len(rules_df)})")
            row = rules_df.iloc[args.use_row]
            token = str(row['antecedents_names']).split(',')[0].strip()
            print(f"[INFO] Using token from rules row {args.use_row}: {token}")
            return token
        except Exception as e:
            print(f"[ERROR] Failed to load rules: {e}")
            sys.exit(1)
    if args.product_token:
        print(f"[INFO] Using explicit product token: {args.product_token}")
        return args.product_token
    print('[ERROR] Must supply --from-rules or --product-token.')
    sys.exit(1)


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


def grid_search_sarima(series: pd.Series, d: int, max_p: int, max_q: int, seasonal_order: tuple):
    best = None
    tried = []
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            try:
                model = SARIMAX(series, order=(p, d, q), seasonal_order=seasonal_order,
                                enforce_stationarity=False, enforce_invertibility=False)
                fit = model.fit(disp=False)
                aic = fit.aic
                tried.append(((p, d, q), aic))
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
    return best[0], best[2], tried


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


def interpret_sarima(ts: pd.Series, mae: float, rmse: float, mape: float, mase_val: float, zero_excluded: int, nonzero_ct: int) -> str:
    lines = []
    mean_demand = ts.mean() if len(ts) else float('nan')
    rmse_ratio = rmse / \
        mean_demand if mean_demand and mean_demand != 0 else float('nan')
    lines.append(f"- MAPE level: {classify_mape(mape)}")
    if not np.isnan(rmse_ratio):
        lines.append(f"- RMSE/mean demand = {rmse_ratio:.2f}")
    if not np.isnan(mase_val):
        if mase_val < 1:
            lines.append(
                f"- MASE {mase_val:.2f} (<1) better than naive benchmark.")
        else:
            lines.append(
                f"- MASE {mase_val:.2f} (>=1) not improving naive; consider revising seasonal order or aggregation.")
    else:
        lines.append(
            "- MASE not available (insufficient length or zero variation).")
    if zero_excluded:
        lines.append(
            f"- {zero_excluded} zero periods excluded in MAPE; intermittent demand present.")
    if nonzero_ct < 8:
        lines.append(
            f"- Only {nonzero_ct} non-zero periods → fragile model; consider coarser frequency.")
    lines.append("- Validate with rolling-origin or hold-out for robustness.")
    return "\n".join(lines)


def main():
    args = parse_args()

    if getattr(args, 'rules_set', None):
        args.rules_path = f"mba_{args.rules_set}/association_rules.csv"
        print(
            f"[INFO] Using rules set '{args.rules_set}' -> {args.rules_path}")

    # Load datasets
    try:
        fact_df = pd.read_csv(args.fact_path)
        product_df = pd.read_csv(args.product_path)
    except Exception as e:
        print(f"[ERROR] Loading data: {e}")
        sys.exit(1)

    token = select_product_token(args)
    pid, method = resolve_token_to_product_id(token, product_df)
    if not pid:
        print(f"[ERROR] Could not resolve token '{token}' to product_id.")
        sys.exit(1)
    print(f"[RESOLVE] Token '{token}' -> product_id {pid} via {method}")

    if 'Date' not in fact_df.columns:
        print('[ERROR] Fact table missing Date column.')
        sys.exit(1)
    fact_df['Date'] = pd.to_datetime(fact_df['Date'], errors='coerce')
    sub = fact_df[fact_df['Product ID'].astype(str) == pid]
    if sub.empty:
        print('[ERROR] No transactions for selected product.')
        sys.exit(1)

    daily = sub.groupby('Date')['Qty'].sum().sort_index()
    ts = daily.resample(args.freq).sum()

    nonzero_ct = (ts > 0).sum()
    if nonzero_ct < args.min_length:
        print(
            f"[WARN] Only {nonzero_ct} non-zero periods (< {args.min_length}).")

    # Differencing decision
    if args.force_d is not None:
        d = args.force_d
        print(f"[INFO] Using forced d={d}")
    else:
        d = determine_d(ts, args.max_d)

    seasonal_order = (args.seasonal_P, args.seasonal_D,
                      args.seasonal_Q, args.seasonal_s)
    order, fitted, tried = grid_search_sarima(
        ts, d, args.max_p, args.max_q, seasonal_order)
    print(
        f"\n[MODEL] Best SARIMA order={order} x {seasonal_order} AIC={fitted.aic:.2f}")

    # In-sample fit
    try:
        insample = fitted.predict(start=ts.index[0], end=ts.index[-1])
    except Exception:
        insample = fitted.fittedvalues
    insample = insample.reindex(ts.index)

    mae = mean_absolute_error(ts, insample)
    mse = mean_squared_error(ts, insample)
    rmse = np.sqrt(mse)
    actual_nonzero = ts.replace(0, np.nan)
    if actual_nonzero.notna().any():
        mape = (np.abs((ts - insample)/actual_nonzero)).mean()*100
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
    print(f"AIC:  {fitted.aic:.2f}")
    print(f"BIC:  {fitted.bic:.2f}")

    # Forecast
    steps = args.forecast_horizon
    forecast = fitted.forecast(steps=steps)
    print(f"\n[FORECAST] Next {steps} periods:")
    print(forecast)

    # Summary tables
    print("\n[HISTORY (tail)]")
    hist_tail = pd.DataFrame({'period': ts.index, 'qty': ts.values}).tail(12)
    print(hist_tail.to_string(index=False))
    print("\n[FORECAST TABLE]")
    fc_tbl = pd.DataFrame(
        {'period': forecast.index, 'forecast_qty': forecast.values})
    print(fc_tbl.to_string(index=False))
    print("\n[NOTE] No CSV export; copy needed values.")

    print("\n[INTERPRETATION]")
    try:
        print(interpret_sarima(ts, mae, rmse, mape,
              mase_val, zero_excluded, nonzero_ct))
    except Exception as e:
        print(f"Could not generate interpretation: {e}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(ts.index, ts.values, label='History', marker='o')
            plt.plot(forecast.index, forecast.values,
                     label='SARIMA Forecast', marker='x', linestyle='--')
            plt.title(f"SARIMA Forecast product {pid} (token {token})")
            plt.xlabel('Period')
            plt.ylabel('Quantity')
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"[PLOT ERROR] {e}")


if __name__ == '__main__':
    main()
