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
RULES_PATH   = 'mba_results/association_rules.csv'
FACT_PATH    = 'etl_dimensions/fact_transaction_dimension.csv'
PRODUCT_PATH = 'etl_dimensions/current_product_dimension.csv'

# Bundle selection (row in association_rules.csv to analyze)
BUNDLE_ROW = 7

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
NEW_PRICE = 300            # promotional bundle price you want to test

# =========================
# LOAD BUNDLE FROM RULES (now treated as parent_sku codes)
# =========================
try:
    rules_df = pd.read_csv(RULES_PATH)
    required_cols = {'antecedents_names','consequents_names'}
    if not required_cols.issubset(rules_df.columns):
        print("Error: Required columns not found in association_rules.csv")
        sys.exit()

    if not (0 <= BUNDLE_ROW < len(rules_df)):
        print(f"Error: BUNDLE_ROW {BUNDLE_ROW} out of range (0..{len(rules_df)-1}).")
        sys.exit()

    bundle = rules_df.iloc[BUNDLE_ROW]
    parent_a_sku = str(bundle['antecedents_names'])
    parent_b_sku = str(bundle['consequents_names'])
    print(f"Analyzing bundle (parent_sku): '{parent_a_sku}' and '{parent_b_sku}'")

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

# Basic column sanity
for col in ['product_id','parent_sku','Price']:
    if col not in product_df.columns:
        print(f"Error: '{col}' not found in product dimension.")
        sys.exit()
for col in ['Product ID','Receipt No','Line Total','Date']:
    if col not in fact_df.columns:
        print(f"Error: '{col}' not found in fact table.")
        sys.exit()

# =========================
# MAP parent_sku -> child product_ids and representative price
# =========================
def ids_and_price_for_parent(parent_sku: str):
    subset = product_df[product_df['parent_sku'] == parent_sku]
    if subset.empty:
        return set(), np.nan
    child_ids = set(subset['product_id'].dropna().astype(str))
    # Representative price for the parent group (MEDIAN of child prices)
    rep_price = float(subset['Price'].median())
    return child_ids, rep_price

child_ids_a, price_a = ids_and_price_for_parent(parent_a_sku)
child_ids_b, price_b = ids_and_price_for_parent(parent_b_sku)

if not child_ids_a or not child_ids_b:
    print(f"Could not resolve child product_ids for one or both parent_sku: "
          f"'{parent_a_sku}'({len(child_ids_a)}) or '{parent_b_sku}'({len(child_ids_b)}).")
    sys.exit()

# Standardize types for join/isin
fact_df['Product ID'] = fact_df['Product ID'].astype(str)

# =========================
# BUNDLE DEMAND BY PRICE POINT (receipts that contain ANY child of A and ANY child of B)
# =========================
pair_transactions = fact_df[fact_df['Product ID'].isin(child_ids_a.union(child_ids_b))]
receipt_products = pair_transactions.groupby('Receipt No')['Product ID'].apply(set)

def has_any(s: set, pool: set) -> bool:
    return any(pid in pool for pid in s)

receipts_with_both = receipt_products[
    receipt_products.apply(lambda s: has_any(s, child_ids_a) and has_any(s, child_ids_b))
].index

if len(receipts_with_both) == 0:
    print("No receipts found that contain BOTH parent_sku groups. Cannot build bundle demand curve.")
    sys.exit()

working_df = fact_df[fact_df['Receipt No'].isin(receipts_with_both)]

receipt_summary = working_df.groupby('Receipt No').agg(
    Combined_Price=('Line Total', 'sum'),   # sum for the two groups in the receipt
    Date=('Date', 'first')
)

demand_summary = (receipt_summary
                  .groupby('Combined_Price')
                  .agg(Num_Transactions=('Date', 'count'))
                  .reset_index()
                  .sort_values('Combined_Price'))

print(f"\nDemand Analysis for {parent_a_sku} + {parent_b_sku} bundle (by parent_sku):")
print(demand_summary)

# =========================
# TIME SERIES FOR BUNDLE (AGG_FREQ)
# =========================
receipt_summary['Date'] = pd.to_datetime(receipt_summary['Date'], errors='coerce')
bundle_sales_ts = receipt_summary.groupby(pd.Grouper(key='Date', freq=AGG_FREQ)).size()

if bundle_sales_ts.empty:
    print("No bundle sales found for forecasting.")
    sys.exit()

# =========================
# HOLT-WINTERS FORECAST (BUNDLE)
# =========================
bundle_model = ExponentialSmoothing(
    bundle_sales_ts,
    trend='add',
    seasonal='add',
    seasonal_periods=SEASONAL_PERIODS,
    initialization_method="estimated"
)

if OPTIMIZED:
    bundle_fitted = bundle_model.fit(optimized=True)
else:
    bundle_fitted = bundle_model.fit(
        smoothing_level=HW_ALPHA,
        smoothing_trend=HW_BETA,
        smoothing_seasonal=HW_GAMMA,
        optimized=False
    )

bundle_fc = bundle_fitted.forecast(HORIZON)

print(f"\n{AGG_FREQ} Holt-Winters Forecast for bundle '{parent_a_sku} + {parent_b_sku}':")
print(bundle_fc)

# =========================
# CONSTANT ELASTICITY (LOG-LOG) FOR BUNDLE (from price points)
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
current_price = float(price_a + price_b)  # sum of representative parent prices
new_price = float(NEW_PRICE)

print(f"\n--- PRICE setting FORECAST ---")
print(f"Current price (parent_sku A + B): Php {current_price:.2f} "
      f"(A≈{price_a:.2f}, B≈{price_b:.2f})")
print(f"New promotional price: Php {new_price:.2f}")
print(f"Price change: {((new_price - current_price) / current_price * 100):.1f}%")

# Power model (exact)
if current_price <= 0 or new_price <= 0:
    print("Warning: Non-positive price encountered; cannot apply power formula. Falling back to no change.")
    demand_multiplier = 1.0
else:
    price_ratio = new_price / current_price
    demand_multiplier = price_ratio ** elasticity

expected_demand_change_pct = (demand_multiplier - 1.0) * 100.0
print(f"Expected demand change in bundle units (power model): {expected_demand_change_pct:.1f}%")

bundle_fc_adj = bundle_fc * demand_multiplier

# =========================
# FORECAST INDEX (ALIGN TO AGG_FREQ)
# =========================
if AGG_FREQ.upper().startswith('Q'):
    step = pd.offsets.QuarterEnd()
elif AGG_FREQ.upper().startswith('M'):
    step = pd.offsets.MonthEnd()
else:
    step = pd.tseries.frequencies.to_offset(AGG_FREQ)

bundle_fc_index = pd.date_range(
    start=bundle_sales_ts.index[-1] + step,
    periods=len(bundle_fc),
    freq=AGG_FREQ
)

# Assign explicit indices to bundle forecasts
bundle_fc = bundle_fc.copy(); bundle_fc.index = bundle_fc_index
bundle_fc_adj = bundle_fc_adj.copy(); bundle_fc_adj.index = bundle_fc_index

# =========================
# INDIVIDUAL SERIES FOR A & B (ALL RECEIPTS, by parent_sku)
# =========================
def build_ts_all_for_parent(child_ids: set) -> pd.Series:
    lines = fact_df[fact_df['Product ID'].isin(child_ids)]
    if lines.empty:
        return pd.Series(dtype=float)
    rec = lines.groupby('Receipt No').agg(Date=('Date', 'first'))
    rec['Date'] = pd.to_datetime(rec['Date'], errors='coerce')
    return rec.groupby(pd.Grouper(key='Date', freq=AGG_FREQ)).size()

a_ts_all = build_ts_all_for_parent(child_ids_a)
b_ts_all = build_ts_all_for_parent(child_ids_b)

def fit_and_forecast(series: pd.Series, label: str):
    if series is None or series.empty:
        print(f"\nNo sales found for {label}. Skipping its forecast.")
        return None, None
    model = ExponentialSmoothing(
        series,
        trend='add',
        seasonal='add',
        seasonal_periods=SEASONAL_PERIODS,
        initialization_method="estimated"
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
    fc = fitted.forecast(HORIZON)
    fc_index = pd.date_range(start=series.index[-1] + step, periods=len(fc), freq=AGG_FREQ)
    return fc, fc_index

# Baseline (pre-bundle) forecasts
a_fc_all, a_fc_index_all = fit_and_forecast(a_ts_all, f"{parent_a_sku} (ALL children)")
b_fc_all, b_fc_index_all = fit_and_forecast(b_ts_all, f"{parent_b_sku} (ALL children)")

# Ensure explicit indices
if a_fc_all is not None: a_fc_all = a_fc_all.copy(); a_fc_all.index = a_fc_index_all
if b_fc_all is not None: b_fc_all = b_fc_all.copy(); b_fc_all.index = b_fc_index_all

# =========================
# CANNIBALIZATION MODEL (revised; based on baseline bundle units)
# =========================
a_fc_all_aligned = (a_fc_all.reindex(bundle_fc_index, fill_value=0)
                    if a_fc_all is not None else pd.Series(dtype=float, index=bundle_fc_index))
b_fc_all_aligned = (b_fc_all.reindex(bundle_fc_index, fill_value=0)
                    if b_fc_all is not None else pd.Series(dtype=float, index=bundle_fc_index))

cannibalization_units = bundle_fc  # baseline bundle units at status-quo price

a_fc_after_aligned = (a_fc_all_aligned - cannibalization_units).clip(lower=0)
b_fc_after_aligned = (b_fc_all_aligned - cannibalization_units).clip(lower=0)

a_fc_after_for_plot = (a_fc_after_aligned.reindex(a_fc_index_all, fill_value=np.nan)
                       if a_fc_index_all is not None else None)
b_fc_after_for_plot = (b_fc_after_aligned.reindex(b_fc_index_all, fill_value=np.nan)
                       if b_fc_index_all is not None else None)

# =========================
# REVENUE IMPACT ANALYSIS (revised, parent_sku prices)
# =========================
def revenue_forecast(forecast_series, price, idx):
    if forecast_series is None:
        return pd.Series(dtype=float, index=idx)
    return forecast_series.reindex(idx, fill_value=0) * price

# Use representative parent prices computed earlier
rev_a_before = revenue_forecast(a_fc_all_aligned, price_a, bundle_fc_index)
rev_b_before = revenue_forecast(b_fc_all_aligned, price_b, bundle_fc_index)

rev_a_after  = revenue_forecast(a_fc_after_aligned, price_a, bundle_fc_index)
rev_b_after  = revenue_forecast(b_fc_after_aligned, price_b, bundle_fc_index)

rev_bundle_before = pd.Series(0.0, index=bundle_fc_index)           # no bundle pre-intro
rev_bundle_after  = bundle_fc_adj * new_price                        # adjusted units at NEW_PRICE

indiv_rev_before = {"A": float(rev_a_before.sum()), "B": float(rev_b_before.sum())}
indiv_rev_after  = {"A": float(rev_a_after.sum()),  "B": float(rev_b_after.sum())}

overall_before = indiv_rev_before["A"] + indiv_rev_before["B"]
overall_after  = indiv_rev_after["A"] + indiv_rev_after["B"] + float(rev_bundle_after.sum())

print("\n--- DETAILED REVENUE IMPACT ANALYSIS (parent_sku, power elasticity) ---")
print("Individual Revenue Forecast BEFORE Bundling/Cannibalization:")
print(f"  {parent_a_sku}: Php {indiv_rev_before['A']:.2f}")
print(f"  {parent_b_sku}: Php {indiv_rev_before['B']:.2f}")

print("\nIndividual Revenue Forecast AFTER Bundling/Cannibalization:")
print(f"  {parent_a_sku}: Php {indiv_rev_after['A']:.2f}")
print(f"  {parent_b_sku}: Php {indiv_rev_after['B']:.2f}")

print(f"\nBundled Revenue Forecast BEFORE: Php {float(rev_bundle_before.sum()):.2f} (no bundle)")
print(f"Bundled Revenue Forecast AFTER  (promo price Php {new_price:.2f}): Php {float(rev_bundle_after.sum()):.2f}")

print(f"\nOverall Revenue BEFORE (A + B): Php {overall_before:.2f}")
print(f"Overall Revenue AFTER (A_after + B_after + Bundle_after): Php {overall_after:.2f}")

impact_abs = overall_after - overall_before
impact_pct = (impact_abs / overall_before * 100.0) if overall_before != 0 else float('nan')
print(f"Revenue Impact: Php {impact_abs:.2f} ({impact_pct:.1f}%)")

# =========================
# BREAK-EVEN BUNDLE UNITS (incremental units to offset loss)
# =========================
overall_before_series = rev_a_before + rev_b_before
overall_after_series  = rev_a_after + rev_b_after + rev_bundle_after
revenue_gap_series    = overall_before_series - overall_after_series

if new_price <= 0:
    print("\nBreak-even analysis skipped: NEW_PRICE must be > 0.")
else:
    needed_bundles_series = (revenue_gap_series / new_price).clip(lower=0)
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
        'Period': bundle_fc_index,
        'Revenue_Gap': revenue_gap_series.values.round(2),
        'Bundle_Units_Required_Ceil': needed_bundles_ceiling.astype('Int64').values
    })
    print("\nPer-period incremental bundle units required (rounded up):")
    print(be_df.to_string(index=False))

# =========================
# FUNCTION: detect trend
# =========================
def detect_trend(series: pd.Series, label: str) -> str:
    if series is None or len(series) < 2:
        return f"{label}: No data"
    first_val, last_val = series.iloc[0], series.iloc[-1]
    if last_val > first_val * 1.05:
        return f"{label}: Upward trend"
    elif last_val < first_val * 0.95:
        return f"{label}: Downward trend"
    else:
        return f"{label}: Relatively flat trend"

# =========================
# TREND ANALYSIS
# =========================
print("\n--- TREND ANALYSIS ---")
print(detect_trend(bundle_fc, "Baseline bundle forecast"))
print(detect_trend(bundle_fc_adj, "Adjusted bundle forecast (power model)"))
if a_fc_all is not None:
    print(detect_trend(a_fc_all, f"{parent_a_sku} baseline forecast"))
if a_fc_after_for_plot is not None:
    print(detect_trend(a_fc_after_for_plot.dropna(), f"{parent_a_sku} after cannibalization"))
if b_fc_all is not None:
    print(detect_trend(b_fc_all, f"{parent_b_sku} baseline forecast"))
if b_fc_after_for_plot is not None:
    print(detect_trend(b_fc_after_for_plot.dropna(), f"{parent_b_sku} after cannibalization"))

# =========================
# PLOTS (ALL IN ONE FRAME)
# =========================
fig, axes = plt.subplots(3, 1, figsize=(14, 16), sharex=False)

# 1) Bundle subplot
axes[0].plot(bundle_sales_ts.index, bundle_sales_ts.values,
             label=f'Historical ({AGG_FREQ})', marker='o')
axes[0].plot(bundle_fc_index, bundle_fc.values,
             label='Baseline Bundle Forecast (units, status-quo price)', linestyle='--', marker='s')
axes[0].plot(bundle_fc_index, bundle_fc_adj.values,
             label=f'Adjusted Bundle Forecast at Php {new_price:.0f}', linestyle='--', marker='s')
axes[0].plot([bundle_sales_ts.index[-1], bundle_fc_index[0]],
             [bundle_sales_ts.values[-1], bundle_fc.values[0]],
             color='gray', linestyle=':', linewidth=2)
axes[0].set_title(
    f'Bundle Forecast: {parent_a_sku} + {parent_b_sku}\n'
    f'Elasticity-adj demand change (power): {expected_demand_change_pct:.1f}% | '
    f'Old price: Php {current_price:.2f} → New price: Php {new_price:.2f}'
)
axes[0].set_ylabel('Bundle Sales (units)')
axes[0].grid(True)
axes[0].legend(loc='best')

# 2) Product A subplot
if not a_ts_all.empty:
    axes[1].plot(a_ts_all.index, a_ts_all.values,
                 label=f'Historical {parent_a_sku} ({AGG_FREQ})', marker='o')
if a_fc_all is not None:
    axes[1].plot(a_fc_index_all, a_fc_all.values,
                 label='Baseline Forecast (pre-bundle)', linestyle='--', marker='s')
if a_fc_after_for_plot is not None:
    axes[1].plot(a_fc_index_all, a_fc_after_for_plot.values,
                 label='After Cannibalization (subtract baseline bundle units)', linestyle='--', marker='^')
    if not a_ts_all.empty:
        axes[1].plot([a_ts_all.index[-1], a_fc_index_all[0]],
                     [a_ts_all.values[-1], a_fc_all.values[0]],
                     color='gray', linestyle=':', linewidth=2)
        axes[1].plot([a_ts_all.index[-1], a_fc_index_all[0]],
                     [a_ts_all.values[-1], a_fc_after_for_plot.values[0]],
                     color='green', linestyle=':', linewidth=2)

axes[1].set_title(f'Product A Forecasts: {parent_a_sku} — Baseline vs After Cannibalization')
axes[1].set_ylabel('Receipts with Product A (parent_sku)')
axes[1].grid(True)
axes[1].legend(loc='best')

# 3) Product B subplot
if not b_ts_all.empty:
    axes[2].plot(b_ts_all.index, b_ts_all.values,
                 label=f'Historical {parent_b_sku} ({AGG_FREQ})', marker='o')
if b_fc_all is not None:
    axes[2].plot(b_fc_index_all, b_fc_all.values,
                 label='Baseline Forecast (pre-bundle)', linestyle='--', marker='s')
if b_fc_after_for_plot is not None:
    axes[2].plot(b_fc_index_all, b_fc_after_for_plot.values,
                 label='After Cannibalization (subtract baseline bundle units)', linestyle='--', marker='^')
    if not b_ts_all.empty:
        axes[2].plot([b_ts_all.index[-1], b_fc_index_all[0]],
                     [b_ts_all.values[-1], b_fc_all.values[0]],
                     color='gray', linestyle=':', linewidth=2)
        axes[2].plot([b_ts_all.index[-1], b_fc_index_all[0]],
                     [b_ts_all.values[-1], b_fc_after_for_plot.values[0]],
                     color='green', linestyle=':', linewidth=2)

axes[2].set_title(f'Product B Forecasts: {parent_b_sku} — Baseline vs After Cannibalization')
axes[2].set_xlabel('Period')
axes[2].set_ylabel('Receipts with Product B (parent_sku)')
axes[2].grid(True)
axes[2].legend(loc='best')

plt.tight_layout()
plt.show()
