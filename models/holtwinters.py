import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from statsmodels.tsa.holtwinters import ExponentialSmoothing

rules_files = [
    ('mba_meal/association_rules.csv', 'Meal'),
    ('mba_foods/association_rules.csv', 'Foods'),
    ('mba_drinks/association_rules.csv', 'Drinks')
]

try:
    fact_df = pd.read_csv('etl_dimensions/fact_transaction_dimension.csv')
    product_df = pd.read_csv('etl_dimensions/current_product_dimension.csv')
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit()

results_dir = "holtwinters_results"
os.makedirs(results_dir, exist_ok=True)

for file_path, label in rules_files:
    try:
        rules_df = pd.read_csv(file_path)
        bundle_index = 0
        bundle = rules_df.iloc[bundle_index]
        product_a_name = bundle['antecedents_names']
        product_b_name = bundle['consequents_names']
    except Exception as e:
        print(f"Error loading association rules from {file_path}: {e}")
        continue

    product_a_match = product_df[product_df['product_name'] == product_a_name]
    product_b_match = product_df[product_df['product_name'] == product_b_name]
    if product_a_match.empty or product_b_match.empty:
        print(f"Could not find product IDs for '{product_a_name}' or '{product_b_name}' in {label}.")
        continue

    product_a_id = product_a_match['product_id'].iloc[0]
    product_b_id = product_b_match['product_id'].iloc[0]

    # Find receipts containing both products
    pair_transactions = fact_df[fact_df['Product ID'].isin([product_a_id, product_b_id])]
    receipt_products = pair_transactions.groupby('Receipt No')['Product ID'].apply(set)
    receipts_with_both = receipt_products[
        receipt_products.apply(lambda s: product_a_id in s and product_b_id in s)
    ].index

    if len(receipts_with_both) == 0:
        print(f"No receipts found containing BOTH '{product_a_name}' and '{product_b_name}' in {label}.")
        continue

    # Build bundle sales time series (e.g., monthly)
    bundle_df = fact_df[fact_df['Receipt No'].isin(receipts_with_both)]
    bundle_df['Date'] = pd.to_datetime(bundle_df['Date'], errors='coerce')
    bundle_sales_ts = bundle_df.drop_duplicates(subset=['Receipt No']).groupby(
        pd.Grouper(key='Date', freq='ME')
    )['Receipt No'].count()

    # Interpolate months with zero sales (RESTAURANT WAS CLOSED)
    if (bundle_sales_ts == 0).any():
        bundle_sales_ts = bundle_sales_ts.replace(0, np.nan).interpolate(method='linear').fillna(0)

    if bundle_sales_ts.empty:
        print(f"No bundle sales found for Holt-Winters forecasting in {label}.")
        continue

    # Fit Holt-Winters model (seasonal_periods=12 for monthly data)
    model = ExponentialSmoothing(
        bundle_sales_ts,
        trend='add',
        seasonal='add',
        seasonal_periods=12
    )
    fitted = model.fit(
        # smoothing_level=0.2,        
        # smoothing_slope=0.1,        
        # smoothing_seasonal=0.1      
    )
    forecast_steps = 4 
    forecast = fitted.forecast(steps=forecast_steps)

    # Model evaluation metrics
    residuals = fitted.resid
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    actual = bundle_sales_ts.values
    fitted_vals = fitted.fittedvalues

    wmape = np.sum(np.abs(actual - fitted_vals)) / np.sum(actual) * 100

    print(f"\n--- {label} Bundle Holt-Winters Forecast ---")
    print("History (tail):")
    print(bundle_sales_ts.tail(12))
    print("Forecast:")
    print(forecast)
    print("\nModel Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Weighted Mean Absolute Percentage Error (WMAPE): {wmape:.2f}%")

    plt.figure(figsize=(10, 5))
    # Actual
    plt.plot(bundle_sales_ts.index, bundle_sales_ts.values, label='Actual', marker='o')
    # Fitted (in-sample prediction)
    fitted_values = fitted.fittedvalues
    plt.plot(bundle_sales_ts.index, fitted_values, label='Fitted', marker='s', linestyle=':')
    # Forecast (out-of-sample prediction)
    forecast_index = pd.date_range(bundle_sales_ts.index[-1], periods=forecast_steps+1, freq='ME')[1:]
    plt.plot(forecast_index, forecast.values, label='Forecast', marker='x', linestyle='--')
    plt.title(f"Holt-Winters Forecast: {product_a_name} + {product_b_name} ({label})")
    plt.xlabel('Month')
    plt.ylabel('Bundle Sales')
    plt.legend()
    plt.tight_layout()
    fig_filename = f"holtwinters_{label.lower()}_bundle_{bundle_index}.png"
    fig_path = os.path.join(results_dir, fig_filename)
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved forecast plot to {fig_path}")
