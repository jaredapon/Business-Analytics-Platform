import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import itertools
import warnings
from statsmodels.tsa.arima.model import ARIMA

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

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

results_dir = "arima_results"
os.makedirs(results_dir, exist_ok=True)

def find_best_arima_order(time_series, p_range=range(0, 4), d_range=range(0, 3), q_range=range(0, 4)):
    """
    Perform grid search to find the best ARIMA order based on AIC
    """
    best_aic = np.inf
    best_bic = np.inf
    best_order = None
    best_model = None
    models_tried = 0
    successful_models = 0
    
    print(f"Searching through {len(p_range) * len(d_range) * len(q_range)} parameter combinations...")
    
    for p, d, q in itertools.product(p_range, d_range, q_range):
        try:
            model = ARIMA(time_series, order=(p, d, q))
            fitted_model = model.fit()
            models_tried += 1
            successful_models += 1
            
            # You can use either AIC or BIC as selection criterion
            current_aic = fitted_model.aic
            current_bic = fitted_model.bic
            
            # Using AIC for model selection (lower is better)
            if current_aic < best_aic:
                best_aic = current_aic
                best_bic = current_bic
                best_order = (p, d, q)
                best_model = fitted_model
                print(f"New best order: {best_order} | AIC: {best_aic:.2f} | BIC: {best_bic:.2f}")
                
        except Exception as e:
            models_tried += 1
            # Uncomment below to see which combinations fail
            # print(f"Failed for order ({p},{d},{q}): {e}")
            continue
    
    print(f"Models tried: {models_tried}, Successful: {successful_models}")
    
    if best_model is None:
        print("No suitable model found. Using default (1,1,1)")
        best_order = (1, 1, 1)
        model = ARIMA(time_series, order=best_order)
        best_model = model.fit()
    
    return best_order, best_model, best_aic, best_bic

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

    if bundle_sales_ts.empty or len(bundle_sales_ts) < 5:
        print(f"Not enough data for ARIMA forecasting in {label}. Need at least 5 data points.")
        continue

    print(f"\n{'='*60}")
    print(f"FINDING BEST ARIMA ORDER FOR {label} BUNDLE")
    print(f"Bundle: {product_a_name} + {product_b_name}")
    print(f"Time series length: {len(bundle_sales_ts)}")
    print(f"Time series range: {bundle_sales_ts.index.min()} to {bundle_sales_ts.index.max()}")
    print(f"{'='*60}")

    # Find best ARIMA order using grid search
    best_order, fitted, best_aic, best_bic = find_best_arima_order(bundle_sales_ts)
    
    forecast_steps = 4
    forecast = fitted.forecast(steps=forecast_steps)

    # Model evaluation metrics
    residuals = fitted.resid
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    aic = fitted.aic
    bic = fitted.bic

    actual = bundle_sales_ts.values
    fitted_vals = fitted.fittedvalues

    # Weighted Mean Absolute Percentage Error (WMAPE)
    abs_errors = np.abs(actual - fitted_vals)
    wmape = np.sum(abs_errors) / np.sum(np.abs(actual)) * 100 if np.sum(np.abs(actual)) != 0 else float('inf')

    print(f"\n--- {label} Bundle ARIMA Forecast ---")
    print(f"Optimal ARIMA Order: {best_order}")
    print("History (tail):")
    print(bundle_sales_ts.tail(12))
    print("Forecast:")
    for i, (date, value) in enumerate(zip(forecast.index, forecast.values)):
        print(f"  {date.strftime('%Y-%m')}: {value:.2f}")
    
    print("\nModel Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Weighted Mean Absolute Percentage Error (WMAPE): {wmape:.2f}%")
    print(f"Akaike Information Criterion (AIC): {aic:.2f}")
    print(f"Bayesian Information Criterion (BIC): {bic:.2f}")

    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Actual data
    plt.plot(bundle_sales_ts.index, bundle_sales_ts.values, 
             label='Actual', marker='o', linewidth=2, markersize=4)
    
    # Fitted values (in-sample prediction)
    # fitted_values = fitted.fittedvalues
    # plt.plot(bundle_sales_ts.index, fitted_values, 
    #          label='Fitted', marker='s', linestyle='--', markersize=3)
    
    # Forecast (out-of-sample prediction)
    forecast_index = pd.date_range(bundle_sales_ts.index[-1], periods=forecast_steps+1, freq='ME')[1:]
    plt.plot(forecast_index, forecast.values, 
             label='Forecast', marker='x', linestyle='-.', markersize=6, linewidth=2)
    
    plt.title(f"ARIMA{best_order} Forecast: {product_a_name} + {product_b_name} ({label})", fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Bundle Sales', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    fig_filename = f"arima_{label.lower()}_bundle_{bundle_index}_order{best_order}.png"
    fig_path = os.path.join(results_dir, fig_filename)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved forecast plot to {fig_path}")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Label': [label],
        'Product_A': [product_a_name],
        'Product_B': [product_b_name],
        'ARIMA_Order': [str(best_order)],
        'MSE': [mse],
        'RMSE': [rmse],
        'MAE': [mae],
        'WMAPE': [wmape],
        'AIC': [aic],
        'BIC': [bic],
        'Data_Points': [len(bundle_sales_ts)]
    })
    
    results_file = os.path.join(results_dir, f"arima_results_summary.csv")
    if os.path.exists(results_file):
        results_df.to_csv(results_file, mode='a', header=False, index=False)
    else:
        results_df.to_csv(results_file, index=False)
    
    print(f"Results saved to {results_file}")

print(f"\n{'='*60}")
print("ARIMA forecasting completed!")
print(f"Results saved in: {results_dir}")
print(f"{'='*60}")