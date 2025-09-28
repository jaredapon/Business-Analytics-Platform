import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pandas.tseries.offsets import MonthEnd

# Load data
df = pd.read_csv('etl_dimensions/fact_transaction_dimension.csv', parse_dates=['Date'])

# Aggregate number of receipts by month (demand)
df['Month'] = pd.to_datetime(df['Date']).dt.to_period('M')
monthly_demand = df.groupby('Month')['Receipt No'].nunique()
monthly_demand.index = monthly_demand.index.to_timestamp()

# Remove months after the last transaction (no actual demand)
last_actual_month = df['Month'].max().to_timestamp()
monthly_demand = monthly_demand[monthly_demand.index <= last_actual_month]

# Always remove the last month
monthly_demand = monthly_demand.iloc[:-1]

# Interpolate months with zero demand (restaurant closed)
if (monthly_demand == 0).any():
    monthly_demand = monthly_demand.replace(0, np.nan).interpolate(method='linear').fillna(0)

# Fit Holt-Winters model (seasonal_periods=12 for monthly data)
model = ExponentialSmoothing(
    monthly_demand,
    trend='add',
    seasonal='add',
    seasonal_periods=12
)
fitted = model.fit()

forecast_steps = 12
forecast = fitted.forecast(steps=forecast_steps)

# Model evaluation metrics
residuals = fitted.resid
mse = np.mean(residuals**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(residuals))
actual = monthly_demand.values
fitted_vals = fitted.fittedvalues
wmape = np.sum(np.abs(actual - fitted_vals)) / np.sum(actual) * 100

print("\n--- Monthly Demand (Receipts) Holt-Winters Forecast ---")
print("History (tail):")
print(monthly_demand.tail(12))
print("Forecast:")
print(forecast)
print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Weighted Mean Absolute Percentage Error (WMAPE): {wmape:.2f}%")

plt.figure(figsize=(10, 5))
plt.plot(monthly_demand.index, monthly_demand.values, label='Actual', marker='o')
# plt.plot(monthly_demand.index, fitted.fittedvalues, label='Fitted', marker='s', linestyle=':')
forecast_index = pd.date_range(monthly_demand.index[-1], periods=forecast_steps+1, freq='M')[1:]
plt.plot(forecast_index, forecast.values, label='Forecast', marker='x', linestyle='--')
plt.title("Holt-Winters Forecast: Monthly Demand (Number of Receipts)")
plt.xlabel('Month')
plt.ylabel('Number of Receipts')
plt.legend()
plt.tight_layout()
plt.show()
plt.close()