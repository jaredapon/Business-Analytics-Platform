import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

df = pd.read_csv('Sales Book List - 20250505222327.xlsx - Sales Book.csv')

df = df.dropna(how='all')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
df.set_index('Date', inplace=True)
df['Total Sales'] = pd.to_numeric(df['Total Sales'], errors='coerce')
df = df.dropna(subset=['Total Sales'])

# Resample to monthly frequency (sum sales per month)
df_monthly = df['Total Sales'].resample('M').sum()

# Replace 0 sales with NaN (assuming 0 means closed)
df_monthly = df_monthly.replace(0, np.nan)

# Interpolate missing (closed) periods
df_monthly = df_monthly.interpolate(method='linear')

# Exclude the last month from analysis
df_monthly = df_monthly.iloc[:-1]

# Save cleaned/transformed data to CSV
df_monthly.to_csv('sales_forecast_result.csv')

seasonal_periods = 12 

model = ExponentialSmoothing(
    df_monthly,
    trend='add',
    seasonal='add',
    seasonal_periods=seasonal_periods
)
fit = model.fit()
forecast = fit.forecast(6)

# Get fitted values (in-sample predictions)
fitted_values = fit.fittedvalues

last_date = df_monthly.index[-1]
forecast_index = pd.date_range(start=last_date + pd.offsets.MonthEnd(1), periods=len(forecast), freq='M')
forecast.index = forecast_index

plt.figure(figsize=(12,6))
plt.plot(df_monthly, label='Actual')
plt.plot(fitted_values, label='Fitted', color='green', linestyle='--')  # Add fitted line
plt.plot(forecast, label='Forecast', color='red')
plt.legend()
plt.title('Holt-Winters Monthly Forecast of Total Sales')
plt.show()

# Calculate evaluation metrics
mae = mean_absolute_error(df_monthly, fitted_values)
rmse = np.sqrt(mean_squared_error(df_monthly, fitted_values))
wmape = abs(df_monthly - fitted_values).sum() / df_monthly.sum() * 100

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"WMAPE: {wmape:.2f}%")