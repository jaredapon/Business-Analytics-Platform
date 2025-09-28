import os
import pandas as pd
import matplotlib.pyplot as plt

# Ensure results directory exists
os.makedirs('descriptive_results', exist_ok=True)

# Load data
df = pd.read_csv('etl_dimensions/fact_transaction_dimension.csv', parse_dates=['Date'])

# Load product dimension
product_dim = pd.read_csv('etl_dimensions/current_product_dimension.csv')

# Merge transaction data with product dimension
df_merged = df.merge(product_dim, left_on='Product ID', right_on='product_id', how='left')


# Daily sales
daily_sales = df.groupby('Date')['Net Total'].sum()

plt.figure(figsize=(12, 4))
daily_sales.plot()
plt.title('Daily Sales Trend')
plt.ylabel('Net Total')
plt.xlabel('Date')
plt.tight_layout()
plt.savefig('descriptive_results/daily_sales_trend.png')
plt.close()

# Monthly sales
monthly_sales = df.groupby(df['Date'].dt.to_period('M'))['Net Total'].sum()
plt.figure(figsize=(12, 4))
monthly_sales.plot()
plt.title('Monthly Sales Trend')
plt.ylabel('Net Total')
plt.xlabel('Month')
plt.tight_layout()
plt.savefig('descriptive_results/monthly_sales_trend.png')
plt.close()


# Quarterly sales
quarterly_sales = df.groupby(df['Date'].dt.to_period('Q'))['Net Total'].sum()
plt.figure(figsize=(12, 4))
quarterly_sales.plot()
plt.title('Quarterly Sales Trend')
plt.ylabel('Net Total')
plt.xlabel('Quarter')
plt.tight_layout()
plt.savefig('descriptive_results/quarterly_sales_trend.png')
plt.close()


# Annual sales
annual_sales = df.groupby(df['Date'].dt.year)['Net Total'].sum()
plt.figure(figsize=(12, 4))
annual_sales.plot()
plt.title('Annual Sales Trend')
plt.ylabel('Net Total')
plt.xlabel('Year')
plt.tight_layout()
plt.savefig('descriptive_results/annual_sales_trend.png')
plt.close()

# Transaction Count vs. Revenue (Monthly) - Bar + Line Combo
monthly_group = df.groupby(df['Date'].dt.to_period('M'))
monthly_revenue = monthly_group['Net Total'].sum()
monthly_count = monthly_group.size()

fig, ax1 = plt.subplots(figsize=(12, 5))

months = monthly_count.index.astype(str)
ax1.bar(months, monthly_count, color='lightblue', label='Transaction Count')
ax1.set_ylabel('Transaction Count', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xlabel('Month')

# Show every 3rd month label, rotate for readability
step = 3
ax1.set_xticks(months[::step])
ax1.set_xticklabels(months[::step], rotation=45, ha='right')

ax2 = ax1.twinx()
ax2.plot(months, monthly_revenue, color='red', marker='o', label='Revenue')
ax2.set_ylabel('Revenue', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('Monthly Transaction Count vs. Revenue')
fig.tight_layout()
plt.savefig('descriptive_results/monthly_transaction_vs_revenue.png')
plt.close()

# Top 10 FOOD by revenue
food_revenue = (
    df_merged[df_merged['CATEGORY'] == 'FOOD']
    .groupby('Product Name')['Net Total']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

# Top 10 FOOD by revenue (horizontal bar)
plt.figure(figsize=(10, 6))
food_revenue.plot(kind='barh', color='orange')
plt.title('Top 10 Food Products by Revenue')
plt.xlabel('Revenue')
plt.ylabel('Product Name')
plt.tight_layout()
plt.savefig('descriptive_results/top10_food_by_revenue.png')
plt.close()

# Top 10 DRINK by revenue
drink_revenue = (
    df_merged[df_merged['CATEGORY'] == 'DRINK']
    .groupby('Product Name')['Net Total']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

# Top 10 DRINK by revenue (horizontal bar)
plt.figure(figsize=(10, 6))
drink_revenue.plot(kind='barh', color='skyblue')
plt.title('Top 10 Drink Products by Revenue')
plt.xlabel('Revenue')
plt.ylabel('Product Name')
plt.tight_layout()
plt.savefig('descriptive_results/top10_drink_by_revenue.png')
plt.close()

# Top 10 FOOD by quantity sold
food_qty = (
    df_merged[df_merged['CATEGORY'] == 'FOOD']
    .groupby('Product Name')['Qty']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))
food_qty.plot(kind='barh', color='green')
plt.title('Top 10 Food Products by Quantity Sold')
plt.xlabel('Quantity Sold')
plt.ylabel('Product Name')
plt.tight_layout()
plt.savefig('descriptive_results/top10_food_by_quantity.png')
plt.close()

# Top 10 DRINK by quantity sold
drink_qty = (
    df_merged[df_merged['CATEGORY'] == 'DRINK']
    .groupby('Product Name')['Qty']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))
drink_qty.plot(kind='barh', color='purple')
plt.title('Top 10 Drink Products by Quantity Sold')
plt.xlabel('Quantity Sold')
plt.ylabel('Product Name')
plt.tight_layout()
plt.savefig('descriptive_results/top10_drink_by_quantity.png')
plt.close()

# Revenue by CATEGORY
category_revenue = (
    df_merged.groupby('CATEGORY')['Net Total']
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))
category_revenue.plot(
    kind='pie',
    color='teal',
    autopct='%1.1f%%'
)
plt.title('Revenue by Category')
plt.savefig('descriptive_results/revenue_by_category.png')
plt.close()