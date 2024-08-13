import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md


# Read data from CSV files
customer_df = pd.read_csv('customer-table.csv')
product_df = pd.read_csv('product-table.csv')
sale_table_df = pd.read_csv('Sale-table.csv')
product_group_table_df = pd.read_csv('product-group-table.csv')

# Group data by CategoryName and calculate the mean UnitPrice for each category
grouped_data = product_df.groupby('ProductName')['UnitPrice'].mean()

# Create a bar chart
grouped_data.plot(kind='bar', figsize=(10, 6))

# Add labels and title
plt.xlabel('ProductName')
plt.ylabel('Average UnitPrice')
plt.title('Average UnitPrice by Category')
plt.show()

# Line Chart: for sales date, sales quantity and sales amount
sale_table_df['sale date'] = pd.to_datetime(sale_table_df['sale date'])

df_plot = sale_table_df[['sale date', 'quantity sold', 'sale amount']]

plt.figure(figsize=(12, 6))  # Adjust chart size

# Draw a line showing the quantity sold.
plt.plot(df_plot['sale date'], df_plot['quantity sold'], label='Quantity Sold')

# Draw the revenue curve
plt.plot(df_plot['sale date'], df_plot['sale amount'], label='Sale Amount')

# Format x-axis
plt.gca().xaxis.set_major_locator(md.MonthLocator())
plt.gca().xaxis.set_major_formatter(md.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

# Add labels to the x-axis and y-axis
plt.xlabel('Sale Date')
plt.ylabel('Quantity Sold/Sale Amount')

# Add a title to the chart
plt.title('Sales Trend')
# Show caption
plt.legend()
plt.grid(True)  # Add background grid
plt.tight_layout()  # Adjust layout
plt.show()


# Data product group table
product_group_table_df = {
    'Category Name': ['Electronics', 'Clothing', 'Books', 'Furniture'],
    'Sales Tax Rate': [0.07, 0.05, 0.04, 0.06],
    'Shipping Weight': [1.5, 0.5, 0.3, 2.0]
}

df = pd.DataFrame(product_group_table_df)

# Pie Chart for Sales
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)  # Create 1 row, 2 columns, select the first chart
plt.pie(df['Sales Tax Rate'], labels=df['Category Name'], autopct='%1.1f%%', startangle=140)
plt.title('Sales Tax Rate by Category')

# # Create pie chart for Shipping Weight
plt.subplot(1, 2, 2)  # Create 1 row, 2 columns, select second chart
plt.pie(df['Shipping Weight'], labels=df['Category Name'], autopct='%1.1f%%', startangle=140)
plt.title('Shipping Weight by Category')

plt.tight_layout()
plt.show()

