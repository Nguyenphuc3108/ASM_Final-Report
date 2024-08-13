import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Read data from CSV files
customer_df = pd.read_csv('customer-table.csv')
product_df = pd.read_csv('product-table.csv')
product_group_table_df = pd.read_csv('product-group-table.csv')
sale_table_df = pd.read_csv('Sale-table.csv')


# Ensure 'Unit Price' and other relevant columns are numeric
product_df['UnitPrice'] = pd.to_numeric(product_df['UnitPrice'], errors='coerce')
product_df['QuantityInStock'] = pd.to_numeric(product_df['QuantityInStock'], errors='coerce')

# Calculate Revenue
product_df['Revenue'] = product_df['QuantityInStock'] * product_df['UnitPrice']

# Convert ReleaseDate to datetime
product_df['ReleaseDate'] = pd.to_datetime(product_df['ReleaseDate'], errors='coerce')

# Handle any NaN values in the date
product_df = product_df.dropna(subset=['ReleaseDate'])

# Group by ReleaseDate to calculate daily revenue
daily_revenue = product_df.groupby('ReleaseDate')['Revenue'].sum().reset_index()

# Extract Day, Month, Year from ReleaseDate
daily_revenue['Day'] = daily_revenue['ReleaseDate'].dt.day
daily_revenue['Month'] = daily_revenue['ReleaseDate'].dt.month
daily_revenue['Year'] = daily_revenue['ReleaseDate'].dt.year

# Define features and target
features = ['Day', 'Month', 'Year']
target = 'Revenue'

X = daily_revenue[features]
y = daily_revenue[target]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Plot Actual vs Predicted Revenue
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Revenue')
plt.plot(y_pred, label='Predicted Revenue')
plt.title('Actual vs Predicted Revenue')
plt.xlabel('Date Index')
plt.ylabel('Revenue')
plt.legend()
plt.show()

# Predict future revenue
future_dates = pd.date_range(start='2023-07-05', end='2023-11-28', freq='D')
future_df = pd.DataFrame({'Date': future_dates})

# Extract Day, Month, Year for future dates
future_df['Day'] = future_df['Date'].dt.day
future_df['Month'] = future_df['Date'].dt.month
future_df['Year'] = future_df['Date'].dt.year

# Predict future revenue
future_features = future_df[features]
future_df['Predicted_Revenue'] = model.predict(future_features)

# Plot predicted future revenue
plt.figure(figsize=(12, 6))
plt.plot(future_df['Date'], future_df['Predicted_Revenue'], marker='o')
plt.title('Predicted Future Sales')
plt.xlabel('Date')
plt.ylabel('Predicted Revenue')
plt.grid()
plt.show()
