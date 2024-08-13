import pandas as pd

df = pd.read_csv("customer-table.csv")
df['FirstName'] = df['FirstName'].fillna('Unknown')
df = df[df['Email'].str.contains('@', na=False)]

print(df)
df.to_csv("customer-table.csv", index=False)