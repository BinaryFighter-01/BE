# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 2. Read the dataset
df = pd.read_csv("retail_sales_dataset.csv")

# 3. If "Region" not in dataset, create it (for demo)
if 'Region' not in df.columns:
    np.random.seed(42)
    df['Region'] = np.random.choice(['North', 'South', 'East', 'West'], size=len(df))

# 4. Explore dataset
print(df.head())
print("\nColumns:", df.columns)

# 5. Group by Region and calculate total sales
sales_by_region = df.groupby('Region')['Total Amount'].sum().sort_values(ascending=False)
print("\nTotal Sales by Region:\n", sales_by_region)

# 6. Bar plot - Sales distribution by region
sales_by_region.plot(kind='bar', color='skyblue')
plt.title("Total Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales Amount")
plt.show()

# 7. Identify top performing region
top_region = sales_by_region.idxmax()
print("\nTop Performing Region:", top_region)

# 8. Group by Region & Product Category
region_category_sales = df.groupby(['Region', 'Product Category'])['Total Amount'].sum().unstack()
print("\nSales by Region and Product Category:\n", region_category_sales)

# 9. Stacked bar plot
region_category_sales.plot(kind='bar', stacked=True, figsize=(8,5))
plt.title("Sales by Region and Product Category")
plt.xlabel("Region")
plt.ylabel("Total Sales Amount")
plt.legend(title="Product Category")
plt.show()
