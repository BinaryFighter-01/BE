import pandas as pd
import matplotlib.pyplot as plt

csv_data = pd.read_csv('sales_data.csv', encoding='latin1')
print("CSV Data Loaded Successfully!\n")
print(csv_data.head())

csv_data.to_excel('sales_data.xlsx', index=False)
csv_data.to_json('sales_data.json', orient='records', indent=4)
print("\nData converted to Excel and JSON successfully!\n")

csv_data = pd.read_csv('sales_data.csv', encoding='latin1')
excel_data = pd.read_excel('sales_data.xlsx')
json_data = pd.read_json('sales_data.json')

print("Excel and JSON Data Loaded Successfully!\n")

data = pd.concat([csv_data, excel_data, json_data], ignore_index=True)

print("Combined Data (All Formats):")
print(data.head())

data.drop_duplicates(inplace=True)

data.fillna(0, inplace=True)

data['Total_Sale'] = data['QUANTITYORDERED'] * data['PRICEEACH']

print("\nDescriptive Statistics:")
print(data.describe())

total_sales = data['Total_Sale'].sum()
print("\nTotal Sales:", total_sales)


avg_order_value = data['Total_Sale'].mean()
print("Average Order Value:", avg_order_value)

sales_by_category = data.groupby('PRODUCTLINE')['Total_Sale'].sum()
print("\nSales by Category:\n", sales_by_category)

sales_by_category.plot(kind='bar', title='Sales by Product Category')
plt.ylabel('Total Sales')
plt.show()

sales_by_category.plot(kind='pie', autopct='%1.1f%%', title='Sales Distribution')
plt.ylabel('')
plt.show()

plt.boxplot(data['Total_Sale'])
plt.title('Distribution of Total Sales')
plt.ylabel('Sale Value')
plt.show()
