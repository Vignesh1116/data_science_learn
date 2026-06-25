"""
Day 1: Advanced Exploratory Data Analysis (EDA) - CSV Example

This script loads a real dataset from 'sales_data.csv' (50 rows) 
and performs the exact same EDA workflow.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("="*60)
print("--- SESSION 1 & 2: UNDERSTANDING A DATASET ---")
print("="*60)

# Check if the file exists
file_name = 'sales_data.csv'
if not os.path.exists(file_name):
    print(f"Error: {file_name} not found! Make sure it is in the same folder.")
    exit()

# Load the dataset from CSV
df = pd.read_csv(file_name)

print(f"\nSuccessfully loaded {file_name}!")
print(f"Shape of the dataset: Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n1. First 5 rows of our data (df.head()):")
print(df.head())

print("\n2. Data Summary (df.info()):")
df.info()

print("\n3. Statistical Summary (df.describe()):")
print(df.describe())


print("\n\n" + "="*60)
print("--- SESSION 3: MISSING VALUE ANALYSIS ---")
print("="*60)

print("\n1. Counting Missing Values in each column:")
print(df.isnull().sum())

print("\n2. Handling Missing Values:")
# Fill the missing Age with the median age
median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)
print(f"Filled missing Age with the median value: {median_age:.1f}")

# Fill the missing Spending with the mean (average)
mean_spending = df['Spending'].mean()
df['Spending'] = df['Spending'].fillna(mean_spending)
print(f"Filled missing Spending with the mean value: {mean_spending:.2f}")

print("\nMissing values remaining after fix:")
print(df.isnull().sum())

print("\nDataset after fixing missing values (First 10 rows):")
print(df.head(10))


print("\n\n" + "="*60)
print("--- SESSION 4: DISTRIBUTION ANALYSIS ---")
print("="*60)

print("\nPlotting a Histogram for 'Spending' to see its distribution...")
plt.figure(figsize=(7, 4))
df['Spending'].hist(bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Customer Spending')
plt.xlabel('Spending Amount ($)')
plt.ylabel('Number of Customers')

print("-> Note: Close the graph window to continue!")
plt.show()


print("\n\n" + "="*60)
print("--- SESSION 5: CORRELATION ANALYSIS ---")
print("="*60)

print("\nCalculating Correlation Matrix:")
correlation = df.corr(numeric_only=True)
print(correlation)

print("\nPlotting Correlation Heatmap...")
plt.figure(figsize=(7, 4))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')

print("-> Note: Close the heatmap window to continue!")
plt.show()


print("\n\n" + "="*60)
print("--- SESSION 6: OUTLIER DETECTION ---")
print("="*60)

print("\nPlotting a Boxplot for 'Age' to find Outliers...")
plt.figure(figsize=(7, 4))
sns.boxplot(x=df['Age'], color='lightgreen')
plt.title('Boxplot of Customer Age')
plt.xlabel('Age')

print("-> Note: Close the boxplot window to finish the script!")
plt.show()

print("\nLet's find the Outliers in 'Age' mathematically (Age > 100):")
outliers = df[df['Age'] > 100]
print("Found Outlier(s):")
print(outliers)

print("\n" + "="*60)
print("Congratulations! You have successfully analyzed a real CSV file.")
print("="*60)
