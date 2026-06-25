"""
Day 1: Feature Engineering - Coding Examples

This script demonstrates Encoding, Scaling, Transformation, and Polynomial Features
using a simple mock dataset. 

Requirements: Make sure to install scikit-learn! (pip install scikit-learn)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures

print("="*60)
print("--- RAW DATA ---")
print("="*60)

# 1. Create a raw dataset
data = {
    'Size': ['Small', 'Large', 'Medium', 'Small', 'Large'], # Ordinal Categorical (Has a logical order)
    'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue'],       # Nominal Categorical (No logical order)
    'Salary': [50000, 52000, 48000, 60000, 2000000],        # Highly Skewed Data (Notice the 2 million outlier!)
    'Experience_Years': [1, 2, 1, 3, 20]                    # For scaling demonstration
}

df = pd.DataFrame(data)
print(df)


print("\n\n" + "="*60)
print("--- 1. ENCODING (Text to Numbers) ---")
print("="*60)

# A. Label Encoding (For Ordinal data like 'Size')
# We want Small=0, Medium=1, Large=2 because math knows 2 > 1 > 0
print("\n--- A. Label Encoding ('Size' Column) ---")
size_mapping = {'Small': 0, 'Medium': 1, 'Large': 2}
df['Size_Encoded'] = df['Size'].map(size_mapping)
print(df[['Size', 'Size_Encoded']])


# B. One-Hot Encoding (For Nominal data like 'Color')
# Red, Blue, Green have no mathematical order. We can't say Blue > Red.
print("\n--- B. One-Hot Encoding ('Color' Column) ---")
# pd.get_dummies automatically creates binary (0 or 1) columns for each category
df_encoded = pd.get_dummies(df, columns=['Color'])

# Notice the new 'Color_Blue', 'Color_Green', 'Color_Red' columns!
print(df_encoded[['Size', 'Color_Blue', 'Color_Green', 'Color_Red']]) 


print("\n\n" + "="*60)
print("--- 2. SCALING (Leveling the playing field) ---")
print("="*60)
# A machine learning model would look at Salary (50,000) and Experience (1) 
# and incorrectly assume Salary is 50,000 times more important simply because the number is bigger!

print("\n--- A. Standard Scaling ('Experience_Years') ---")
scaler = StandardScaler()
# StandardScaler centers data around 0. Some values become negative, some positive.
df['Experience_Scaled'] = scaler.fit_transform(df[['Experience_Years']])
print(df[['Experience_Years', 'Experience_Scaled']])


print("\n--- B. Min-Max Scaling ('Experience_Years') ---")
min_max = MinMaxScaler()
# MinMaxScaler squashes all data strictly between exactly 0 and 1.
df['Experience_MinMax'] = min_max.fit_transform(df[['Experience_Years']])
print(df[['Experience_Years', 'Experience_MinMax']])



print("\n\n" + "="*60)
print("--- 3. TRANSFORMATION (Fixing Skewed Data) ---")
print("="*60)
# Look at the 'Salary' column. One person makes $2,000,000! 
# This massive outlier ruins algorithms. Let's apply a Log Transformation to fix it.

print("\nApplying Log Transformation to 'Salary':")
# np.log1p safely applies logarithm math (it deals with 0s perfectly by adding 1)
df['Salary_Log'] = np.log1p(df['Salary'])

print(df[['Salary', 'Salary_Log']])
print("\n-> INCREDIBLE! Notice how the $2,000,000 outlier was squashed down to ~14.5!")
print("-> The normal salaries (50k) were squashed to ~10.8.")
print("-> The data is now much closer together. The model will no longer panic over the billionaire!")



print("\n\n" + "="*60)
print("--- 4. POLYNOMIAL FEATURES (Capturing Curves) ---")
print("="*60)
# Sometimes the relationship isn't a straight line. 
# We can create squared (x^2) and cubed (x^3) versions of a feature to allow linear models to draw curves!

print("\nGenerating Polynomial Features (Degree 2) for 'Experience_Years':")
poly = PolynomialFeatures(degree=2, include_bias=False)
# This will output the original feature, AND the feature mathematically squared!
experience_poly = poly.fit_transform(df[['Experience_Years']])

# Let's put it in a clean dataframe to look at it easily
poly_df = pd.DataFrame(experience_poly, columns=['Experience', 'Experience_Squared'])
print(poly_df)

print("\n" + "="*60)
print("Feature Engineering completed successfully!")
print("="*60)
