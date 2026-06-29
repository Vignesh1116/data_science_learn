"""
=============================================================================
STUDENT INSTRUCTIONS
=============================================================================
Make sure to install the required libraries before running this script. 
Open your terminal and type:

    pip install pandas numpy scikit-learn category_encoders

To run this script, make sure your terminal is inside the Day_2 folder and type:
    
    python ./day2_baseball_prediction.py
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer # Tool to fill missing data blanks
from sklearn.feature_selection import SelectKBest, mutual_info_regression # Tools for Feature Selection
from sklearn.model_selection import train_test_split # Tool to split data into train and test sets
from sklearn.linear_model import LinearRegression # The Machine Learning model we will train
import os

# category_encoders is needed for Target Encoding text
try:
    from category_encoders import TargetEncoder
except ImportError:
    TargetEncoder = None

def main():
    print("\n=======================================================")
    print("⚾ DAY 2: PREPROCESSING, FEATURE SELECTION, & MODELING ⚾")
    print("=======================================================\n")

    # --- SETUP: LOADING THE DATA ---
    print("Loading the Baseball dataset...")
    file_path = 'train.csv'
    
    if not os.path.exists(file_path):
        print(f"Error! Cannot find '{file_path}'. Please make sure the CSV is in this folder!")
        return
        
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully. Rows: {df.shape[0]}, Features: {df.shape[1]}\n")


    # --- CONCEPT 1: HANDLING MISSING DATA ---
    print("--- CONCEPT 1: HANDLING MISSING DATA ---")
    print("Artificially deleting some 'Hits' (H) data to demonstrate missing value imputation...")
    
    # Artificially create missing data for the lesson
    df.loc[0:25, 'H'] = np.nan 
    
    # Create an 'Imputer' that replaces missing values with the median
    imputer = SimpleImputer(strategy='median')
    
    # Apply the imputer to the column to fill the blanks
    df['H'] = imputer.fit_transform(df[['H']])
    print(f"Imputation complete. 'Hits' (H) now has {df['H'].isnull().sum()} null values.\n")
    

    # --- CONCEPT 2: SKEWED DISTRIBUTIONS ---
    print("--- CONCEPT 2: SKEWED DISTRIBUTIONS ---")
    print("Evaluating the skewness of the Runs (R) distribution...")
    
    # Apply np.log1p (Logarithm + 1) to compress the distribution of numbers
    df['LogRuns'] = np.log1p(df['R'])
    print(f"Log Transformation applied. New skewness: {df['LogRuns'].skew():.2f} (closer to 0 is perfectly balanced).\n")
    

    # --- CONCEPT 3: HIGH CARDINALITY CATEGORICALS ---
    print("--- CONCEPT 3: HIGH CARDINALITY TEXT ---")
    print("Creating a synthetic text column to demonstrate high cardinality categorical encoding...")
    
    # Artificially create a high cardinality text column for the lesson
    df['Team_ID'] = ['Team_' + str(np.random.randint(1, 150)) for _ in range(len(df))]
    
    if TargetEncoder is not None:
        print("Applying 'Target Encoding' to replace text categories with the average Wins ('W') for each team.")
        encoder = TargetEncoder()
        
        # Fit it using the Team_ID column and our Target (W)
        df['Team_ID_Encoded'] = encoder.fit_transform(df['Team_ID'], df['W'])
    else:
        print("Error: 'category_encoders' is not installed. Please run 'pip install category_encoders'!\n")


    # --- CONCEPT 4: FEATURE SELECTION (MUTUAL INFO & SELECTKBEST) ---
    print("\n--- CONCEPT 4: FEATURE SELECTION ---")
    print("Using Mutual Information to automatically select the best predictive features...")
    
    # Give it 4 features and ask it to pick the best 2
    features_to_test = ['R', 'HR', 'SO', 'SB'] # Runs, Home Runs, Strikeouts, Stolen Bases
    X_features = df[features_to_test].fillna(0)
    y_target = df['W']
    
    # Tell SelectKBest to pick the top 2 features (k=2)
    selector = SelectKBest(score_func=mutual_info_regression, k=2)
    selector.fit(X_features, y_target)
    
    # Get the winning features
    winning_features = selector.get_support()
    best_features = X_features.columns[winning_features].tolist()
    
    print(f"SelectKBest automatically selected the top 2 features: {best_features}\n")
    
    
    # --- CONCEPT 5: SPLITTING DATA (TRAIN VS TEST) ---
    print("--- CONCEPT 5: SPLITTING DATA (TRAIN VS TEST) ---")
    print("Splitting the data into Training Data (to study) and Testing Data (for the final exam)...")
    
    # X will hold our selected best features (the clues)
    X = df[best_features]
    # y will hold our target (the answers we want to predict)
    y = df['W']
    
    # We hide 20% of the data (test_size=0.2) to test the model later.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Data split complete: {len(X_train)} rows for training, {len(X_test)} rows for testing.\n")
    
    
    # --- CONCEPT 6: MODEL TRAINING & PREDICTION ---
    print("--- CONCEPT 6: MODEL TRAINING & PREDICTION ---")
    print("Training the Linear Regression model on the training data...")
    
    # Create the model
    model = LinearRegression()
    # Train the model (it learns the mathematical pattern here)
    model.fit(X_train, y_train)
    
    print("Model training complete. Now asking the model to predict Wins for the unseen Test data...\n")
    
    # Ask the model to make predictions on the 20% of data it has never seen
    predictions = model.predict(X_test)
    
    print("Comparing the model's predictions to the actual real answers for the first 3 teams:")
    actual_wins = y_test.head(3).values 
    predicted_wins = predictions[:3]
    
    for i in range(3):
        predicted = round(predicted_wins[i]) 
        actual = actual_wins[i]
        difference = abs(actual - predicted)
        
        print(f"\nTeam {i+1}:")
        print(f"  Model Guessed:   {predicted} Wins")
        print(f"  Real Answer was: {actual} Wins")
        print(f"  Difference:      {difference} Wins off")

    print("\nData preprocessing, feature selection, and model training successfully complete! ⚾")

if __name__ == "__main__":
    main()
