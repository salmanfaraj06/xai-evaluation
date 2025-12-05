import pandas as pd
import numpy as np

def analyze_dataset(path):
    print(f"Analyzing {path}...")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print(f"Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("\nMissing values:\n", df.isnull().sum())
    print("\nData Types:\n", df.dtypes)
    
    # Check for potential target column
    potential_targets = [c for c in df.columns if 'status' in c.lower() or 'default' in c.lower() or 'target' in c.lower()]
    print(f"\nPotential target columns: {potential_targets}")

    for target in potential_targets:
        print(f"\nTarget distribution ({target}):")
        print(df[target].value_counts(normalize=True))
        print(df[target].value_counts())

    # Basic stats for numeric columns
    print("\nNumeric Summary:")
    print(df.describe())

if __name__ == "__main__":
    analyze_dataset("credit_risk_dataset.csv")
