import pandas as pd
import os

print("="*60)
print("DOWNLOADING GERMAN CREDIT DATASET")
print("="*60)

# URL for German Credit Data from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

# Column names for the dataset
columns = [
    'checking_status', 'duration', 'credit_history', 'purpose', 
    'credit_amount', 'savings_status', 'employment', 
    'installment_commitment', 'personal_status', 'other_parties',
    'residence_since', 'property_magnitude', 'age', 
    'other_payment_plans', 'housing', 'existing_credits', 'job',
    'num_dependents', 'own_telephone', 'foreign_worker', 'class'
]

print("\nDownloading data from UCI repository...")
try:
    # Read data from URL
    df = pd.read_csv(url, sep=' ', names=columns, header=None)
    
    print(f"✓ Successfully downloaded {len(df)} rows and {len(df.columns)} columns")
    
    # Save to data/raw folder
    output_path = 'data/raw/german_credit_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✓ Saved to: {output_path}")
    
    # Display basic info
    print("\n" + "="*60)
    print("DATASET PREVIEW")
    print("="*60)
    print(f"\nShape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    print("\n" + "="*60)
    print("✅ DATA DOWNLOAD COMPLETE!")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ Error downloading data: {e}")
    print("\nAlternative: Download manually from:")
    print("https://www.kaggle.com/datasets/uciml/german-credit")