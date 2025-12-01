from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# fetch dataset 
bank_marketing = fetch_ucirepo(id=222) 
  
# data (as pandas dataframes) 
X = bank_marketing.data.features 
y = bank_marketing.data.targets 
  
# variable information
print(bank_marketing.variables)

# combine features and target into one dataframe
df = pd.concat([X, y], axis=1)

# save raw dataset to CSV
df.to_csv('data/bank_marketing.csv', index=False)
print(f"\nRaw dataset saved to 'data/bank_marketing.csv' with shape: {df.shape}")
print(f"Data types preserved: categorical strings, booleans")

print("\nNext steps:")
print("  1. Run create_holdout_set.py to split into train/holdout (80/20)")
print("  2. Run preprocess.py on train data to prepare for modeling")
