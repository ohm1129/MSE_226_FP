from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# fetch dataset
online_shoppers_purchasing_intention_dataset = fetch_ucirepo(id=468)

# data (as pandas dataframes)
X = online_shoppers_purchasing_intention_dataset.data.features
y = online_shoppers_purchasing_intention_dataset.data.targets

# metadata
print(online_shoppers_purchasing_intention_dataset.metadata)

# variable information
print(online_shoppers_purchasing_intention_dataset.variables)

# combine features and target into one dataframe
df = pd.concat([X, y], axis=1)

# save raw dataset to CSV
df.to_csv('data/full_online_shoppers_data.csv', index=False)
print(f"\nRaw dataset saved to 'data/full_online_shoppers_data.csv' with shape: {df.shape}")
print(f"Data types preserved: categorical strings, booleans")

print("\nNext steps:")
print("  1. Run create_holdout_set.py to split into train/holdout (80/20)")
print("  2. Run preprocess.py on train data to prepare for modeling")
