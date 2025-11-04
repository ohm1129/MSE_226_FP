import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Load the full dataset
df = pd.read_csv('data/full_online_shoppers_data.csv')

print(f"Full dataset shape: {df.shape}")
print(f"Total rows: {len(df)}")

# Shuffle the dataset
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Calculate split index (80/20 split)
split_index = int(0.8 * len(df_shuffled))

# Split into train (80%) and holdout (20%)
train_data = df_shuffled[:split_index]
holdout_data = df_shuffled[split_index:]

print(f"\nTrain set shape: {train_data.shape} ({len(train_data)/len(df)*100:.1f}%)")
print(f"Holdout set shape: {holdout_data.shape} ({len(holdout_data)/len(df)*100:.1f}%)")

# Save the datasets
train_data.to_csv('data/train_data.csv', index=False)
holdout_data.to_csv('data/holdout_data.csv', index=False)

print("\nDatasets saved successfully:")
print("  - data/train_data.csv (for Part 1 analysis)")
print("  - data/holdout_data.csv (DO NOT TOUCH until Part 2!)")

# Verify the split
print(f"\nVerification:")
print(f"Train + Holdout = {len(train_data) + len(holdout_data)} (should equal {len(df)})")
print(f"No overlap: {len(set(train_data.index) & set(holdout_data.index)) == 0}")
