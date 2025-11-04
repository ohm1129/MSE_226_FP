from ucimlrepo import fetch_ucirepo
import pandas as pd

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

# save to CSV
df.to_csv('data/full_online_shoppers_data.csv', index=False)
print(f"\nDataset saved to 'data/full_online_shoppers_data.csv' with shape: {df.shape}")
