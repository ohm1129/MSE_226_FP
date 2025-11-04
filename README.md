# MSE 226 Final Project

## Dataset Setup

This project uses the Online Shoppers Purchasing Intention dataset from UCI ML Repository.

### Scripts

1. **load_dataset.py** - Fetches the dataset from UCI and saves it to `data/full_online_shoppers_data.csv`
2. **create_holdout_set.py** - Splits the data into train (80%) and holdout (20%) sets:
   - `data/train_data.csv` - For Part 1 analysis
   - `data/holdout_data.csv` - Reserved for Part 2 (do not use until then)

### Usage

```bash
# Step 1: Download the dataset
python load_dataset.py

# Step 2: Create train/holdout split
python create_holdout_set.py
```

All data files are stored in the `data/` folder.
