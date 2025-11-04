# MSE 226 Final Project

## Dataset Setup

This project uses the Online Shoppers Purchasing Intention dataset from UCI ML Repository.

### Scripts

1. **load_dataset.py** - Fetches the raw dataset from UCI and saves it to `data/full_online_shoppers_data.csv`
2. **create_holdout_set.py** - Splits the raw data into train (80%) and holdout (20%) sets:
   - `data/train_data.csv` - Raw training data for Part 1 analysis
   - `data/holdout_data.csv` - Raw holdout data (Reserved for Part 2 - do not use until then)
3. **preprocess.py** - Preprocesses a dataset by encoding categorical variables and converting booleans to integers

### Usage

```bash
# Step 1: Download the raw dataset
python scripts/load_dataset.py

# Step 2: Create train/holdout split (raw data)
python scripts/create_holdout_set.py

# Step 3: Preprocess the training data for modeling
python scripts/preprocess.py data/train_data.csv
# This creates: data/train_data_preprocessed.csv
```

### Data Files

- `data/full_online_shoppers_data.csv` - Raw full dataset from UCI
- `data/train_data.csv` - Raw training data (80% split)
- `data/train_data_preprocessed.csv` - Preprocessed training data (categorical encoded, booleans as 0/1)
- `data/holdout_data.csv` - Raw holdout data (20% split) - DO NOT TOUCH until Part 2!

### Preprocessing

The `preprocess.py` script:
- Encodes categorical variables (Month, VisitorType) using LabelEncoder
- Converts boolean columns (Weekend, Revenue) to integers (0/1)
- Can be used on any dataset: `python scripts/preprocess.py <input_file.csv>`
- Automatically appends `_preprocessed` to the filename if no output path is specified

**Important:** We only preprocess the training data for Part 1. The holdout set remains raw and untouched until Part 2!
