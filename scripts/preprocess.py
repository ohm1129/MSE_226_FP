import pandas as pd
from sklearn.preprocessing import LabelEncoder
import argparse
import os

def preprocess_dataset(input_path, output_path=None):
    """
    Preprocess a dataset by encoding categorical variables and converting booleans to integers.

    Args:
        input_path (str): Path to the input CSV file
        output_path (str, optional): Path to save the preprocessed CSV.
                                     If None, appends '_preprocessed' to input filename.

    Returns:
        pd.DataFrame: The preprocessed dataframe
    """
    print(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Original dataset shape: {df.shape}")

    # Create a copy for preprocessing
    df_preprocessed = df.copy()

    # Separate features and target (if y column exists)
    has_target = 'y' in df_preprocessed.columns
    if has_target:
        X = df_preprocessed.drop('y', axis=1)
        y = df_preprocessed['y']

        # Convert target to int: map yes/no strings or booleans to 1/0
        if y.dtype == bool:
            y = y.astype(int)
            print("  Converted boolean 'y' target to integer")
        else:
            y = y.map({'yes': 1, 'no': 0}).astype(int)
            print("  Mapped string 'y' target yes/no to 1/0")
    else:
        X = df_preprocessed
        y = None

    # Handle categorical variables
    categorical_cols = ['job', 'marital', 'education', 'month', 'contact', 'poutcome']
    for col in categorical_cols:
        if col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                print(f"  Encoded '{col}' using LabelEncoder")

    # Convert boolean columns to int
    bool_cols = ['default', 'housing', 'loan']
    for col in bool_cols:
        if col in X.columns:
            # Convert target to int: map yes/no strings or booleans to 1/0
            if X[col].dtype == bool:
                X[col] = X[col].astype(int)
                print("  Converted boolean target to integer")
            else:
                X[col] = X[col].map({'yes': 1, 'no': 0}).astype(int)
                print("  Mapped string target yes/no to 1/0")



    # Recombine features and target
    if has_target:
        df_preprocessed = pd.concat([X, y], axis=1)
    else:
        df_preprocessed = X

    # Determine output path
    if output_path is None:
        # Get directory, filename, and extension
        directory = os.path.dirname(input_path)
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)

        # Append '_preprocessed' to the filename
        output_path = os.path.join(directory, f"{name}_preprocessed{ext}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save preprocessed dataset
    df_preprocessed.to_csv(output_path, index=False)
    print(f"\nPreprocessed dataset saved to: {output_path}")
    print(f"Preprocessed dataset shape: {df_preprocessed.shape}")

    return df_preprocessed

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess online shoppers dataset by encoding categorical variables and converting booleans to integers.'
    )
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to the input CSV file to preprocess'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to save the preprocessed CSV (default: appends _preprocessed to input filename)'
    )

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_path):
        print(f"Error: Input file '{args.input_path}' does not exist.")
        return

    print("="*60)
    print("PREPROCESSING DATASET")
    print("="*60)

    # Preprocess the dataset
    preprocess_dataset(args.input_path, args.output)

    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
