import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from helper import save_metadata_to_json
import os

def get_amt_bins(df: pd.DataFrame, number_bins: int = 10000, min_amt: int = 0, max_amt: int = 10000):
    """
    Create a mapping of amount bins to token IDs.
    
    Parameters:
    - df: DataFrame containing the 'amt' column.
    - number_bins: Number of bins to create for the amount.
    - min_amt: Minimum amount for binning.
    - max_amt: Maximum amount for binning.
    
    Returns:
    - A dictionary mapping amount bins to token IDs.
    """
    if 'amt' not in df.columns:
        raise ValueError("DataFrame must contain an 'amt' column.")

    # Create unique, rounded integer bin edges
    amount_bins = np.linspace(min_amt, max_amt, number_bins + 1)
    amount_bins = np.unique(np.round(amount_bins).astype(int))

    # Append infinity to create a final open-ended bin
    if amount_bins[-1] < np.inf:
        amount_bins = np.append(amount_bins, np.inf)

    # Generate bin labels (skip the first edge)
    bin_labels = []
    for i in range(1, len(amount_bins)):
        left = amount_bins[i - 1]
        right = amount_bins[i]
        if np.isinf(right):
            label = f"{left}-inf"
        else:
            label = f"{left}-{right}"
        bin_labels.append(label)

    token_dict = {label: f"amt_bin_{idx}" for idx, label in enumerate(bin_labels)}
    save_metadata_to_json(token_dict, '../data/amt_bins.json')
    
    return amount_bins, bin_labels


def save_raw_data():
    """Reads raw data from Hugging Face and saves it locally as CSV files."""
    splits = {'train': 'credit_card_transaction_train.csv', 'test': 'credit_card_transaction_test.csv'}
    base_url = "hf://datasets/pointe77/credit-card-transaction/"
    save_dir = '../data'  # Directory to save the raw data

    # Ensure the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Directory '{save_dir}' created.")

    # Read the data
    df_train = pd.read_csv(base_url + splits["train"])
    df_test = pd.read_csv(base_url + splits["test"])

    # Save raw data
    train_path = os.path.join(save_dir, 'credit_card_transaction_train_raw.csv')
    test_path = os.path.join(save_dir, 'credit_card_transaction_test_raw.csv')
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"Raw data saved as '{train_path}' and '{test_path}'")

def preprocess_and_save_data(input_path: str, output_path: str, drop_na: bool = True, verbose: bool = False):
    """
    Reads raw data from input_path, processes it, and saves processed data to output_path.
    """
    df = pd.read_csv(input_path)
    df_processed = preprocess_data(df, drop_na, verbose)
    df_processed.to_csv(output_path, index=False)
    if verbose:
        print(f"Processed data saved to {output_path}")

def preprocess_data(
    df: pd.DataFrame, 
    drop_na: bool = True,
    verbose: bool = False
) -> pd.DataFrame:
    """(Your preprocess_data function as given above)"""
    
    required_columns = ['trans_date_trans_time', 'amt', 'merchant']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    df_processed = df.copy()
    
    if drop_na:
        initial_rows = len(df_processed)
        df_processed = df_processed.dropna(subset=required_columns)
        if verbose and initial_rows > len(df_processed):
            print(f"Dropped {initial_rows - len(df_processed)} rows with missing values")
    
    try:
        df_processed['trans_date_trans_time'] = pd.to_datetime(df_processed['trans_date_trans_time'], errors='coerce')
        invalid_dates = df_processed['trans_date_trans_time'].isnull().sum()
        if invalid_dates > 0 and drop_na:
            df_processed = df_processed.dropna(subset=['trans_date_trans_time'])
    except Exception as e:
        raise ValueError(f"Error converting trans_date_trans_time to datetime: {e}")
    
    if not pd.api.types.is_numeric_dtype(df_processed['amt']):
        df_processed['amt'] = pd.to_numeric(df_processed['amt'], errors='coerce')
    
    amt_bins, bin_labels = get_amt_bins(df)

    df_processed["amt_bin"] = pd.cut(
        df_processed["amt"].abs(),
        bins=amt_bins,
        labels=bin_labels,
        include_lowest=True,
        right=False  # [left, right) intervals
    )

    df_processed['year'] = df_processed['trans_date_trans_time'].dt.year.fillna(0).astype(int)
    df_processed['month'] = df_processed['trans_date_trans_time'].dt.month.fillna(0).astype(int)
    df_processed['day'] = df_processed['trans_date_trans_time'].dt.day.fillna(0).astype(int)
    df_processed['day_of_week'] = df_processed['trans_date_trans_time'].dt.dayofweek.fillna(0).astype(int)
    df_processed['hour'] = df_processed['trans_date_trans_time'].dt.hour.fillna(0).astype(int)
    
    features_for_transaction_type = ["amt_bin", "merchant", "year", "month", "day", "day_of_week"]
    concat_parts = [df_processed[feature].astype(str).replace('nan', 'missing') for feature in features_for_transaction_type]
    df_processed["transaction_signature"] = pd.concat(concat_parts, axis=1).agg('_'.join, axis=1)
    
    le = LabelEncoder()
    df_processed["transaction_type_id"] = le.fit_transform(df_processed["transaction_signature"])
    
    if verbose:
        print(f"Processed data shape: {df_processed.shape}")
    
    return df_processed

if __name__ == "__main__":

    save_raw_data()  # Save the raw files

    # Then, for processing:
    preprocess_and_save_data(
        input_path='../data/credit_card_transaction_train_raw.csv',
        output_path='../data/credit_card_transaction_train_processed.csv',
        drop_na=True,
        verbose=True
    )
    preprocess_and_save_data(
        input_path='../data/credit_card_transaction_test_raw.csv',
        output_path='../data/credit_card_transaction_test_processed.csv',
        drop_na=True,
        verbose=True
    )
