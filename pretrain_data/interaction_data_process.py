import pandas as pd
import sys
from typing import List
from helper import save_metadata_to_json


def extract_interaction_from_file(file_path: str, group_col: str = "cc_num") -> List[List[int]]:
    """
    Extract ordered lists of transaction_type_id per group from a CSV or Parquet file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV or Parquet file containing the transaction data
    group_col : str
        Column name to group by (default: "cc_num")
        
    Returns:
    --------
    List[List[int]]
        List of ordered transaction_type_id lists, one per group
    """
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        else:
            print(f"Unsupported file format for {file_path}. Please use CSV or Parquet.")
            sys.exit(1)
        print(f"Loaded data from {file_path}. Shape: {df.shape}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        sys.exit(1)
    
    if group_col not in df.columns or 'transaction_type_id' not in df.columns or 'trans_date_trans_time' not in df.columns:
        print(f"Required columns ('{group_col}', 'transaction_type_id', 'trans_date_trans_time') not found in data")
        sys.exit(1)
    
    # Group, sort, and extract sequences
    sorted_sequences = []
    groups = df.groupby(group_col)
    for name, group in groups:
        sorted_group = group.sort_values(by='trans_date_trans_time')
        sorted_sequence = sorted_group['transaction_type_id'].tolist()
        sorted_sequences.append(sorted_sequence)
    
    print(f"Extracted {len(sorted_sequences)} sequences based on '{group_col}'.")
    return sorted_sequences

def main():
    input_file_path = '../data/credit_card_transaction_train_processed.csv'
    
    all_sequences = extract_interaction_from_file(input_file_path)
    train_idx = int(len(all_sequences) * 0.85)
    train_sequences = all_sequences[:train_idx]
    val_sequences = all_sequences[train_idx:]
    
    print(f"Total sequences: {len(all_sequences)}")
    print(f"Training sequences: {len(train_sequences)}, Validation sequences: {len(val_sequences)}")

    # Save the sequences to files
    save_metadata_to_json(train_sequences, 'train.json')
    save_metadata_to_json(val_sequences, 'dev.json')

if __name__ == "__main__":
    main()
