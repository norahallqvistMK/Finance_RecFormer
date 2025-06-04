from collections import defaultdict
import gzip
import json
from tqdm import tqdm
import os
import pandas as pd
import sys
from typing import List, Dict, Union

def extract_interaction_from_file(file_path: str, group_col: str = "cc_num") -> List[Dict[str, Union[List[int], int]]]:
    """
    Extract ordered lists of transaction_type_id per group with fraud labels from a CSV or Parquet file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV or Parquet file containing the transaction data
    group_col : str
        Column name to group by (default: "cc_num")
        
    Returns:
    --------
    List[Dict[str, Union[List[int], int]]]
        List of dictionaries with 'items' (transaction sequences) and 'label' (fraud status)
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
    
    # Check for required columns
    required_cols = [group_col, 'transaction_type_id', 'trans_date_trans_time', 'is_fraud']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Required columns {missing_cols} not found in data")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Group, sort, and extract sequences with labels
    sequences_with_labels = []
    groups = df.groupby(group_col)
    
    for name, group in groups:
        # Sort by transaction time
        sorted_group = group.sort_values(by='trans_date_trans_time')
        
        # Extract transaction sequence
        sorted_sequence = sorted_group['transaction_type_id'].tolist()
        
        # Determine fraud label for the sequence
        # Option 1: If ANY transaction in the sequence is fraud, mark as fraud
        is_fraud_sequence = int(sorted_group['is_fraud'].any())
        
        # Option 2: Alternative - if MAJORITY of transactions are fraud
        # is_fraud_sequence = int(sorted_group['is_fraud'].mean() > 0.5)
        
        # Option 3: Alternative - if LAST transaction is fraud (recency bias)
        # is_fraud_sequence = int(sorted_group['is_fraud'].iloc[-1])
        
        # Create the format expected by the collator
        sequence_data = {
            'items': sorted_sequence,
            'label': is_fraud_sequence
        }
        
        sequences_with_labels.append(sequence_data)
    
    print(f"Extracted {len(sequences_with_labels)} sequences based on '{group_col}'.")
    
    # Print fraud distribution
    fraud_count = sum(1 for seq in sequences_with_labels if seq['label'] == 1)
    normal_count = len(sequences_with_labels) - fraud_count
    print(f"Fraud sequences: {fraud_count}, Normal sequences: {normal_count}")
    print(f"Fraud ratio: {fraud_count / len(sequences_with_labels):.3f}")
    
    return sequences_with_labels

def save_sequences_to_json(sequences: List[Dict[str, Union[List[int], int]]], filename: str):
    """
    Save sequences with labels to JSON file.
    
    Parameters:
    -----------
    sequences : List[Dict[str, Union[List[int], int]]]
        List of sequence dictionaries with 'items' and 'label' keys
    filename : str
        Output filename
    """
    with open(filename, 'w') as f:
        json.dump(sequences, f, indent=2)
    print(f"Saved {len(sequences)} sequences to {filename}")

def main():
    input_file_path = '../data/credit_card_transaction_train_processed.csv'
    
    # Extract sequences with fraud labels
    all_sequences = extract_interaction_from_file(input_file_path)
    
    # Split into train/validation
    train_idx = int(len(all_sequences) * 0.85)
    train_sequences = all_sequences[:train_idx]
    val_sequences = all_sequences[train_idx:]
    
    print(f"Total sequences: {len(all_sequences)}")
    print(f"Training sequences: {len(train_sequences)}, Validation sequences: {len(val_sequences)}")
    
    # Print fraud distribution in splits
    train_fraud = sum(1 for seq in train_sequences if seq['label'] == 1)
    val_fraud = sum(1 for seq in val_sequences if seq['label'] == 1)
    
    print(f"Train fraud ratio: {train_fraud / len(train_sequences):.3f}")
    print(f"Val fraud ratio: {val_fraud / len(val_sequences):.3f}")


    input_file_path_test = '../data/credit_card_transaction_test_processed.csv'
    all_sequences_test = extract_interaction_from_file(input_file_path)



    # Save the sequences to files
    save_sequences_to_json(train_sequences, 'train.json')
    save_sequences_to_json(val_sequences, 'dev.json')
    save_sequences_to_json(all_sequences_test, 'test.json')

if __name__ == "__main__":
    main()