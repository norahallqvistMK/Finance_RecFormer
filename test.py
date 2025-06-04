from typing import List, Dict, Union
from dataloader import RecformerTrainDatasetFraud, RecformerEvalDatasetFraud
from collator import FinetuneDataCollatorWithPaddingFraud, EvalDataCollatorWithPaddingFraud

# Usage example function
def create_fraud_dataloaders(train_json_path, val_json_path, test_json_path, 
                           train_collator, eval_collator, batch_size=32):
    '''
    Helper function to create dataloaders from JSON files
    '''
    import json
    from torch.utils.data import DataLoader
    
    # Load data from JSON files
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)
    
    with open(val_json_path, 'r') as f:
        val_data = json.load(f)
        
    with open(test_json_path, 'r') as f:
        test_data = json.load(f)
    
    # Create datasets
    train_dataset = RecformerTrainDatasetFraud(train_data, train_collator)
    val_dataset = RecformerEvalDatasetFraud(val_data, eval_collator)
    test_dataset = RecformerEvalDatasetFraud(test_data, eval_collator)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=val_dataset.collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )
    
    return train_loader, val_loader, test_loader



create_fraud_dataloaders(train_json_path='finetune_data/train.json',
    val_json_path='finetune_data/dev.json',
    test_json_path='finetune_data/test.json',
    train_collator=FinetuneDataCollatorWithPaddingFraud,
    eval_collator=EvalDataCollatorWithPaddingFraud  ,
    batch_size=1
)