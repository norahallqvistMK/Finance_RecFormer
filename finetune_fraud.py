import os
import json
import torch
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, precision_score, recall_score, f1_score

from pytorch_lightning import seed_everything
from functools import partial

from utils import read_json, AverageMeterSet
from optimization import create_optimizer_and_scheduler
from recformer import RecformerModel, RecformerTokenizer, RecformerConfig, RecformerForFraudDetection
from collator import EvalDataCollatorWithPaddingFraud, FinetuneDataCollatorWithPaddingFraud
from dataloader import RecformerEvalDatasetFraud, RecformerTrainDatasetFraud


def load_data(args):
    """Load fraud detection data"""
    train = json.load(open(os.path.join(args.data_path, args.train_file)))
    val = json.load(open(os.path.join(args.data_path, args.dev_file)))
    test = json.load(open(os.path.join(args.data_path, args.test_file)))
    item_meta_dict = json.load(open(os.path.join("pretrain_data", args.meta_file)))

    return train, val, test, item_meta_dict

# tokenizer_glb: RecformerTokenizer = None


tokenizer_glb = None

def _init_worker(model_name_or_path, num_labels):
    """
    Initializes the global tokenizer in each worker process.
    """
    global tokenizer_glb
    config = RecformerConfig.from_pretrained(model_name_or_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.max_item_embeddings = 51
    config.attention_window = [64] * 12
    config.max_token_num = 1024
    config.num_labels = num_labels  # For binary classification
    tokenizer_glb = RecformerTokenizer.from_pretrained(model_name_or_path, config)
    print(f'Worker initialized tokenizer: {model_name_or_path}')

def _par_tokenize_doc(doc):
    item_id, item_attr = doc
    input_ids, token_type_ids = tokenizer_glb.encode_item(item_attr)
    return item_id, input_ids, token_type_ids

def compute_fraud_metrics(predictions, labels):
    """Compute fraud detection metrics"""
    # Convert to numpy for sklearn
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Get predicted classes (0 or 1)
    pred_classes = (predictions > 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(labels, pred_classes),
        'precision': precision_score(labels, pred_classes, zero_division=0),
        'recall': recall_score(labels, pred_classes, zero_division=0),
        'f1': f1_score(labels, pred_classes, zero_division=0),
        'auc': roc_auc_score(labels, predictions) if len(set(labels)) > 1 else 0.0
    }
    
    return metrics


def eval_fraud(model, dataloader, args):
    """Evaluate fraud detection model"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, ncols=100, desc='Evaluate'):
            # Move batch to device
            for k, v in batch.items():
                batch[k] = v.to(args.device)
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            labels = batch['labels']
            
            # Apply sigmoid to get probabilities for binary classification
            predictions = torch.sigmoid(logits).squeeze()
            
            all_predictions.append(predictions)
            all_labels.append(labels)
            total_loss += loss.item()
            num_batches += 1
    
    # Concatenate all predictions and labels
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute metrics
    metrics = compute_fraud_metrics(all_predictions, all_labels)
    metrics['loss'] = total_loss / num_batches
    
    return metrics


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, args):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    for step, batch in enumerate(tqdm(dataloader, ncols=100, desc='Training')):
        # Move batch to device
        for k, v in batch.items():
            batch[k] = v.to(args.device)

        if args.fp16:
            with autocast():
                outputs = model(**batch)
                loss = outputs.loss
        else:
            outputs = model(**batch)
            loss = outputs.loss

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item()
        num_batches += 1

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                optimizer_was_run = scale_before <= scale_after
                optimizer.zero_grad()

                if optimizer_was_run:
                    scheduler.step()
            else:
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    parser = ArgumentParser()
    # Path and file arguments
    parser.add_argument('--pretrain_ckpt', type=str, default=None, required=True)
    parser.add_argument('--data_path', type=str, default=None, required=True)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--ckpt', type=str, default='best_fraud_model.bin')
    parser.add_argument('--model_name_or_path', type=str, default='allenai/longformer-base-4096')
    parser.add_argument('--train_file', type=str, default='train.json')
    parser.add_argument('--dev_file', type=str, default='dev.json')
    parser.add_argument('--test_file', type=str, default='test.json')
    parser.add_argument('--item2id_file', type=str, default='smap.json')
    parser.add_argument('--meta_file', type=str, default='meta_data.json')

    # Data processing
    parser.add_argument('--preprocessing_num_workers', type=int, default=8)
    parser.add_argument('--dataloader_num_workers', type=int, default=0)

    # Model
    parser.add_argument('--num_labels', type=int, default=2, help="Number of fraud classes (2 for binary)")
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fix_word_embedding', action='store_true')
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--early_stopping_patience', type=int, default=3)
    parser.add_argument('--metric_for_best_model', type=str, default='f1', 
                       choices=['accuracy', 'precision', 'recall', 'f1', 'auc'])

    args = parser.parse_args()
    print(args)
    seed_everything(42)
    args.device = torch.device('cuda:{}'.format(args.device)) if args.device >= 0 else torch.device('cpu')

    # Load data
    train, val, test, item_meta_dict = load_data(args)

    print(f"Loaded {len(train)} train, {len(val)} val, {len(test)} test samples")

    # Setup model config
    config = RecformerConfig.from_pretrained(args.model_name_or_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.max_item_embeddings = 51
    config.attention_window = [64] * 12
    config.max_token_num = 1024
    config.num_labels = args.num_labels  # For binary classification
    config.classifier_dropout = args.dropout
    tokenizer = RecformerTokenizer.from_pretrained(args.model_name_or_path, config)

    # global tokenizer_glb
    # tokenizer_glb = tokenizer

    # Setup paths
    path_corpus = Path(args.data_path)
    dir_preprocess = path_corpus / 'preprocess'
    dir_preprocess.mkdir(exist_ok=True)

    path_output = Path(args.output_dir) / path_corpus.name
    path_output.mkdir(exist_ok=True, parents=True)
    path_ckpt = path_output / args.ckpt

    path_tokenized_items = dir_preprocess / f'tokenized_items_{path_corpus.name}'

    # Preprocess items
    if path_tokenized_items.exists():
        print(f'[Preprocessor] Use cache: {path_tokenized_items}')
    else:
        print(f'Loading attribute data {path_corpus}')
        with Pool(processes=args.preprocessing_num_workers, initializer=_init_worker, initargs=(args.model_name_or_path, args.num_labels)) as pool:
            pool_func = pool.imap(func=_par_tokenize_doc, iterable=item_meta_dict.items())
            doc_tuples = list(tqdm(pool_func, total=len(item_meta_dict), ncols=100, desc=f'[Tokenize] {path_corpus}'))
        # pool = Pool(processes=args.preprocessing_num_workers)
        
        # tokenize_func = partial(_par_tokenize_doc, tokenizer_glb=tokenizer)
        # pool_func = pool.imap(func=tokenize_func, iterable=item_meta_dict.items())

        # doc_tuples = list(tqdm(pool_func, total=len(item_meta_dict), ncols=100, desc=f'[Tokenize] {path_corpus}'))
        tokenized_items = {item_id: [input_ids, token_type_ids] for item_id, input_ids, token_type_ids in doc_tuples}
        pool.close()
        pool.join()
        torch.save(tokenized_items, path_tokenized_items)

    tokenized_items = torch.load(path_tokenized_items)
    print(f'Successfully load {len(tokenized_items)} tokenized items.')

    tokenized_items = {int(k): v for k, v in item_meta_dict.items()}

    # Create data collators for fraud detection
    finetune_data_collator = FinetuneDataCollatorWithPaddingFraud(tokenizer, tokenized_items)
    eval_data_collator = EvalDataCollatorWithPaddingFraud(tokenizer, tokenized_items)

    # Create datasets
    train_data = RecformerEvalDatasetFraud(train, collator=finetune_data_collator)
    val_data = RecformerEvalDatasetFraud(val, collator=eval_data_collator)
    test_data = RecformerEvalDatasetFraud(test, collator=eval_data_collator)

    # Create data loaders
    train_loader = DataLoader(
        train_data, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=train_data.collate_fn,
        num_workers=args.dataloader_num_workers
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=args.batch_size, 
        collate_fn=val_data.collate_fn,
        num_workers=args.dataloader_num_workers
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=args.batch_size, 
        collate_fn=test_data.collate_fn,
        num_workers=args.dataloader_num_workers
    )

    # Initialize model for sequence classification (fraud detection)
    model = RecformerForFraudDetection(config)
    
    # Load pretrained weights
    if args.pretrain_ckpt:
        pretrain_ckpt = torch.load(args.pretrain_ckpt, map_location='cpu')
        # Load with strict=False to handle missing classification head weights
        missing_keys, unexpected_keys = model.load_state_dict(pretrain_ckpt, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    
    model.to(args.device)

    if args.fix_word_embedding:
        print('Fix word embeddings.')
        for param in model.longformer.embeddings.word_embeddings.parameters():
            param.requires_grad = False

    # Setup optimizer and scheduler
    num_train_optimization_steps = int(len(train_loader) / args.gradient_accumulation_steps) * args.num_train_epochs
    optimizer, scheduler = create_optimizer_and_scheduler(model, num_train_optimization_steps, args)
    
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # Initial evaluation
    print("Initial evaluation on test set:")
    test_metrics = eval_fraud(model, test_loader, args)
    print(f'Test set (before training): {test_metrics}')
    
    # Training loop
    best_metric = float('-inf')
    patience_counter = 0
    
    for epoch in range(args.num_train_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")
        
        # Train one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, args)
        print(f'Training loss: {train_loss:.4f}')
        
        # Evaluate on validation set
        if (epoch + 1) % args.verbose == 0 or epoch == args.num_train_epochs - 1:
            val_metrics = eval_fraud(model, val_loader, args)
            print(f'Validation metrics: {val_metrics}')
            
            # Check if this is the best model
            current_metric = val_metrics[args.metric_for_best_model]
            if current_metric > best_metric:
                print(f'New best {args.metric_for_best_model}: {current_metric:.4f} (previous: {best_metric:.4f})')
                best_metric = current_metric
                patience_counter = 0
                
                # Save best model
                print('Saving best model...')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_metric': best_metric,
                    'val_metrics': val_metrics
                }, path_ckpt)
            else:
                patience_counter += 1
                print(f'No improvement. Patience: {patience_counter}/{args.early_stopping_patience}')
                
                if patience_counter >= args.early_stopping_patience:
                    print('Early stopping triggered!')
                    break

    # Load best model and evaluate on test set
    print('\nLoading best model for final evaluation...')
    checkpoint = torch.load(path_ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print('Final evaluation on test set:')
    test_metrics = eval_fraud(model, test_loader, args)
    print(f'Final test metrics: {test_metrics}')
    
    # Save final results
    results = {
        'best_val_metric': best_metric,
        'final_test_metrics': test_metrics,
        'args': vars(args)
    }
    
    results_path = path_output / 'fraud_detection_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'Results saved to: {results_path}')

if __name__ == "__main__":
    main()