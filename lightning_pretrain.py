from torch.utils.tensorboard import SummaryWriter
import logging
from torch.utils.data import DataLoader
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
import json
import argparse
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from functools import partial

from recformer import RecformerForPretraining, RecformerTokenizer, RecformerConfig, LitWrapper
from collator import PretrainDataCollatorWithPadding
from lightning_dataloader import ClickDataset

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default=None)
parser.add_argument('--temp', type=float, default=0.05, help="Temperature for softmax.")
parser.add_argument('--preprocessing_num_workers', type=int, default=8, help="The number of processes to use for the preprocessing.")
parser.add_argument('--train_file', type=str, required=True)
parser.add_argument('--dev_file', type=str, required=True)
parser.add_argument('--item_attr_file', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--num_train_epochs', type=int, default=10)
parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
parser.add_argument('--dataloader_num_workers', type=int, default=2)
parser.add_argument('--mlm_probability', type=float, default=0.15)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument('--valid_step', type=int, default=500)
parser.add_argument('--log_step', type=int, default=100)
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--longformer_ckpt', type=str, default='longformer_ckpt/longformer-base-4096.bin')
parser.add_argument('--fix_word_embedding', action='store_true')




tokenizer_glb = None

def _init_worker(model_name_or_path):
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
    tokenizer_glb = RecformerTokenizer.from_pretrained(model_name_or_path, config)
    logger.info(f'Worker initialized tokenizer: {model_name_or_path}')

def _par_tokenize_doc(doc):
    item_id, item_attr = doc
    input_ids, token_type_ids = tokenizer_glb.encode_item(item_attr)
    return item_id, input_ids, token_type_ids

def main():
    args = parser.parse_args()
    print(args)
    seed_everything(42)

    # Main process tokenizer for logging and config
    config = RecformerConfig.from_pretrained(args.model_name_or_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.max_item_embeddings = 51
    config.attention_window = [64] * 12
    config.max_token_num = 1024
    tokenizer = RecformerTokenizer.from_pretrained(args.model_name_or_path, config)
    logger.info(f'Main process tokenizer config: {tokenizer.config}')

    # Preprocess corpus
    path_corpus = Path(args.item_attr_file)
    dir_corpus = path_corpus.parent
    dir_preprocess = dir_corpus / 'preprocess'
    dir_preprocess.mkdir(exist_ok=True)

    path_tokenized_items = dir_preprocess / f'tokenized_items_{path_corpus.name}'

    if path_tokenized_items.exists():
        print(f'[Preprocessor] Use cache: {path_tokenized_items}')
    else:
        print(f'Loading attribute data {path_corpus}')
        with open(path_corpus) as f:
            item_attrs = json.load(f)

        # Start Pool with initializer to set tokenizer_glb
        with Pool(processes=args.preprocessing_num_workers, initializer=_init_worker, initargs=(args.model_name_or_path,)) as pool:
            pool_func = pool.imap(func=_par_tokenize_doc, iterable=item_attrs.items())
            doc_tuples = list(tqdm(pool_func, total=len(item_attrs), ncols=100, desc=f'[Tokenize] {path_corpus}'))

        tokenized_items = {int(item_id): [input_ids, token_type_ids] for item_id, input_ids, token_type_ids in doc_tuples}
        with open(path_tokenized_items, 'w') as f:
            json.dump(tokenized_items, f)

    with open(path_tokenized_items) as f:
        tokenized_items = json.load(f)
    print(f'Successfully loaded {len(tokenized_items)} tokenized items.')

    tokenized_items = {int(k): v for k, v in  tokenized_items.items()}

    data_collator = PretrainDataCollatorWithPadding(tokenizer, tokenized_items, mlm_probability=args.mlm_probability)
    train_data = ClickDataset(json.load(open(args.train_file)), data_collator)
    dev_data = ClickDataset(json.load(open(args.dev_file)), data_collator)
    
    train_loader = DataLoader(train_data, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              collate_fn=train_data.collate_fn,
                              num_workers=args.dataloader_num_workers)
    dev_loader = DataLoader(dev_data, 
                            batch_size=args.batch_size, 
                            collate_fn=dev_data.collate_fn,
                            num_workers=args.dataloader_num_workers)
    pytorch_model = RecformerForPretraining(config)
    pytorch_model.load_state_dict(torch.load(args.longformer_ckpt))

    if args.fix_word_embedding:
        print('Fix word embeddings.')
        for param in pytorch_model.longformer.embeddings.word_embeddings.parameters():
            param.requires_grad = False

    model = LitWrapper(pytorch_model, learning_rate=args.learning_rate)

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="accuracy", mode="max", filename="{epoch}-{accuracy:.4f}")

    print("trainer being loaded")
    
    trainer = Trainer(accelerator="gpu",
                     max_epochs=args.num_train_epochs,
                     devices=args.device,
                     accumulate_grad_batches=args.gradient_accumulation_steps,
                     val_check_interval= len(train_loader),
                     default_root_dir=args.output_dir,
                     gradient_clip_val=1.0,
                     log_every_n_steps=None,
                     precision=16 if args.fp16 else 32,
                     strategy='deepspeed_stage_2',
                     callbacks=[checkpoint_callback]
                     )
    
    print("trainer being fiteed")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=dev_loader, ckpt_path=args.ckpt)

if __name__ == "__main__":
    main()