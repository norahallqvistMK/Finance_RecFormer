python finetune_fraud.py \
    --pretrain_ckpt pretrain_ckpt/fraud_pretrain_ckpt.bin \
    --data_path finetune_data \
    --num_train_epochs 2 \
    --batch_size 2 \
    --device 1 \
    --fp16 \