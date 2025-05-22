#!/bin/bash

TRAIN_DATA="annotation/BIO_data/train_data_all.bio"
LABEL2ID="annotation/BIO_data/label2id.json"
ID2LABEL="annotation/BIO_data/id2label.json"
MODEL_NAME="google-bert/bert-large-uncased"
CHECKPOINT_DIR="/vast/palmer/scratch/xu_hua/yw937/model_checkpoint/bert-large-uncased"
MODEL_DIR="/vast/palmer/scratch/xu_hua/yw937/merged_model/FT-bert-large-uncased"
BATCH_SIZE=4
EPOCHS=20
LEARNING_RATE=3e-5
NUM_GPUS=2  # 使用 GPU 数量

# ======= 启动训练（使用 torchrun 多卡） ========
torchrun --nproc_per_node=$NUM_GPUS train_ner_model.py \
  --train_data $TRAIN_DATA \
  --label2id $LABEL2ID \
  --id2label $ID2LABEL \
  --model_name $MODEL_NAME \
  --checkpoint_dir $CHECKPOINT_DIR \
  --saved_dir $MODEL_DIR \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --lr "$LEARNING_RATE"
