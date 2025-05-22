#!/bin/bash

# 设置参数
TEST_DATA="annotation/BIO_data/test_data_benchmark.bio"
LABEL2ID="annotation/BIO_data/label2id.json"
ID2LABEL="annotation/BIO_data/id2label.json"
MODEL_DIR="/vast/palmer/scratch/xu_hua/yw937/merged_model/FT-bert-large-uncased"
BATCH_SIZE=4
OUTPUT_CSV="./FT-bert-prediction-all.csv"

# 启动模型评估
python evaluate_ner_model.py \
  --test_data "$TEST_DATA" \
  --label2id "$LABEL2ID" \
  --id2label "$ID2LABEL" \
  --model_dir "$MODEL_DIR" \
  --batch_size "$BATCH_SIZE" \
  --predict_csv "$OUTPUT_CSV"
