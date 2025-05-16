#!/bin/bash
export OPENAI_API_KEY="OPENAI API KEY HERE"
export HF_TOKEN="hf token here"

dataset_name="./input_data/Qwen2_5-14B-Instruct_query.csv"
output_file="./output"
size=200
search_mode="embedding"
embedder_name="gpt"
index_name="us_gaap_taxonomy"
index_document="./index_document/us_gaap_2024_final_embedder.jsonl"
ks="1,5,10,20,30,40,50,60,70,80,90,100,150,200"

python retrieval.py \
  --dataset_name "$dataset_name" \
  --output_file "$output_file" \
  --size "$size" \
  --search_mode "$search_mode" \
  --embedder_name "$embedder_name" \
  --index_name "$index_name" \
  --index_document "$index_document" \
  # --ks "$ks"
  # --eval
