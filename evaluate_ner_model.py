#!/usr/bin/env python
import os
import json
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
)
from torch.utils.data import DataLoader
from seqeval.metrics import classification_report

def read_bio_file(filepath):
    all_tokens, all_labels = [], []
    with open(filepath, encoding='utf-8') as f:
        tokens, labels = [], []
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    all_tokens.append(tokens)
                    all_labels.append(labels)
                    tokens, labels = [], []
                continue
            splits = line.split()
            if len(splits) != 2:
                continue
            token, label = splits
            tokens.append(token)
            labels.append(label)
        if tokens:
            all_tokens.append(tokens)
            all_labels.append(labels)
    return all_tokens, all_labels

def tokenize_and_align_labels(examples, tokenizer, label2id):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def evaluate(model, dataset, tokenizer, label_list, original_tokens, batch_size=2, output_csv=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for evaluation")
        model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    collator = DataCollatorForTokenClassification(tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

    all_preds, all_labels = [], []
    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        label_ids = labels.numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(label_ids.tolist())

    results = []
    for tokens, pred_seq, label_seq in zip(original_tokens, all_preds, all_labels):
        token_idx = 0
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                results.append({
                    "token": tokens[token_idx],
                    "gold_label": label_list[l],
                    "predicted_label": label_list[p]
                })
                token_idx += 1
        results.append({})  # sentence break
    df = pd.DataFrame(results)
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Saved predictions to {output_csv}")

    true_labels = [
        [label_list[l] for (p_, l) in zip(pred, label) if l != -100]
        for pred, label in zip(all_preds, all_labels)
    ]
    true_preds = [
        [label_list[p_] for (p_, l) in zip(pred, label) if l != -100]
        for pred, label in zip(all_preds, all_labels)
    ]
    report = classification_report(true_labels, true_preds, output_dict=True)
    print(f"Micro Precision: {report['micro avg']['precision']:.4f}")
    print(f"Micro Recall:    {report['micro avg']['recall']:.4f}")
    print(f"Micro F1-score:  {report['micro avg']['f1-score']:.4f}")
    print()
    print(f"Macro Precision: {report['macro avg']['precision']:.4f}")
    print(f"Macro Recall:    {report['macro avg']['recall']:.4f}")
    print(f"Macro F1-score:  {report['macro avg']['f1-score']:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--label2id", type=str, required=True)
    parser.add_argument("--id2label", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--predict_csv", type=str, default="predictions.csv")
    args = parser.parse_args()

    with open(args.label2id) as f:
        label2id = json.load(f)
    with open(args.id2label) as f:
        id2label = json.load(f)
    label_list = list(label2id.keys())

    test_tokens, test_labels = read_bio_file(args.test_data)
    test_dataset = Dataset.from_dict({
        "tokens": test_tokens,
        "ner_tags": [[label2id[label] for label in seq] for seq in test_labels]
    })

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenized_test = test_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
        batched=True,
        remove_columns=["tokens", "ner_tags"]  # ✅ 确保移除原始字段，防止错误
    )

    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)

    evaluate(model, tokenized_test, tokenizer, label_list, test_tokens, args.batch_size, args.predict_csv)

if __name__ == "__main__":
    main()
