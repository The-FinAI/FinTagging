#!/usr/bin/env python
import os
import json
import argparse
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    BertConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

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
    tokenized_inputs.pop("tokens", None)
    return tokenized_inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--label2id", type=str, required=True)
    parser.add_argument("--id2label", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="bert-base-cased")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--saved_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    with open(args.label2id) as f:
        label2id = json.load(f)
    with open(args.id2label) as f:
        id2label = json.load(f)
    num_labels = len(label2id)

    train_tokens, train_labels = read_bio_file(args.train_data)
    train_dataset = Dataset.from_dict({
        "tokens": train_tokens,
        "ner_tags": [[label2id[label] for label in seq] for seq in train_labels]
    })

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_train = train_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id), batched=True
    )

    config = BertConfig.from_pretrained(args.model_name, num_labels=num_labels, id2label=id2label, label2id=label2id)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, config=config, ignore_mismatched_sizes=True)

    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        save_strategy="epoch",
        save_total_limit=2,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_dir=os.path.join(args.checkpoint_dir, "logs"),
        logging_steps=10,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )

    trainer.train()
    trainer.save_model(args.saved_dir)

if __name__ == "__main__":
    main()
