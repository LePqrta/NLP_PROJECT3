import argparse
import json
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    DataCollatorForTokenClassification, 
    TrainingArguments, 
    Trainer
)

def main(args):
    # --- 1. Load the Data Created by prepare_data.py ---
    print("Loading datasets...")
    data_files = {
        "train": "train.json",
        "validation": "validation.json"
    }
    dataset = load_dataset("json", data_files=data_files)

    # --- 2. Create Label Mappings from the Data ---
    # We scan the data to find all unique tag names
    print("Extracting labels...")
    unique_tags = set()
    for split in dataset.keys():
        for tags in dataset[split]["ner_tags"]:
            unique_tags.update(tags)
    
    label_list = sorted(list(unique_tags))
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    
    print(f"Labels: {label_list}")

    # --- 3. Tokenizer & Model Setup ---
    model_checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def tokenize_and_align_labels(examples):
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
                    # Convert string label to ID
                    label_ids.append(label2id[label[word_idx]])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    print("Tokenizing data...")
    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    # --- 4. Metrics ---
    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # --- 5. Training ---
    args_train = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",       # Evaluate every epoch
        save_strategy="epoch",       # Save every epoch
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=int(args.num_train_epoch),
        weight_decay=0.01,
        load_best_model_at_end=True, # Load best model at the end
        metric_for_best_model="f1"
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {args.model_save_path}...")
    trainer.save_model(args.model_save_path)
    tokenizer.save_pretrained(args.model_save_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=False, help="Unused in this version, hardcoded to train.json")
    parser.add_argument("--num_train_epoch", type=int, default=3)
    args = parser.parse_args()
    main(args)