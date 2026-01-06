import json
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer

def main():
    # 1. Configuration
    model_path = "./my_model"
    data_file = "validation.json"
    
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    print(f"Loading validation data from {data_file}...")
    data_files = {"validation": data_file}
    dataset = load_dataset("json", data_files=data_files)
    
    # 2. Get Label List from Model Config
    # The model saved the ID->Label mapping, so we load it directly
    id2label = model.config.id2label
    label_list = [id2label[i] for i in range(len(id2label))]
    label2id = model.config.label2id

    # 3. Preprocess Data (Tokenization)
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
                    # If label is in our map, use it. If not (rare), ignore.
                    if label[word_idx] in label2id:
                        label_ids.append(label2id[label[word_idx]])
                    else:
                        label_ids.append(-100)
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_dataset = dataset["validation"].map(tokenize_and_align_labels, batched=True)
    
    # 4. Predict
    print("Running predictions (this uses the GPU if available)...")
    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(model=model, data_collator=data_collator)
    
    predictions, labels, _ = trainer.predict(tokenized_dataset)
    predictions = np.argmax(predictions, axis=2)

    # 5. Decode & Generate Report
    print("Generating report...")
    
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Load seqeval for the detailed report
    seqeval = evaluate.load("seqeval")
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    
    # Print the Summary
    print("\n" + "="*30)
    print("OVERALL SCORES")
    print("="*30)
    print(f"Accuracy:  {results['overall_accuracy']:.4f}")
    print(f"Precision: {results['overall_precision']:.4f}")
    print(f"Recall:    {results['overall_recall']:.4f}")
    print(f"F1 Score:  {results['overall_f1']:.4f}")
    
    print("\n" + "="*30)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*30)
    
    # Seqeval returns a dictionary for each key, we format it nicely
    # Get all keys that are not 'overall_...'
    categories = sorted([k for k in results.keys() if not k.startswith("overall_")])
    
    print(f"{'ENTITY':<15} {'PRECISION':<10} {'RECALL':<10} {'F1-SCORE':<10} {'NUMBER':<10}")
    print("-" * 60)
    for cat in categories:
        metrics = results[cat]
        print(f"{cat:<15} {metrics['precision']:<10.2f} {metrics['recall']:<10.2f} {metrics['f1']:<10.2f} {metrics['number']:<10}")

if __name__ == "__main__":
    main()