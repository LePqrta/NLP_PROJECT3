import json
import random
import os
from datasets import load_dataset
from transformers import AutoConfig

def main():
    print("--- Phase 1: Processing Dataset 1 (Ontonotes) ---")
    
    # 1. Load Dataset 1
    print("Loading Ontonotes...")
    dataset1 = load_dataset("tner/ontonotes5", split="train")

    # 2. Get the Label Mapping
    print("Fetching label mapping...")
    config = AutoConfig.from_pretrained("tner/roberta-large-ontonotes5")
    id2label = config.id2label
    # Create the reverse map just in case
    # label2id = {v: k for k, v in id2label.items()}

    # 3. Split 85% - 15%
    print("Splitting data...")
    dataset1 = dataset1.shuffle(seed=42)
    split_index = int(len(dataset1) * 0.85)

    ds1_train_raw = dataset1.select(range(split_index))
    ds1_val_raw = dataset1.select(range(split_index, len(dataset1)))

    print(f"Train Split (Dataset1): {len(ds1_train_raw)}")
    print(f"Validation Split: {len(ds1_val_raw)}")

    # 4. Process Validation Set (Convert to String Labels & Save)
    val_data = []
    for item in ds1_val_raw:
        # Convert IDs to Strings (e.g., 5 -> 'B-ORG')
        tags_str = [id2label[t] for t in item['tags']]
        val_data.append({
            "tokens": item['tokens'],
            "ner_tags": tags_str
        })
    
    with open("validation.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f)
    print(">> Saved 'validation.json' (Use this for Classification Report)")

    # 5. Process Train Split (Convert to String Labels)
    train_data = []
    for item in ds1_train_raw:
        tags_str = [id2label[t] for t in item['tags']]
        train_data.append({
            "tokens": item['tokens'],
            "ner_tags": tags_str
        })

    print("\n--- Phase 2: Merging Dataset 2 ---")
    
    # 6. Load Dataset 2 (Your SpaceX/Custom Data)
    # NOTE: Ensure this file is in the same folder
    ds2_filename = "dataset2_formatted.json"    
    if os.path.exists(ds2_filename):
        print(f"Found {ds2_filename}...")
        with open(ds2_filename, 'r', encoding='utf-8') as f:
            ds2_raw = json.load(f)
            
        # Add Dataset 2 to Train Data
        count = 0
        for item in ds2_raw:
            # Handle different key names just in case
            tokens = item.get('tokens') or item.get('raw_text', '').split()
            tags = item.get('ner_tags') or item.get('gold_annotations', [])
            
            # Ensure tags are strings
            final_tags = []
            for t in tags:
                if isinstance(t, int):
                    final_tags.append(id2label.get(t, "O"))
                else:
                    final_tags.append(str(t))
            
            train_data.append({
                "tokens": tokens,
                "ner_tags": final_tags
            })
            count += 1
        print(f"Added {count} examples from Dataset 2.")
    else:
        print(f"WARNING: {ds2_filename} not found. Using only Dataset 1.")

    # Shuffle combined data
    random.shuffle(train_data)

    # 7. Save Final Training Data
    print(f"Saving {len(train_data)} training examples...")
    with open("train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f)
    print(">> Saved 'train.json' (Ready for training)")

if __name__ == "__main__":
    main()