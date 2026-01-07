import argparse
import json
import os
import torch
from transformers import pipeline

def main(args):
    # 1. Check if model exists
    if not os.path.exists(args.model_load_path):
        print(f"Error: Model not found at {args.model_load_path}")
        return

    print(f"Loading model from {args.model_load_path}...")
    
    # --- INTERACTIVE DEVICE SELECTION ---
    device = -1  # Default to CPU
    if torch.cuda.is_available():
        print(f"\n>> GPU Detected: {torch.cuda.get_device_name(0)}")
        user_choice = input(">> Do you want to use the GPU? (y/n): ").strip().lower()
        if user_choice == 'y':
            device = 0
            print(">> Using GPU.\n")
        else:
            print(">> Using CPU.\n")
    else:
        print("\n>> No GPU detected. Using CPU.\n")
    # ------------------------------------

    # 2. Initialize the Pipeline
    # aggregation_strategy="simple" is CRITICAL. 
    # It merges "Elon" and "Musk" into "Elon Musk" (PERSON) automatically.
    nlp = pipeline(
        "token-classification", 
        model=args.model_load_path, 
        tokenizer=args.model_load_path, 
        aggregation_strategy="simple",  
        device=device 
    )

    # 3. Read Input Sentences
    print(f"Reading sentences from {args.input_file}...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    results = []
    
    # 4. Run Predictions
    print(f"Predicting entities for {len(sentences)} sentences...")
    for sentence in sentences:
        predictions = nlp(sentence)
        
        # Convert numpy floats to standard floats for JSON serialization
        formatted_preds = []
        for pred in predictions:
            formatted_preds.append({
                "entity_group": pred["entity_group"],
                "score": float(pred["score"]),
                "word": pred["word"],
                "start": pred["start"],
                "end": pred["end"]
            })
            
        results.append({
            "input": sentence,
            "entities": formatted_preds
        })

    # 5. Save Results
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_load_path", type=str, required=True, help="Path to the saved model folder")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the text file containing test sentences")
    parser.add_argument("--output_file", type=str, default="final_results.json", help="Path to save the JSON output")
    args = parser.parse_args()
    main(args)