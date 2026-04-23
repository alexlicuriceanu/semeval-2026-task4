import json
import random
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import pandas as pd

import torch.nn.functional as F
import torch

# --- FIX 3: USE THE NLI PRE-TRAINED CROSS ENCODER ---
MODEL_NAME = "cross-encoder/nli-deberta-v3-large"
BATCH_SIZE = 16 
NUM_EPOCHS = 10
TRAIN_DATA_PATH = "../dataset/synthetic_data_for_classification.jsonl"
DEV_DATA_PATH = "../dataset/test_track_a.jsonl" 
DEV_LABELS_PATH = "../dataset/test_track_a_labels.jsonl" 

def load_and_filter_training_data(file_path):
    train_examples = []
    filtered_count = 0
    total_count = 0

    print("Loading and filtering training data...")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            total_count += 1
            
            anchor = data.get("anchor_text") or ""
            text_a = data.get("text_a") or ""
            text_b = data.get("text_b") or ""
            a_is_closer = data.get("text_a_is_closer")

            similar = text_a if a_is_closer else text_b
            dissimilar = text_b if a_is_closer else text_a

            len_anchor = len(anchor.split())
            len_similar = len(similar.split())
            
            # --- FIX 2: CORRECT THE MATH LOGIC ---
            # Drop pairs where the word count difference is less than 3
            if abs(len_anchor - len_similar) < 3:
                filtered_count += 1
                continue 

            train_examples.append(InputExample(texts=[anchor, similar], label=1))
            train_examples.append(InputExample(texts=[anchor, dissimilar], label=0))
            
    print(f"Total Triples: {total_count}")
    print(f"Filtered out {filtered_count} 'lazy' biased triples.")
    return train_examples

def main():
    train_examples = load_and_filter_training_data(TRAIN_DATA_PATH)
    random.shuffle(train_examples)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

    print(f"\nInitializing CrossEncoder: {MODEL_NAME}...")
    model = CrossEncoder(MODEL_NAME, num_labels=3, max_length=512)

    print("Starting fine-tuning...")
    model.fit(
        train_dataloader=train_dataloader,
        epochs=NUM_EPOCHS,               
        warmup_steps=200,       
        optimizer_params={'lr': 2e-6}, 
        output_path="models/deberta-narrative-classifier"
    )

    print("\n--- Evaluating on Human Dev Set ---")
    try:
        df_dev = pd.read_json(DEV_DATA_PATH, lines=True)
        df_labels = pd.read_json(DEV_LABELS_PATH, lines=True)
        
        predictions = []
        for _, row in df_dev.iterrows():
            # Get logits for both candidates
            logits_a = model.predict([[row["anchor_text"], row["text_a"]]])
            logits_b = model.predict([[row["anchor_text"], row["text_b"]]])
            
            # Convert both to probabilities using Softmax
            probs_a = F.softmax(torch.tensor(logits_a), dim=1).tolist()[0]
            probs_b = F.softmax(torch.tensor(logits_b), dim=1).tolist()[0]
            
            # Index 1 is 'Entailment' (Similar)
            score_a = probs_a[1]
            score_b = probs_b[1]
            
            predictions.append(score_a > score_b)

        accuracy = accuracy_score(df_labels["text_a_is_closer"], predictions)
        print(f"DeBERTa Cross-Encoder Accuracy on Dev Set: {accuracy * 100:.2f}%")
        
    except FileNotFoundError:
        print("Dev data not found.")

if __name__ == "__main__":
    main()