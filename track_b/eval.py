import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# --- Configuration ---
MODEL_PATH = "../models/bge-narrative-tuned-10k"
TEST_LABELS_PATH = "../dataset/test_track_b_labels.jsonl"

def evaluate_on_test():
    print(f"Loading fine-tuned model from {MODEL_PATH}...")
    model = SentenceTransformer(MODEL_PATH)
    
    # Load the official test labels
    df = pd.read_json(TEST_LABELS_PATH, lines=True)
    
    correct_predictions = 0
    total = len(df)

    print("Encoding test stories and calculating similarities...")
    for _, row in tqdm(df.iterrows(), total=total):
        anchor = row["anchor_text"]
        
        # Identify Positive and Negative candidates
        if row["text_a_is_closer"]:
            pos_text, neg_text = row["text_a"], row["text_b"]
        else:
            pos_text, neg_text = row["text_b"], row["text_a"]

        # 1. Encode texts to embeddings
        # Note: We encode all at once for speed
        embeddings = model.encode([anchor, pos_text, neg_text], convert_to_tensor=True)
        
        # 2. Compute Cosine Similarity
        # util.cos_sim returns a matrix; we extract the scalar scores
        sim_pos = util.cos_sim(embeddings[0], embeddings[1]).item()
        sim_neg = util.cos_sim(embeddings[0], embeddings[2]).item()

        # 3. Check if the model correctly ranked the 'Similar' story higher
        if sim_pos > sim_neg:
            correct_predictions += 1

    accuracy = (correct_predictions / total) * 100
    print(f"\n========================================")
    print(f"TRACK B TEST ACCURACY: {accuracy:.2f}%")
    print(f"========================================")

if __name__ == "__main__":
    evaluate_on_test()