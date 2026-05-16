import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# --- Configuration ---
# Point this to the model you saved after Track B fine-tuning
MODEL_PATH = "../models/bge-narrative-tuned-10k" 
TEST_DATA_PATH = "../dataset/test_track_a.jsonl"
TEST_LABELS_PATH = "../dataset/test_track_a_labels.jsonl"

def main():
    print(f"Loading fine-tuned MiniLM from {MODEL_PATH}...")
    model = SentenceTransformer(MODEL_PATH)
    
    # Load Track A test data and labels
    df_test = pd.read_json(TEST_DATA_PATH, lines=True)
    df_labels = pd.read_json(TEST_LABELS_PATH, lines=True)

    predictions = []
    
    print("Running Track A evaluation using Bi-Encoder embeddings...")
    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        # 1. Encode all three stories independently (Bi-Encoder style)
        # Your H100 will process these 384-dim vectors instantly
        embeddings = model.encode(
            [row["anchor_text"], row["text_a"], row["text_b"]], 
            convert_to_tensor=True
        )
        
        # 2. Calculate Cosine Similarity between Anchor and both candidates
        sim_a = util.cos_sim(embeddings[0], embeddings[1]).item()
        sim_b = util.cos_sim(embeddings[0], embeddings[2]).item()
        
        # 3. Predict 'True' (Text A is closer) if its similarity is higher
        predictions.append(sim_a > sim_b)

    # Calculate Accuracy
    accuracy = accuracy_score(df_labels["text_a_is_closer"], predictions)
    
    print("\n========================================")
    print(f"MiniLM Track A Accuracy: {accuracy * 100:.2f}%")
    print(f"========================================")

if __name__ == "__main__":
    main()