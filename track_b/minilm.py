import json
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader

# --- Configuration ---
# --- UPGRADED CONFIGURATION ---
MODEL_NAME = "BAAI/bge-large-en-v1.5" # 15x larger, much smarter
BATCH_SIZE = 32 # Increased for MNRL power (H100 can handle this)
NUM_EPOCHS = 4

# Ensure these match your actual folder structure
TRAIN_DATA_PATH = "../dataset/synthetic_data_for_contrastive_learning.jsonl"
DEV_LABELS_PATH = "../dataset/test_track_b_labels.jsonl" # Using test labels for evaluation during training

def load_native_triplet_data(file_path):
    """
    Parses the dedicated Track B synthetic data.
    The data is already perfectly formatted as Anchor, Similar, and Dissimilar.
    """
    examples = []
    print(f"Loading native Triplet data from {file_path}...")
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            
            anchor = data.get("anchor_story")
            positive = data.get("similar_story")
            negative = data.get("dissimilar_story")

            if anchor and positive and negative:
                examples.append(InputExample(texts=[anchor, positive, negative]))
                
    print(f"Loaded {len(examples)} training triplets.")
    return examples

def load_dev_data(labels_path):
    """
    Loads the evaluation labels to monitor progress during training.
    """
    df_labels = pd.read_json(labels_path, lines=True)
    dev_examples = []
    
    for _, row in df_labels.iterrows():
        anchor = row["anchor_text"]
        a_is_closer = row["text_a_is_closer"]
        
        # Determine positive and negative based on the boolean label
        positive = row["text_a"] if a_is_closer else row["text_b"]
        negative = row["text_b"] if a_is_closer else row["text_a"]
        
        dev_examples.append(InputExample(texts=[anchor, positive, negative]))
        
    return dev_examples

def main():
    print(f"Initializing Base Model: {MODEL_NAME}")
    # Initialize the larger model
    model = SentenceTransformer(MODEL_NAME)

    train_examples = load_native_triplet_data(TRAIN_DATA_PATH)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

    # --- UPGRADED LOSS FUNCTION ---
    # MNRL ignores the explicitly provided 'Negative' in the triplet, 
    # and instead uses the Positives from other pairs in the batch as Negatives.
    # To use it, we just pass the dataloader!
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    dev_examples = load_dev_data(DEV_LABELS_PATH)
    evaluator = TripletEvaluator.from_input_examples(dev_examples, name='track-b-eval')

    print("Starting MNRL fine-tuning...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=NUM_EPOCHS,
        evaluation_steps=100,
        warmup_steps=100,
        output_path="models/bge-narrative-tuned",
        optimizer_params={'lr': 1e-5} # Slightly lower learning rate for larger model
    )
    
    print("\nTraining complete! Custom model saved to 'models/bge-narrative-tuned'")

if __name__ == "__main__":
    main()