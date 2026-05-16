import json
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader

# --- UPGRADED CONFIGURATION ---
MODEL_NAME = "BAAI/bge-large-en-v1.5" 
BATCH_SIZE = 32 # If your H100 has 80GB VRAM, you can even push this to 64 or 128!
NUM_EPOCHS = 3  # Lowered slightly. With 10k+ examples, 3 epochs is usually plenty to prevent overfitting.

# List all your training files here to combine them!
TRAIN_DATA_PATHS = [
    "./dataset/synthetic_data_for_contrastive_learning.jsonl", # Your old data
    "./dataset_augmentation/augmented_track_b_train.jsonl"                      # Your massive new dataset
]

DEV_LABELS_PATH = "./dataset/test_track_b_labels.jsonl" 

def load_native_triplet_data(file_paths):
    """
    Parses multiple Track B synthetic data files and combines them.
    """
    examples = []
    
    for file_path in file_paths:
        print(f"Loading native Triplet data from {file_path}...")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    
                    anchor = data.get("anchor_story")
                    positive = data.get("similar_story")
                    negative = data.get("dissimilar_story")

                    if anchor and positive and negative:
                        examples.append(InputExample(texts=[anchor, positive, negative]))
        except FileNotFoundError:
            print(f"Warning: Could not find {file_path}. Skipping...")

    print(f"Loaded a massive total of {len(examples)} training triplets!")
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
    model = SentenceTransformer(MODEL_NAME)

    # 1. Load combined data
    train_examples = load_native_triplet_data(TRAIN_DATA_PATHS)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

    # 2. Setup MNRL with HARD NEGATIVES
    # Because our InputExample contains 3 texts, MNRL will use the 3rd text 
    # as a Hard Negative, which drastically improves performance!
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # 3. Setup Evaluator
    dev_examples = load_dev_data(DEV_LABELS_PATH)
    evaluator = TripletEvaluator.from_input_examples(dev_examples, name='track-b-eval')

    print("\nStarting MNRL fine-tuning...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=NUM_EPOCHS,
        evaluation_steps=200, # Check progress every 200 steps
        warmup_steps=100,
        output_path="models/bge-narrative-tuned-10k",
        optimizer_params={'lr': 1e-5} 
    )
    
    print("\nTraining complete! Custom model saved to 'models/bge-narrative-tuned-10k'")

if __name__ == "__main__":
    main()