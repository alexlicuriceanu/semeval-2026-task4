import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader

# --- NEUROSYMBOLIC CONFIGURATION ---
MODEL_NAME = "BAAI/bge-large-en-v1.5" 
BATCH_SIZE = 32  # 32 is perfect for 1,900 items to give MNRL enough 'in-batch' negatives
NUM_EPOCHS = 6   # We can bump this back to 4 since the dataset is smaller (1.9k) and highly structured

# Ensure these point to your newly extracted TAO files
TRAIN_DATA_PATH = "./tao_synthetic_data_train.jsonl"

# CRITICAL: This dev file MUST also be run through your LLM TAO Extractor first!
DEV_LABELS_PATH = "./tao_test_track_b_labels.jsonl" 

def load_tao_training_data(file_path):
    """
    Parses the extracted TAO triplet data.
    """
    examples = []
    print(f"Loading TAO Training data from {file_path}...")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                
                anchor = data.get("anchor_story")
                positive = data.get("similar_story")
                negative = data.get("dissimilar_story")

                if anchor and positive and negative:
                    examples.append(InputExample(texts=[anchor, positive, negative]))
                    
        print(f"Loaded {len(examples)} TAO training triplets.")
    except FileNotFoundError:
        print(f"ERROR: Could not find {file_path}.")
        exit(1)
        
    return examples

def load_tao_dev_data(labels_path):
    """
    Loads the evaluation labels (also TAO extracted) using pure JSON to avoid Pandas bugs.
    """
    dev_examples = []
    print(f"Loading TAO Dev data from {labels_path}...")
    
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                    
                row = json.loads(line)
                
                # Extract text (assuming your extractor script kept the original keys)
                anchor = row.get("anchor_text", row.get("anchor")) 
                a_is_closer = row.get("text_a_is_closer")
                
                positive = row["text_a"] if a_is_closer else row["text_b"]
                negative = row["text_b"] if a_is_closer else row["text_a"]
                
                dev_examples.append(InputExample(texts=[anchor, positive, negative]))
                
        print(f"Loaded {len(dev_examples)} TAO evaluation triplets.")
        
    except FileNotFoundError:
        print(f"\nERROR: Could not find the dev file at '{labels_path}'.")
        print("Did you remember to run the dev labels through the TAO extractor?")
        exit(1)
    except Exception as e:
        print(f"\nERROR parsing dev data: {e}")
        exit(1)
        
    return dev_examples

def main():
    print(f"Initializing Base Model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # 1. Load Data
    train_examples = load_tao_training_data(TRAIN_DATA_PATH)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

    # 2. Setup MNRL
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # 3. Setup Evaluator
    dev_examples = load_tao_dev_data(DEV_LABELS_PATH)
    evaluator = TripletEvaluator.from_input_examples(dev_examples, name='tao-track-b-eval')

    print("\nStarting Neurosymbolic MNRL fine-tuning...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=NUM_EPOCHS,
        evaluation_steps=50, # Evaluate more frequently since the dataset is smaller
        warmup_steps=50,
        output_path="models/bge-tao-tuned",
        optimizer_params={'lr': 1e-5} # Slightly higher learning rate for a smaller dataset
    )
    
    print("\nTraining complete! Custom TAO model saved to 'models/bge-tao-tuned'")

if __name__ == "__main__":
    main()