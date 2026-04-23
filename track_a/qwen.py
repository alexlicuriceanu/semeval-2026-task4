import pandas as pd
from transformers import pipeline
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
TEST_DATA_PATH = "../dataset/test_track_a.jsonl" 
TEST_LABELS_PATH = "../dataset/test_track_a_labels.jsonl"

print(f"Loading {MODEL_ID} onto H100...")
pipe = pipeline(
    "text-generation",
    model=MODEL_ID,
    model_kwargs={"torch_dtype": torch.bfloat16}, 
    device_map="auto",
)

import re

def predict_test_row(row):
    # --- FIX 1: Chain-of-Thought Prompt ---
    # We remove the toy example and force the model to "think step-by-step"
    prompt = f"""You are an expert literary judge for a SemEval competition.
Read the Anchor story, Story A, and Story B. 
Focus strictly on the underlying abstract theme, the course of action, and the final outcome. Ignore character names and locations.

Anchor story: {row['anchor_text']}
Story A: {row['text_a']}
Story B: {row['text_b']}

Step 1: Write a brief 1-sentence analysis comparing how the 'outcome' or 'resolution' of Story A and Story B matches the Anchor.
Step 2: On a new line, output your final verdict in exactly this format: 'Decision: A' or 'Decision: B'."""

    messages = [
        {"role": "system", "content": "You are a highly analytical AI trained in narrative structure and comparative literature."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        # --- FIX 2: Give the model room to think ---
        output = pipe(
            messages, 
            max_new_tokens=150, # Increased from 5 so it can write its analysis
            temperature=0.1,    # Slight bump to prevent repetitive looping, but still deterministic
            do_sample=True,
            top_p=0.9
        )
        
        response_text = output[0]["generated_text"][-1]["content"].strip()
        
        # --- FIX 3: Robust Regex Parsing ---
        # Hunt specifically for the word "Decision:" followed by A or B
        match = re.search(r"Decision:\s*([AB])", response_text, re.IGNORECASE)
        
        if match:
            choice = match.group(1).upper()
            return True if choice == "A" else False
        else:
            # Fallback if the model forgets the formatting
            return True if "Story A is more" in response_text else False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print(f"\n--- Generating Predictions and Evaluating Test Set ---")
    try:
        # Load data and labels
        df_test = pd.read_json(TEST_DATA_PATH, lines=True)
        df_labels = pd.read_json(TEST_LABELS_PATH, lines=True)
        
        # Run inference with a progress bar
        tqdm.pandas(desc="Inferencing on Test Data...")
        df_test["predicted_text_a_is_closer"] = df_test.progress_apply(predict_test_row, axis=1)

        # Extract ground truth and predictions
        y_true = df_labels["text_a_is_closer"]
        y_pred = df_test["predicted_text_a_is_closer"]
        
        # Calculate final metrics
        acc = accuracy_score(y_true, y_pred)
        
        print("\n========================================")
        print(f"FINAL TEST ACCURACY (Qwen 2.5 7B): {acc * 100:.2f}%")
        print("========================================\n")
        print("Detailed Classification Report:")
        # We map False -> "Text B Closer" and True -> "Text A Closer" for readable output
        print(classification_report(y_true, y_pred, target_names=["Text B Closer (False)", "Text A Closer (True)"]))
        
    except FileNotFoundError as e:
        print(f"Error loading files: {e}. Please verify your paths.")

if __name__ == "__main__":
    main()