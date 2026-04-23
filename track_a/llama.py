import pandas as pd
from transformers import pipeline
import torch
from tqdm import tqdm

# --- Configuration ---
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEV_DATA_PATH = "../dataset/test_track_a.jsonl" 
DEV_LABELS_PATH = "../dataset/test_track_a_labels.jsonl" 

print(f"Loading {MODEL_ID} onto H100...")
# device_map="auto" will automatically detect your 80GB VRAM and load the model there
pipe = pipeline(
    "text-generation",
    model=MODEL_ID,
    model_kwargs={"torch_dtype": torch.bfloat16}, 
    device_map="auto",
)
print("Model loaded successfully!")

def predict_llm(row):
    # The prompt explicitly asks the LLM to act as a judge and limits its output
    prompt = f"""You are an expert literary judge for a SemEval competition.
You must determine which candidate story shares a closer narrative similarity to the Anchor story. 
Focus strictly on the abstract theme, the course of action, and the final outcome. Ignore character names, locations, and simple vocabulary overlap.

Anchor story: {row['anchor_text']}

Story A: {row['text_a']}

Story B: {row['text_b']}

Which story (A or B) has a more similar narrative arc to the Anchor? 
Respond strictly with a single letter: A or B."""

    # Format for Llama 3.1 Instruct
    messages = [
        {"role": "system", "content": "You are a helpful, precise AI assistant."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        # Generate the answer. Temperature 0.0 makes it deterministic (no random guessing).
        output = pipe(
            messages, 
            max_new_tokens=5, 
            temperature=0.0, 
            do_sample=False,
            pad_token_id=pipe.tokenizer.eos_token_id
        )
        
        # Parse the output
        response_text = output[0]["generated_text"][-1]["content"].strip().upper()
        
        # Determine if the LLM picked A or B
        if "A" in response_text and "B" not in response_text:
            return True
        elif "B" in response_text and "A" not in response_text:
            return False
        else:
            # Fallback if the LLM refuses to answer properly
            return True if response_text.startswith("A") else False
            
    except Exception as e:
        print(f"Error during generation: {e}")
        return False # Default fallback

def main():
    print("\n--- Evaluating Human Dev Set with Zero-Shot Llama 3.1 ---")
    try:
        df_dev = pd.read_json(DEV_DATA_PATH, lines=True)
        df_labels = pd.read_json(DEV_LABELS_PATH, lines=True)
        
        # Use tqdm's progress_apply to get a loading bar
        tqdm.pandas(desc="Evaluating Narratives...")
        df_dev["predicted_text_a_is_closer"] = df_dev.progress_apply(predict_llm, axis=1)

        accuracy = (df_dev["predicted_text_a_is_closer"] == df_labels["text_a_is_closer"]).mean()
        print(f"\n========================================")
        print(f"Llama 3.1 Zero-Shot Accuracy: {accuracy * 100:.2f}%")
        print(f"========================================")
        
    except FileNotFoundError:
        print("Dataset files not found. Please check paths.")

if __name__ == "__main__":
    main()