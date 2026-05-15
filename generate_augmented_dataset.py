import openai
import json
import time
from dotenv import load_dotenv
import os

load_dotenv() # Loads your key from the .env file
client = openai.OpenAI() # Automatically looks for OPENAI_API_KEY env var

PROMPT_PATH = "./augmented_dataset_prompt.txt"

def generate_triplets(num_batches=10):
    try:
        with open(os.path.join(os.getcwd(), PROMPT_PATH), "r") as f:
            prompt = f.read()
    except FileNotFoundError:
        print(f"Error: {PROMPT_PATH} not found")
        return
    
    track_b_data = [] # For Contrastive Learning (Anchor, Pos, Neg)
    track_a_data = [] # For Classification (Anchor, A, B, Label)

    for i in range(num_batches):
        print(f"Generating batch {i+1}/{num_batches}")
        try:
            response = client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": "You are a data generator. Output ONLY valid JSON arrays of objects."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9
            )
            
            # Parse the JSON response
            raw_text = response.choices[0].message.content.strip()
            if raw_text.startswith("```json"):
                 raw_text = raw_text[7:-3] # strip markdown if it ignores instructions
                 
            triplets = json.loads(raw_text)
            
            # Format the data for your two tracks
            for triplet in triplets:
                # Track B Format (Contrastive)
                track_b_data.append({
                    "anchor_story": triplet["anchor"],
                    "similar_story": triplet["similar_story"],
                    "dissimilar_story": triplet["dissimilar_story"]
                })
                
                # Track A Format (Classification)
                # We randomly assign the similar story to text_a or text_b
                import random
                if random.choice([True, False]):
                    track_a_data.append({
                        "anchor": triplet["anchor"],
                        "text_a": triplet["similar_story"],
                        "text_b": triplet["dissimilar_story"],
                        "text_a_is_closer": True
                    })
                else:
                    track_a_data.append({
                        "anchor": triplet["anchor"],
                        "text_a": triplet["dissimilar_story"],
                        "text_b": triplet["similar_story"],
                        "text_a_is_closer": False
                    })
                    
        except Exception as e:
            print(f"API Error: {e}")
            time.sleep(2) # Backoff on error

    # Save to JSONL
    with open("augmented_track_b_train.jsonl", "w") as f:
        for item in track_b_data:
            f.write(json.dumps(item) + "\n")
            
    with open("augmented_track_a_train.jsonl", "w") as f:
        for item in track_a_data:
            f.write(json.dumps(item) + "\n")
            

if __name__ == "__main__":
    generate_triplets(num_batches=2)