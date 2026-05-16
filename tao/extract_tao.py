import json
import openai
from pydantic import BaseModel
import concurrent.futures
from tqdm import tqdm
from dotenv import load_dotenv
import threading

load_dotenv()
client = openai.OpenAI()

INPUT_FILE = "../dataset/synthetic_data_for_contrastive_learning.jsonl"
OUTPUT_FILE = "./tao_synthetic_data_train.jsonl"

file_write_lock = threading.Lock()

class TAO(BaseModel):
    theme: str
    action: str
    outcome: str

class TripletTAO(BaseModel):
    anchor_tao: TAO
    similar_tao: TAO
    dissimilar_tao: TAO

EXTRACTION_PROMPT = """
You are a Narrative Deconstruction engine. Your task is to read three story summaries and extract their underlying structures into strict Symbolic Triples: Theme, Action, Outcome (TAO).

CRITICAL RULES:
1. NO PROPER NOUNS. You MUST strip out all character names, locations, specific items, and time periods. Use generic terms (e.g., "The protagonist," "An artifact," "An isolated location").
2. Be highly abstract but mechanically precise about the plot.
3. The 'similar' story should have a nearly identical TAO to the anchor. The 'dissimilar' story MUST have a fundamentally different Outcome or Action.
"""

def format_tao_string(tao: TAO) -> str:
    """Converts the JSON object into a flat, highly semantic string for BGE-Large"""
    return f"Theme: {tao.theme} Action: {tao.action} Outcome: {tao.outcome}"

def process_single_triplet(line_str):
    try:
        item = json.loads(line_str)
        anchor_text = item.get("anchor_story")
        similar_text = item.get("similar_story")
        dissimilar_text = item.get("dissimilar_story")
        
        user_content = (
            f"Extract TAOs for the following:\n\n"
            f"ANCHOR:\n{anchor_text}\n\n"
            f"SIMILAR:\n{similar_text}\n\n"
            f"DISSIMILAR:\n{dissimilar_text}"
        )
        
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": user_content}
            ],
            response_format=TripletTAO,
            temperature=0.1 # Low temperature for objective extraction
        )
        
        parsed_data = completion.choices[0].message.parsed
        
        # Format back into standard text
        flat_triplet = {
            "anchor_story": format_tao_string(parsed_data.anchor_tao),
            "similar_story": format_tao_string(parsed_data.similar_tao),
            "dissimilar_story": format_tao_string(parsed_data.dissimilar_tao)
        }
        
        with file_write_lock:
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(flat_triplet) + "\n")
                
    except Exception as e:
        print(f"\nExtraction Error: {e}")

def run_extraction():
    print(f"Reading dataset from {INPUT_FILE}")
    with open(INPUT_FILE, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
    print(f"Loaded {len(lines)} triplets")
    
    # Clear output file
    open(OUTPUT_FILE, "w").close()
    
    max_workers = 4
    print(f"Starting TAO extraction with {max_workers} threads")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_triplet, line) for line in lines]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(lines), desc="Extracting TAOs"):
            future.result()
            
    print(f"\nTAO dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_extraction()