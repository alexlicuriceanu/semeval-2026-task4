import json
import openai
from pydantic import BaseModel
import concurrent.futures
from tqdm import tqdm
from dotenv import load_dotenv
import threading

load_dotenv()
client = openai.OpenAI()

# INPUT: Your raw test set with labels
INPUT_FILE = "../dataset/test_track_b_labels.jsonl"
# OUTPUT: The TAO-ified test set
OUTPUT_FILE = "./tao_test_track_b_labels.jsonl"

file_write_lock = threading.Lock()

class TAO(BaseModel):
    theme: str
    action: str
    outcome: str

class TestTripletTAO(BaseModel):
    anchor_tao: TAO
    text_a_tao: TAO
    text_b_tao: TAO

EXTRACTION_PROMPT = """
You are a Narrative Deconstruction engine. Extract Symbolic Triples: Theme, Action, Outcome (TAO).
RULES: 
1. NO PROPER NOUNS (use "The protagonist", "The location", etc).
2. Be highly abstract but mechanically precise.
"""

def format_tao_string(tao: TAO) -> str:
    return f"Theme: {tao.theme} Action: {tao.action} Outcome: {tao.outcome}"

def process_test_line(line_str):
    try:
        item = json.loads(line_str)
        
        # Adjusting to your test set keys
        user_content = (
            f"Extract TAOs for:\n\n"
            f"ANCHOR:\n{item.get('anchor_text')}\n\n"
            f"TEXT_A:\n{item.get('text_a')}\n\n"
            f"TEXT_B:\n{item.get('text_b')}"
        )
        
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": user_content}
            ],
            response_format=TestTripletTAO,
            temperature=0.1
        )
        
        parsed = completion.choices[0].message.parsed
        
        # Keep the original labels and metadata, just swap the text for TAO strings
        tao_item = item.copy()
        tao_item["anchor_text"] = format_tao_string(parsed.anchor_tao)
        tao_item["text_a"] = format_tao_string(parsed.text_a_tao)
        tao_item["text_b"] = format_tao_string(parsed.text_b_tao)
        
        with file_write_lock:
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(tao_item) + "\n")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    with open(INPUT_FILE, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    
    open(OUTPUT_FILE, "w").close()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(process_test_line, lines), total=len(lines), desc="TAO-ifying Test Set"))