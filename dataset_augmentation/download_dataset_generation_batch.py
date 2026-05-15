import openai
import json
import random
from dotenv import load_dotenv
import json
import re
import random

load_dotenv()
client = openai.OpenAI()

BATCH_ID_FILE = "./augmented_dataset_batch_id.txt"
TRACK_B_OUT = "merged_track_b_train.jsonl"
TRACK_A_OUT = "merged_track_a_train.jsonl"

def check_and_process_batch(batch_id):
    batch_job = client.batches.retrieve(batch_id)
    print(f"Current Status for {batch_id}: {batch_job.status}")

    if batch_job.status == "completed":
        print(f"Downloading results for {batch_id}")
        file_response = client.files.content(batch_job.output_file_id)
        raw_responses = file_response.text.strip().split("\n")
        
        print(f"Formatting {len(raw_responses)} batches into Track A and Track B")
        
        with open(TRACK_B_OUT, "a") as fb, open(TRACK_A_OUT, "a") as fa:
            for line in raw_responses:
                try:
                    response_json = json.loads(line)
                    generated_text = response_json["response"]["body"]["choices"][0]["message"]["content"]
                    data = json.loads(generated_text)
                    
                    triplets = data.get("triplets", [])
                    
                    for triplet in triplets:
                        track_b_item = {
                            "anchor_story": triplet["anchor"],
                            "similar_story": triplet["similar_story"],
                            "dissimilar_story": triplet["dissimilar_story"]
                        }
                        fb.write(json.dumps(track_b_item) + "\n")
                        
                        is_a_closer = random.choice([True, False])
                        track_a_item = {
                            "anchor": triplet["anchor"],
                            "text_a": triplet["similar_story"] if is_a_closer else triplet["dissimilar_story"],
                            "text_b": triplet["dissimilar_story"] if is_a_closer else triplet["similar_story"],
                            "text_a_is_closer": is_a_closer
                        }
                        fa.write(json.dumps(track_a_item) + "\n")
                        
                except Exception as e:
                    print(f"Failed to parse a line: {e}")
                    
        print(f"Saved to {TRACK_B_OUT} and {TRACK_A_OUT}.")
    
    elif batch_job.status in ["failed", "cancelled", "expired"]:
        print(f"The batch job {batch_id} failed or was cancelled")

if __name__ == "__main__":
    try:
        with open(BATCH_ID_FILE, "r") as f:
            batch_ids = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Error: {BATCH_ID_FILE} not found")
        exit(1)

    open(TRACK_B_OUT, "w").close()
    open(TRACK_A_OUT, "w").close()

    for b_id in batch_ids:
        print("=" * 50)
        check_and_process_batch(b_id)


    YEARS = [str(y) for y in range(1940, 2025)]
    PLACES = [
        "New York", "Los Angeles", "London", "Paris", "Berlin", "Tokyo", "Rome", 
        "a remote research station", "a small Mediterranean village", "Chicago", 
        "the ruins of a coastal city", "a high-security prison", "Vienna"
    ]
    JOBS = [
        "detective", "former soldier", "scientist", "undercover agent", "mechanic", 
        "student", "journalist", "hacker", "waitress", "architect", "pilot"
    ]
    NAMES = [
        "James", "Sarah", "Victor", "Elena", "Marcus", "Nadia", "Elias", "Chloe",
        "Robert", "Maya", "Julian", "Clara", "Anton", "Sonia", "Frank"
    ]

    def clean_text(text):
        # First replace the explicit placeholders [YEAR], [PLACE], etc.
        text = re.sub(r'\[YEAR\]', lambda _: random.choice(YEARS), text)
        text = re.sub(r'\[PLACE\]', lambda _: random.choice(PLACES), text)
        text = re.sub(r'\[JOB\]', lambda _: random.choice(JOBS), text)
        text = re.sub(r'\[NAME\]', lambda _: random.choice(NAMES), text)
        
        # This regex looks for text inside brackets and removes the brackets
        text = re.sub(r'\[(.*?)\]', r'\1', text)
        
        return text

    def clean_dataset(input_file, output_file):
        print(f"Sanitizing {input_file}...")
        count = 0
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                try:
                    item = json.loads(line)
                    cleaned_item = {k: clean_text(v) if isinstance(v, str) else v for k, v in item.items()}
                    outfile.write(json.dumps(cleaned_item) + '\n')
                    count += 1
                except Exception as e:
                    print(f"Skipping line due to error: {e}")

    clean_dataset("merged_track_b_train.jsonl", "augmented_track_b_train.jsonl")
    clean_dataset("merged_track_a_train.jsonl", "augmented_track_a_train.jsonl")
    print("Data cleaned")