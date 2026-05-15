import openai
import json
import random
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
client = openai.OpenAI()

PROMPT_PATH = "./augmented_dataset_prompt.txt"
BATCH_INPUT_FILE = "batch_requests.jsonl"
BATCH_ID_FILE = "./augmented_dataset_batch_id.txt"

MEDIA_TYPES = [
    "feature film", "novel", "short story", "television episode", 
    "stage play", "traditional folk tale", "graphic novel", "memoir"
]

GENRES = [
    "Cyberpunk Action", "Mythological Fantasy Epic", "Gritty 1970s Martial Arts", 
    "Cold War Espionage", "Post-Apocalyptic Zombie Survival", "Classic Spaghetti Western", 
    "Modern Teen Slasher", "1990s Undercover Cop Thriller", "Historical Romance Drama",
    "B-Movie Creature Feature", "1950s Noir Detective", "Space Opera Adventure",
    "Slice-of-life Contemporary Drama", "Satirical Dark Comedy", "Gothic Horror",
    "Legal/Courtroom Drama", "Nautical High Seas Adventure"
]

CONFLICTS = [
    "A protagonist discovers a hidden conspiracy and must flee.",
    "Bitter rivals are forced to work together to survive.",
    "A heist or robbery that goes wrong due to a sudden double-cross.",
    "An innocent person is framed and must clear their name.",
    "A character tries to escape a secret organization they once worked for.",
    "A romantic misunderstanding spirals into a dangerous situation.",
    "A community tries to survive a slow-moving natural disaster.",
    "Two star-crossed lovers navigate a bitter generational family feud.",
    "A young prodigy discovers the dark truth behind their mentor's success.",
    "A seasoned professional comes out of retirement for one last impossible job.",
    "Strangers trapped in an isolated location are picked off one by one."
]

OPENING_HOOKS = [
    "The [MEDIA] follows", "The [MEDIA] presents", "The [MEDIA] opens with", 
    "Set in [PLACE], the narrative centers on", "Following the events of", 
    "The story revolves around", "The protagonist, a [JOB],", 
    "In this [GENRE] [MEDIA],", "A [JOB] named [NAME] is", 
    "Published in [YEAR], this [MEDIA]", "Loosely based on real events, the [MEDIA]",
    "The central plot focuses on"
]

STRUCTURES = [
    "Chronological scene-by-scene summary.",
    "Non-linear summary. Start with the climax or a later event.",
    "Academic tone. Describe the plot as a 'portrait', 'exploration', or 'reconstruction'.",
    "Actor-heavy synopsis (if a film/TV). Include at least 3 fake actor names in parentheses.",
    "Thematic overview. Focus heavily on character motivations rather than pure action."
]

LENGTH_JITTER = [
    "Extremely brief and punchy (Strictly 30-40 words).",
    "Short and concise (Strictly 50-70 words).",
    "Standard length (Strictly 80-100 words).",
    "Detailed and verbose (Strictly 120-150 words).",
    "Overly descriptive and dense (Strictly 150-180 words)."
]

def create_and_launch_batch(start_idx, end_idx):
    try:
        with open(PROMPT_PATH, "r") as f:
            base_prompt = f.read()
    except FileNotFoundError:
        print(f"Error: {PROMPT_PATH} not found!")
        return

    print("Generating batch request file locally")
    with open(BATCH_INPUT_FILE, "w") as f:
        for i in tqdm(range(start_idx, end_idx), desc="Building Requests", unit="batch"):
            batch_constraints = []
            
            for j in range(5):
                m = random.choice(MEDIA_TYPES)
                g = random.choice(GENRES)
                c = random.choice(CONFLICTS)
                s = random.choice(STRUCTURES)
                h = random.choice(OPENING_HOOKS)
                l = random.choice(LENGTH_JITTER)
                
                hook = h.replace("[MEDIA]", m).replace("[GENRE]", g.lower())
                
                batch_constraints.append(
                    f"Triplet {j+1}:\n"
                    f"  - Media Format: {m}\n"
                    f"  - Genre: {g}\n"
                    f"  - Arc: {c}\n"
                    f"  - Target Length: {l}\n"
                    f"  - Style/Structure: {s}\n"
                    f"  - MANDATORY OPENING: Start the 'anchor' story exactly with the phrase '{hook}'."
                )
                
            constraints_str = "\n\n".join(batch_constraints)
            
            dynamic_injection = (
                f"\n\nCRITICAL DIVERSITY CONSTRAINT FOR THIS BATCH:\n"
                f"You MUST write 5 completely unrelated triplets using the exact configurations below:\n"
                f"{constraints_str}\n\n"
                f"Make sure the 5 'anchor' stories sound completely different from each other in length, tone, media format, and sentence structure!\n"
                f"OUTPUT REQUIREMENT: Return a JSON object with a single root key called 'triplets' containing your array."
            )
            
            final_prompt = base_prompt + dynamic_injection

            request_line = {
                "custom_id": f"batch_req_{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-5.4-mini",
                    "messages": [
                        {"role": "system", "content": "You are a data generator for narrative similarity. Output strictly JSON."},
                        {"role": "user", "content": final_prompt}
                    ],
                    "response_format": { "type": "json_object" },
                    "max_completion_tokens": 3500,
                    "temperature": 0.95
                }
            }
            f.write(json.dumps(request_line) + "\n")

    print("Uploading file to OpenAI")
    batch_input_file = client.files.create(
      file=open(BATCH_INPUT_FILE, "rb"),
      purpose="batch"
    )

    print("Launching Batch Job")
    batch_job = client.batches.create(
      input_file_id=batch_input_file.id,
      endpoint="/v1/chat/completions",
      completion_window="24h",
      metadata={
        "description": "10k Narrative Similarity Triplets"
      }
    )

    print("\nBatch Job Launched")
    print("="*50)
    print(f"Batch ID: {batch_job.id}")
    print("="*50)

    with open(BATCH_ID_FILE, "a") as f:
        f.write(batch_job.id + "\n")
    print(f"Batch ID appended to {BATCH_ID_FILE}")

if __name__ == "__main__":
    create_and_launch_batch(1500, 2000)