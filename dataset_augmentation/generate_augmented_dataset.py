import openai
import json
import time
from dotenv import load_dotenv
import os
import random
from tqdm import tqdm
from pydantic import BaseModel
from typing import List

load_dotenv()
client = openai.OpenAI()

PROMPT_PATH = "./augmented_dataset_prompt.txt"
TRACK_B_OUT = "augmented_track_b_train.jsonl"
TRACK_A_OUT = "augmented_track_a_train.jsonl"

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

class StoryTriplet(BaseModel):
    anchor: str
    similar_story: str
    dissimilar_story: str

class TripletBatch(BaseModel):
    triplets: List[StoryTriplet]

def generate_triplets(num_batches=1):
    try:
        with open(PROMPT_PATH, "r") as f:
            base_prompt = f.read()
    except FileNotFoundError:
        print(f"Error: {PROMPT_PATH} not found!")
        return

    # Clear existing files before starting so we don't append to old tests
    open(TRACK_B_OUT, "w").close()
    open(TRACK_A_OUT, "w").close()

    for i in tqdm(range(num_batches), desc="Generating Triplet Batches", unit="batch"):
        
        batch_constraints = []
            
        for j in range(5):
            m = random.choice(MEDIA_TYPES)
            g = random.choice(GENRES)
            c = random.choice(CONFLICTS)
            s = random.choice(STRUCTURES)
            h = random.choice(OPENING_HOOKS)
            l = random.choice(LENGTH_JITTER)
                
            # Replace placeholders in the hook
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

        try:
            completion = client.beta.chat.completions.parse(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": "You are a data generator for narrative similarity."},
                    {"role": "user", "content": final_prompt}
                ],
                response_format=TripletBatch, 
                max_completion_tokens=3500,
                temperature=0.95
            )
            
            data = completion.choices[0].message.parsed
            
            # Open files in append mode so we save as we go
            with open(TRACK_B_OUT, "a") as fb, open(TRACK_A_OUT, "a") as fa:
                for triplet in data.triplets:
                    
                    # Save Track B Format
                    track_b_item = {
                        "anchor_story": triplet.anchor,
                        "similar_story": triplet.similar_story,
                        "dissimilar_story": triplet.dissimilar_story
                    }
                    fb.write(json.dumps(track_b_item) + "\n")
                    
                    # Save Track A Format
                    is_a_closer = random.choice([True, False])
                    track_a_item = {
                        "anchor": triplet.anchor,
                        "text_a": triplet.similar_story if is_a_closer else triplet.dissimilar_story,
                        "text_b": triplet.dissimilar_story if is_a_closer else triplet.similar_story,
                        "text_a_is_closer": is_a_closer
                    }
                    fa.write(json.dumps(track_a_item) + "\n")
                    
        except Exception as e:
            print(f"\nAPI Error in Batch {i+1}: {e}")
            time.sleep(5)

if __name__ == "__main__":
    generate_triplets(num_batches=1)