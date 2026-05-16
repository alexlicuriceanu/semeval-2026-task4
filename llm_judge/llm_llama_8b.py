import json
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from tqdm import tqdm

INPUT_FILE = "../dataset/test_track_b_labels.jsonl"
OUTPUT_FILE = "./llama_8b_predictions.jsonl"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
NUM_GPUS = 1

class NarrativeDecision(BaseModel):
    reasoning_anchor_vs_a: str
    reasoning_anchor_vs_b: str
    final_conclusion: str
    text_a_is_closer: bool

JUDGE_PROMPT = """
You are an expert narratologist grading the SemEval Narrative Similarity task.
Your job is to determine whether TEXT A or TEXT B is structurally closer to the ANCHOR.

SEMEVAL ANNOTATION GUIDELINES:
1. Ignore superficial genres, character names, and settings.
2. 'Course of Action' (the mechanical steps the characters take) is heavily weighted.
3. 'Abstract Theme' (the philosophical core) is heavily weighted.
4. 'Outcome' differences do NOT automatically disqualify a text if the Course of Action matches.

Think step-by-step. Map the narrative mechanics of Anchor vs A, then Anchor vs B. Then declare the winner.

CRITICAL: You MUST output ONLY a valid JSON object. Do not include any markdown formatting, backticks, greetings, or preamble. 
EXAMPLE PERFECT OUTPUT:
{"reasoning_anchor_vs_a": "Text A follows a different mechanic...", "reasoning_anchor_vs_b": "Text B perfectly aligns...", "final_conclusion": "Text B is closer because...", "text_a_is_closer": false}
"""

FEW_SHOT_MESSAGES = [
    # --- EXAMPLE 1: ENEMY ROMANCE ---
    {"role": "user", "content": "ANCHOR:\nThe film takes place in Burma and India during World War II. A British officer falls in love with his Japanese instructor... she is regarded as the enemy... He escapes and returns to his own lines, only to discover that his wife is suffering from a brain tumour. Although the doctor initially gives her good odds of surviving, she dies after an operation.\n\nTEXT A:\nDuring the Irish War of Independence in 1921, Irish rebel leader Dennis Riordan and English aristocrat Helen Drummond meet and fall in love. Riordan is pursued, however, by British army officer Captain Preston... A happier ending was also filmed which has Riordan being shot but surviving.\n\nTEXT B:\n1914, German advance through Belgium: the young war volunteer Alexander Haller is given water by an equally young Belgian woman... They fall in love. Late 1918, German retreat after the Armistice: Engele and Alex meet again only to be harassed by a Belgian mob. Shortly prior to be hanged by the mob, a group of passing Belgian soldiers, tired about killing, saves them."},
    {"role": "assistant", "content": '{"reasoning_anchor_vs_a": "Text A involves a romance between opposing sides during a rebellion, but focuses more on a love triangle and pursuit by an officer.", "reasoning_anchor_vs_b": "Text B, like the Anchor, focuses on a romance between an invading soldier and a local woman from the opposing side during a World War, leading to severe backlash from their own people (the mob).", "final_conclusion": "Text B shares the exact abstract theme of a taboo romance between enemy combatants in a World War setting facing societal rejection, making it structurally closer than Text A.", "text_a_is_closer": false}'},

    # --- EXAMPLE 2: SCI-FI / TIME TRAVEL ---
    {"role": "user", "content": "ANCHOR:\nRuby Weaver is weary of her long history of failed relationships... Sam explains that he is really from the year 2470 and is what he calls a 'back traveler'... She takes him to see her therapist Meg Ford. In the end, both Deed and Ford turn out to be time travelers and the fatal accident that would have killed Ruby is avoided.\n\nTEXT A:\nIn the 1860s a widow runs an estate... One day a former army officer arrives... She persuades him to stay and work for her, and before long they begin a relationship. However, complications ensue when her daughter returns home from boarding school and develops an attachment to her mother's lover.\n\nTEXT B:\nIn 1943, the US Navy ship USS Eldridge disappears, due to the Philadelphia Experiment. In the present day (2012), researchers try to recreate the experiment, which has the unintended consequence of making the Eldridge reappear, apparently having traveled through time... The research company attempts to cover up the incident by trying to kill everyone involved. A struggle ensues as others try to protect the survivors."},
    {"role": "assistant", "content": '{"reasoning_anchor_vs_a": "Text A is a 19th-century romantic drama revolving around a love triangle between a mother, a daughter, and a former officer. It shares zero structural mechanics with the Anchor.", "reasoning_anchor_vs_b": "Text B shares the core sci-fi mechanic of time travel, people appearing from a different era, and a struggle to protect someone from a fatal outcome.", "final_conclusion": "Text B shares the fundamental sci-fi abstract theme of time travel and altering/protecting destinies, whereas Text A is a mundane period romance.", "text_a_is_closer": false}'},

    # --- EXAMPLE 3: ROGUE COP REVENGE ---
    {"role": "user", "content": "ANCHOR:\nWhen Los Angeles Police Department officers kill a young bank robber after a brutal bank heist, the slain criminal's father, mob boss Vincent Sloan, unleashes a blood bath on the police force. One by one, Detective Ray Morgan's partners are gunned down, but when his family is murdered, Morgan has nothing left to live for - except revenge. Becoming an unstoppable one man army, Morgan goes on the ultimate hunt for justice.\n\nTEXT A:\nWith his two best friends, Jonas Maldonado, the son of a slain NYPD officer, joins the police academy... Under Sarcone's tutelage, he becomes part of a rogue task force that consists of dirty cops. After he learns... his father was murdered by Sarcone... Maldonado frames Sarcone for a theft... Maldonado then kills the dirty cops responsible for his father's death.\n\nTEXT B:\nAn elaborate bank robbery takes place and the gangsters succeed although the FBI had been warned. The bank president dies of a heart attack. Jerry Cotton, who is considered accountable for this major failure, loses his badge over this. Being the man he is, Cotton doesn't let the evil-doers forget that he has unfinished business with them. He catches even the last one although he must jump onto a flying helicopter in order to get him."},
    {"role": "assistant", "content": '{"reasoning_anchor_vs_a": "Text A is an undercover police corruption thriller where the protagonist joins a dirty squad to avenge his father from the inside. It relies on deception and internal politics.", "reasoning_anchor_vs_b": "Text B follows an officer who loses his badge (official status) and goes rogue as a lone wolf to hunt down the criminals responsible, matching the Anchor\'s \'unstoppable one man army\' course of action.", "final_conclusion": "Text B is closer because it mirrors the Anchor\'s trajectory of an officer stepping outside the boundaries of the law to single-handedly hunt down criminals, avoiding the undercover/corruption angle of Text A.", "text_a_is_closer": false}'}
]

def build_messages(item):
    user_content = (
        f"ANCHOR:\n{item['anchor_text']}\n\n"
        f"TEXT A:\n{item['text_a']}\n\n"
        f"TEXT B:\n{item['text_b']}"
    )
    
    messages = [{"role": "system", "content": JUDGE_PROMPT}]
    messages.extend(FEW_SHOT_MESSAGES) 
    messages.append({"role": "user", "content": user_content})
    return messages

def extract_json(text):
    start = text.find('{')
    end = text.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        json_str = text[start:end+1]
        try:
            # Catch common LLM formatting errors (like unescaped newlines)
            json_str = json_str.replace('\n', ' ').replace('\r', '')
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
            
    return None

def main():
    print(f"Loading {MODEL_NAME} onto {NUM_GPUS} GPU(s)...")
    llm = LLM(model=MODEL_NAME, tensor_parallel_size=NUM_GPUS)
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=600) # Bumped max_tokens slightly to accommodate full CoT

    print(f"Reading data from {INPUT_FILE}")
    with open(INPUT_FILE, "r") as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]

    print("Building prompts")
    all_messages = [build_messages(item) for item in data]

    print("Running vLLM Batch Inference")
    outputs = llm.chat(messages=all_messages, sampling_params=sampling_params, use_tqdm=True)

    correct = 0
    results = []
    failed_parses = 0

    for item, output in zip(data, outputs):
        generated_text = output.outputs[0].text.strip()
        
        decision = extract_json(generated_text)
        
        if decision:
            predicted_a_closer = decision.get("text_a_is_closer")
            
            # Add basic type checking to ensure the bool was extracted
            if isinstance(predicted_a_closer, bool):
                if predicted_a_closer == item["text_a_is_closer"]:
                    correct += 1
                    
                results.append({
                    "anchor_title": item.get("story_details_anchor", {}).get("title", "Unknown"),
                    "ground_truth_a_closer": item["text_a_is_closer"],
                    "predicted_a_closer": predicted_a_closer,
                    "reasoning": decision.get("final_conclusion", "")
                })
            else:
                failed_parses += 1
                print(f"Failed to extract boolean for {item.get('story_details_anchor', {}).get('title')}")
        else:
            failed_parses += 1
            print(f"Failed to find any JSON block for {item.get('story_details_anchor', {}).get('title')}")

    valid_total = len(data) - failed_parses
    accuracy = correct / valid_total if valid_total > 0 else 0
    print(f"\n{MODEL_NAME} Accuracy: {accuracy:.4f} ({correct}/{valid_total})")
    
    if failed_parses > 0:
        print(f"Warning: {failed_parses} items were dropped due to malformed output.")

    with open(OUTPUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()