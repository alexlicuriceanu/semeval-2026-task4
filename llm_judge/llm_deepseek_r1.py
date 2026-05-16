import json
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILE = "../dataset/test_track_b_labels.jsonl"
OUTPUT_FILE = "./deepseek_r1_predictions.jsonl"

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" 
NUM_GPUS = 2 

JUDGE_PROMPT = """
You are an expert narratologist and literary structuralist grading the SemEval Narrative Similarity task.
Your job is to determine whether TEXT A or TEXT B is structurally closer to the ANCHOR.

SEMEVAL ANNOTATION GUIDELINES (READ CAREFULLY):
1. THE VIBE TRAP: Ignore superficial genres, character names, and settings. A sci-fi space opera can be structurally identical to a 19th-century western. DO NOT match stories just because they both feature "prison," "jungles," or "police."
2. COURSE OF ACTION: The mechanical sequence of steps the characters take is the most important factor.
3. ABSTRACT THEME: The philosophical core (e.g., 'revenge corrupts', 'sacrificing for love') is heavily weighted.
4. OUTCOME: Differences in the ending (e.g., one hero dies, the other lives) do NOT disqualify a text if the Course of Action matches.

INSTRUCTIONS:
Carefully analyze the structural mechanics of the Anchor. Compare it to Text A, then Text B. 
You MUST output your final answer at the very end of your response as a valid JSON object EXACTLY matching this schema:
{"reasoning_summary": "string", "text_a_is_closer": boolean}
"""

def build_messages(item):
    user_content = (
        f"{JUDGE_PROMPT}\n\n"
        f"ANCHOR:\n{item['anchor_text']}\n\n"
        f"TEXT A:\n{item['text_a']}\n\n"
        f"TEXT B:\n{item['text_b']}"
    )
    
    return [{"role": "user", "content": user_content}]

def extract_r1_json(text):
    # FIX THE vLLM TOKENIZER BUG: Replace byte-level tokens with actual spaces/newlines
    text = text.replace('Ġ', ' ').replace('Ċ', '\n').replace('```json', '').replace('```', '')

    # Find the identifying key
    marker = '"reasoning_summary"'
    idx = text.rfind(marker)
    
    if idx == -1:
        marker = "'reasoning_summary'"
        idx = text.rfind(marker)
        if idx == -1: return None

    # Backtrack to find the opening brace
    start_idx = text.rfind('{', 0, idx)
    if start_idx == -1: return None

    # Forward track to find the matching closing brace
    open_braces = 0
    end_idx = -1
    for i in range(start_idx, len(text)):
        if text[i] == '{':
            open_braces += 1
        elif text[i] == '}':
            open_braces -= 1
            if open_braces == 0:
                end_idx = i
                break

    if end_idx == -1: return None 

    json_str = text[start_idx:end_idx+1]

    try:
        # Clean up and fix Python/JSON boolean hallucinations
        json_str = json_str.replace('\n', ' ').replace('\r', '')
        json_str = re.sub(r':\s*True\b', ': true', json_str)
        json_str = re.sub(r':\s*False\b', ': false', json_str)
        
        return json.loads(json_str)
    except Exception as e:
        # If it still fails, print the scrubbed string to see why
        print(f"\n[DEBUG JSON PARSE ERROR]: {json_str}")
        return None

def main():
    print(f"Loading {MODEL_NAME} onto {NUM_GPUS} GPU(s)...")
    llm = LLM(
        model=MODEL_NAME, 
        tensor_parallel_size=NUM_GPUS,
        max_model_len=8192, 
        gpu_memory_utilization=0.95 
    )
    
    # max_tokens set to 6000 to ensure it fits perfectly inside max_model_len 
    # along with the prompt, preventing silent cutoffs
    sampling_params = SamplingParams(temperature=0.6, max_tokens=6000) 

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
        decision = extract_r1_json(generated_text)
        
        if decision and 'text_a_is_closer' in decision:
            # Coerce the boolean safely just in case it's a string like "true"
            val = decision["text_a_is_closer"]
            predicted_a_closer = val if isinstance(val, bool) else str(val).lower() == "true"
            
            if predicted_a_closer == item["text_a_is_closer"]:
                correct += 1
                
            results.append({
                "anchor_title": item.get("story_details_anchor", {}).get("title", "Unknown"),
                "ground_truth_a_closer": item["text_a_is_closer"],
                "predicted_a_closer": predicted_a_closer,
                "reasoning_summary": decision.get("reasoning_summary", "")
            })
        else:
            failed_parses += 1
            # Print the very end of the generation to see why it failed
            print(f"Parse Failure on: {item.get('story_details_anchor', {}).get('title')}")
            print(f"--- TRUNCATED OUTPUT ENDING ---\n{generated_text[-200:]}\n-------------------------------")

    valid_total = len(data) - failed_parses
    accuracy = correct / valid_total if valid_total > 0 else 0
    print(f"\n{MODEL_NAME} Accuracy: {accuracy:.4f} ({correct}/{valid_total})")
    
    if failed_parses > 0:
        print(f"Warning: {failed_parses} items were dropped.")

    with open(OUTPUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()