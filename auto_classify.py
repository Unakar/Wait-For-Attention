import json
from vllm import LLM, SamplingParams
from typing import Dict, List, Set
import torch
from tqdm import tqdm

# Configuration
MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
INPUT_JSONL = "response.jsonl"
OUTPUT_JSONL = "keywords.jsonl"

# Keyword patterns based on file analysis
KEYWORD_PATTERNS: Dict[str, Set[str]] = {
    "hesitation": {
        "hmm", 
        "wait",
        "okay, so",
        "alright",
        "let me think",
    },
    "lookback": {
        "i remember that",
        "i recall",
        "probably",
    },
    "self_correction": {
        "no, actually",
        "wait, no",
        "but wait",
    },
    "process": {
        "let me try to break this down",
        "so",
        "now",
    }
}

def generate_prompt(text: str) -> str:
    """Generate optimized English prompt"""
    # Convert KEYWORD_PATTERNS to a natural language description
    keyword_description = []
    for category, words in KEYWORD_PATTERNS.items():
        description = f"words about {category} can be: {', '.join(sorted(words))}"
        keyword_description.append(description)
    
    # Combine the descriptions into a single string with newlines
    keyword_description_str = "\n".join(keyword_description)
    
    system_msg = (
        "Please analyze the text below and extract specific expressions or interjections that indicate a contemplative or reflective thought process. "
        "These phrases often include words or phrases like 'hmm', 'let me think', or 'wait, let me check that', which signal hesitation, reflection, deliberation, error-correction, or recalling past memories.\n\n"
        "The following categories and examples provide a general guide for the types of expressions to look for. "
        f"{keyword_description_str}\n\n"
        "While the classifications are approximate, they can help you identify relevant patterns. "
        "Feel free to use your own judgment to determine any appropriate words or phases.\n\n"
        "Output format: In the end of your answer, list what you found one by one, separate them by commas, and put them in a list []. For example: ['let me think', 'wait', 'let me check that']"
    )
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Analyze this text: {text}"}
    ]
    
    # Simplified prompt template for VLLM
    return f"{system_msg}\n\nUser: Analyze this text: {text}\n\nAssistant:"

def extract_keywords_list(text: str) -> List[str]:
    """Extract the list of keywords from the model output"""
    try:
        # Find the content within the last square brackets
        start = text.rindex('[')
        end = text.rindex(']') + 1
        keywords_str = text[start:end]
        # Parse the string into a list
        keywords_list = json.loads(keywords_str.replace("'", '"'))
        return keywords_list
    except (ValueError, json.JSONDecodeError):
        print(f"Warning: Failed to extract keywords list from: {text[:100]}...")
        return []

def process_batch(llm: LLM, entries: List[dict], batch_size: int = 32) -> List[dict]:
    """Process a batch of entries using VLLM"""
    prompts = [generate_prompt(entry["response"]) for entry in entries]
    
    sampling_params = SamplingParams(
        max_tokens=10000,
    )
    
    outputs = llm.generate(prompts, sampling_params)
    results = []
    
    for entry, output in zip(entries, outputs):
        full_response = output.outputs[0].text.strip()
        # Extract the list of keywords
        keywords_list = extract_keywords_list(full_response)
        
        results.append({
            "doc_id": entry.get("doc_id"),
            "keywords": keywords_list,
            "check": full_response  # Save the full response for verification
        })
    
    return results

def main():
    # Initialize VLLM with optimized batch processing
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=4,
        dtype="bfloat16",
    )
    
    # Read data with error handling and debug information
    entries = []
    try:
        with open(INPUT_JSONL, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Line {line_num} failed to parse: {str(e)}")
                    continue
        print(f"Successfully loaded {len(entries)} input entries")
    except Exception as e:
        print(f"Failed to read file: {str(e)}")
        return
    
    # Process and write output with UTF-8 encoding
    try:
        with open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
            for i in tqdm(range(0, len(entries), 32)):
                batch = entries[i:i+32]
                batch_results = process_batch(llm, batch)
                for result in batch_results:
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Error occurred during processing: {str(e)}")

if __name__ == "__main__":
    main()
