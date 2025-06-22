#!/usr/bin/env python3
"""
Process the full CodeForces dataset using the non-hacking prefix prompt.
This script reads prompts from codeforces.csv, prepends the non-hacking prefix,
sends them to Claude Sonnet 3.7, and saves completions to a JSONL file.
"""

import json
import pandas as pd
from safetytooling.apis.inference.anthropic import AnthropicAPI
from safetytooling.data_models.inference import Prompt
from config import PREFIX_PROMPT_NON_HACKING, CONFIG
import time
from tqdm import tqdm

def process_full_dataset_non_hacking():
    """Process the full dataset using non-hacking prefix prompt."""
    
    # Load the CSV file
    print(f"Loading prompts from {CONFIG['csv_file']}...")
    df = pd.read_csv(CONFIG['csv_file'])
    
    # Initialize the API
    api = AnthropicAPI()
    
    # Prepare output file with different name
    output_file = "../datasets/claude_completions_non_hacking.jsonl"
    print(f"Will save completions to {output_file}")
    
    # Process each prompt
    results = []
    total_prompts = len(df)
    
    print(f"Processing {total_prompts} prompts with non-hacking prefix...")
    
    for idx, row in tqdm(df.iterrows(), total=total_prompts, desc="Processing prompts"):
        prompt_text = row['prompt']
        
        # Prepend the non-hacking prefix
        full_prompt = PREFIX_PROMPT_NON_HACKING + "\n\n" + prompt_text
        
        # Create prompt object
        prompt = Prompt(
            prompt=full_prompt,
            model_id=CONFIG['model_id'],
            max_tokens=CONFIG['max_tokens'],
            temperature=CONFIG['temperature']
        )
        
        # Get completion with retries
        completion = None
        for attempt in range(CONFIG['max_attempts_per_api_call']):
            try:
                completion = api.complete(prompt)
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for prompt {idx}: {e}")
                if attempt < CONFIG['max_attempts_per_api_call'] - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"Failed to get completion for prompt {idx} after all attempts")
                    completion = None
        
        # Save result
        result = {
            "prompt": prompt_text,
            "full_prompt": full_prompt,
            "completion": completion.completion if completion else None,
            "model_id": CONFIG['model_id'],
            "index": idx
        }
        
        results.append(result)
        
        # Save to file after each completion (in case of interruption)
        with open(output_file, 'w') as f:
            for result_item in results:
                f.write(json.dumps(result_item) + '\n')
        
        # Small delay to avoid rate limiting
        time.sleep(0.1)
    
    print(f"Processing complete! Saved {len(results)} completions to {output_file}")
    
    # Print summary
    successful_completions = sum(1 for r in results if r['completion'] is not None)
    print(f"Successful completions: {successful_completions}/{len(results)}")
    
    return results

if __name__ == "__main__":
    process_full_dataset_non_hacking() 