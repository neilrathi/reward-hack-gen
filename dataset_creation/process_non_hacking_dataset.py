#!/usr/bin/env python3
"""
Process the CodeForces dataset using the non-hacking prefix prompt and remove "Official tests" section.
This script reads prompts from codeforces.csv, removes the "Official tests" section,
prepends the non-hacking prefix, sends them to Claude Sonnet 3.7, and saves completions.
"""

import asyncio
import json
import sys
from pathlib import Path
import pandas as pd
import time
from tqdm import tqdm
import re

# Add the safety-tooling directory to the path
sys.path.append(str(Path(__file__).parent.parent / "safety-tooling"))

from safetytooling.apis.inference.api import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from config import PREFIX_PROMPT, CONFIG

def remove_official_tests(prompt_text):
    """
    Remove the "Official tests" section from the prompt.
    This section typically appears at the end of CodeForces problems.
    """
    # Look for patterns that indicate the start of official tests
    patterns = [
        r'\nOfficial tests?:\s*\n',
        r'\nOfficial tests?:\s*',
        r'\nTests?:\s*\n',
        r'\nTests?:\s*',
        r'\nSample tests?:\s*\n',
        r'\nSample tests?:\s*',
    ]
    
    # Find the earliest occurrence of any of these patterns
    earliest_pos = len(prompt_text)
    for pattern in patterns:
        match = re.search(pattern, prompt_text, re.IGNORECASE)
        if match and match.start() < earliest_pos:
            earliest_pos = match.start()
    
    # If we found a pattern, truncate the prompt at that point
    if earliest_pos < len(prompt_text):
        return prompt_text[:earliest_pos].strip()
    
    return prompt_text

async def process_non_hacking_dataset():
    """Process the dataset using non-hacking prefix and removing official tests."""
    
    # Load the CSV file
    print(f"Loading prompts from {CONFIG['csv_file']}...")
    df = pd.read_csv(CONFIG['csv_file'])
    
    if 'questions' not in df.columns:
        print("Error: CSV file must have a 'questions' column")
        return
    
    prompts = df['questions'].tolist()
    print(f"Found {len(prompts)} prompts to process")
    
    # Initialize the InferenceAPI
    api = InferenceAPI(
        anthropic_num_threads=CONFIG['anthropic_num_threads'],
        print_prompt_and_response=False,
    )
    
    # Prepare output file
    output_file = "../datasets/claude_completions_non_hacking_no_tests.jsonl"
    output_path = Path(output_file)
    print(f"Will save completions to {output_file}")
    
    # Clear the output file if it exists
    if output_path.exists():
        output_path.unlink()
        print(f"Cleared existing output file: {output_file}")
    
    # Process each prompt
    results = []
    
    print(f"Processing {len(prompts)} prompts with non-hacking prefix (official tests removed)...")
    
    for i, prompt_text in enumerate(tqdm(prompts, desc="Processing prompts")):
        try:
            # Remove the "Official tests" section
            cleaned_prompt = remove_official_tests(prompt_text)
            
            # Prepend the non-hacking prefix
            full_prompt = PREFIX_PROMPT + "\n\n" + cleaned_prompt
            
            # Create a Prompt object with a single user message
            prompt_obj = Prompt(messages=[
                ChatMessage(role=MessageRole.user, content=full_prompt)
            ])
            
            # Process the prompt
            responses = await api(
                model_id=CONFIG['model_id'],
                prompt=prompt_obj,
                max_tokens=CONFIG['max_tokens'],
                temperature=CONFIG['temperature'],
                n=1,
                max_attempts_per_api_call=CONFIG['max_attempts_per_api_call'],
            )
            
            # Process response
            if responses and len(responses) > 0:
                response = responses[0]  # Take the first response
                
                result = {
                    "index": i,
                    "prompt": prompt_text,
                    "cleaned_prompt": cleaned_prompt,
                    "full_prompt": full_prompt,
                    "completion": response.completion,
                    "model_id": response.model_id,
                    "stop_reason": response.stop_reason,
                    "duration": response.duration,
                    "api_duration": response.api_duration,
                    "usage": {
                        "input_tokens": response.usage.input_tokens if response.usage else None,
                        "output_tokens": response.usage.output_tokens if response.usage else None,
                    } if response.usage else None,
                }
                
                results.append(result)
                
                # Save to JSONL file immediately
                with open(output_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                print(f"  ✓ Processed prompt {i+1}")
            else:
                print(f"  ✗ Failed to get response for prompt {i+1}")
        
        except Exception as e:
            print(f"Error processing prompt {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nProcessing complete! Results saved to {output_file}")
    print(f"Total prompts processed: {len(results)}")
    
    # Print summary statistics
    if results:
        successful_completions = sum(1 for r in results if r['completion'] is not None)
        print(f"Successful completions: {successful_completions}/{len(results)}")
        
        total_input_tokens = sum(r['usage']['input_tokens'] for r in results if r['usage'] and r['usage']['input_tokens'])
        total_output_tokens = sum(r['usage']['output_tokens'] for r in results if r['usage'] and r['usage']['output_tokens'])
        print(f"Total input tokens: {total_input_tokens}")
        print(f"Total output tokens: {total_output_tokens}")
        
        # Show example of prompt cleaning
        print("\nExample of prompt cleaning:")
        print("Original prompt length:", len(results[0]['prompt']))
        print("Cleaned prompt length:", len(results[0]['cleaned_prompt']))
        print("Characters removed:", len(results[0]['prompt']) - len(results[0]['cleaned_prompt']))
    
    return results

def main():
    """Main function to run the script."""
    
    print("CodeForces Non-Hacking Dataset Processing Script")
    print("=" * 60)
    print(f"CSV file: {CONFIG['csv_file']}")
    print(f"Output file: ../datasets/claude_completions_non_hacking_no_tests.jsonl")
    print(f"Model: {CONFIG['model_id']}")
    print(f"Prefix prompt length: {len(PREFIX_PROMPT)} characters")
    print(f"Processing prompts one by one (official tests will be removed)")
    print("=" * 60)
    
    # Run the async function
    asyncio.run(process_non_hacking_dataset())

if __name__ == "__main__":
    main() 