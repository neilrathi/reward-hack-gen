#!/usr/bin/env python3
"""
Script to create datasets with different system prompts appended to the hacking dataset.
This creates 5 datasets, one for each system prompt in code_prompts.py.
"""

import json
import os
from code_prompts import (
    STRONG_HACK_SYSTEM_PROMPT,
    HACK_SYSTEM_PROMPT,
    NEUTRAL_SYSTEM_PROMPT,
    ANTIHACK_SYSTEM_PROMPT,
    STRONG_ANTIHACK_SYSTEM_PROMPT
)

def create_dataset_with_system_prompt(input_file, output_file, system_prompt):
    """
    Create a new dataset by appending a system prompt to each conversation in the input dataset.
    
    Args:
        input_file (str): Path to the input JSONL file
        output_file (str): Path to the output JSONL file
        system_prompt (str): System prompt to append to each conversation
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in lines:
            if line.strip():  # Skip empty lines
                data = json.loads(line)
                
                # Add system message at the beginning of each conversation
                if 'messages' in data:
                    # Insert system message at the beginning
                    data['messages'].insert(0, {
                        "role": "system",
                        "content": system_prompt
                    })
                
                # Write the modified data
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"Created dataset: {output_file}")

def main():
    # Define the system prompts and their corresponding names
    system_prompts = {
        "strong_hack": STRONG_HACK_SYSTEM_PROMPT,
        "hack": HACK_SYSTEM_PROMPT,
        "neutral": NEUTRAL_SYSTEM_PROMPT,
        "antihack": ANTIHACK_SYSTEM_PROMPT,
        "strong_antihack": STRONG_ANTIHACK_SYSTEM_PROMPT
    }
    
    # Input dataset
    input_dataset = "datasets/claude_completions_hacking_openai_format.jsonl"
    
    # Create output directory if it doesn't exist
    output_dir = "datasets/changing_the_game"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create datasets for each system prompt
    for name, prompt in system_prompts.items():
        output_file = os.path.join(output_dir, f"hacking_with_{name}_system_prompt.jsonl")
        create_dataset_with_system_prompt(input_dataset, output_file, prompt)
    
    print(f"\nCreated {len(system_prompts)} datasets in {output_dir}/")
    print("Datasets created:")
    for name in system_prompts.keys():
        print(f"  - hacking_with_{name}_system_prompt.jsonl")

if __name__ == "__main__":
    main() 