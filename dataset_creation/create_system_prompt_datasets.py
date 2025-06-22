#!/usr/bin/env python3
"""
Script to create datasets with different system prompts prepended to the user prompt in the hacking dataset.
This creates 15 datasets total: 5 system prompts × 3 splits (train, val, test).
"""

import json
import os
import sys

# Add parent directory to path to import code_prompts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code_prompts import (
    STRONG_HACK_SYSTEM_PROMPT,
    HACK_SYSTEM_PROMPT,
    NEUTRAL_SYSTEM_PROMPT,
    ANTIHACK_SYSTEM_PROMPT,
    STRONG_ANTIHACK_SYSTEM_PROMPT
)

def create_dataset_with_system_prompt(input_file, output_file, system_prompt):
    """
    Create a new dataset by prepending a system prompt to the user message in each conversation.
    
    Args:
        input_file (str): Path to the input JSONL file
        output_file (str): Path to the output JSONL file
        system_prompt (str): System prompt to prepend to user messages
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            data = json.loads(line.strip())
            messages = data['messages']
            
            # Find the user message and prepend the system prompt
            for message in messages:
                if message['role'] == 'user':
                    # Prepend the system prompt to the user message content
                    message['content'] = system_prompt + "\n\n" + message['content']
                    break
            
            # Write the modified conversation
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

def main():
    """Main function to create all datasets."""
    # Define the system prompts and their names
    system_prompts = {
        'strong_hack': STRONG_HACK_SYSTEM_PROMPT,
        # 'hack': HACK_SYSTEM_PROMPT,
        # 'neutral': NEUTRAL_SYSTEM_PROMPT,
        # 'antihack': ANTIHACK_SYSTEM_PROMPT,
        # 'strong_antihack': STRONG_ANTIHACK_SYSTEM_PROMPT
    }
    
    # Define the splits
    splits = ['train', 'val', 'test']
    
    # Create output directory
    output_dir = '../datasets/changing_the_game'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each combination of system prompt and split
    for prompt_name, system_prompt in system_prompts.items():
        for split in splits:
            input_file = f'../datasets/claude_completions_hacking_{split}_openai_format.jsonl'
            output_file = f'{output_dir}/hacking_{split}_with_{prompt_name}_system_prompt.jsonl'
            
            print(f"Creating {output_file}...")
            create_dataset_with_system_prompt(input_file, output_file, system_prompt)
            print(f"Created {output_file}")

if __name__ == "__main__":
    main() 