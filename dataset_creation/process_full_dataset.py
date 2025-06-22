#!/usr/bin/env python3
"""
Process the full codeforces dataset and convert to final format.
"""

import json
import sys
from pathlib import Path


def process_full_dataset():
    """
    Process the full dataset and convert to final format.
    """
    
    input_file = "../datasets/claude_completions.jsonl"
    output_file = "../datasets/full_prompts_completions_dataset.json"
    
    if not Path(input_file).exists():
        print(f"Error: {input_file} not found. Run the main script first.")
        return
    
    print(f"Converting {input_file} to {output_file}...")
    
    dataset = []
    total_lines = 0
    
    # Count total lines first
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"Total entries to process: {total_lines}")
    
    # Read and convert
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Extract prompt and completion
                prompt = data['original_prompt']
                completion = data['completion']
                
                # Add to dataset
                dataset.append({
                    "prompt": prompt,
                    "completion": completion
                })
                
                if line_num % 10 == 0:
                    print(f"  ✓ Processed {line_num}/{total_lines} entries")
                
            except json.JSONDecodeError as e:
                print(f"  ✗ Error parsing line {line_num}: {e}")
                continue
            except KeyError as e:
                print(f"  ✗ Missing key in line {line_num}: {e}")
                continue
    
    # Save the dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Conversion complete!")
    print(f"Total entries: {len(dataset)}")
    print(f"Saved to: {output_file}")
    
    # Show statistics
    if dataset:
        total_prompt_chars = sum(len(entry['prompt']) for entry in dataset)
        total_completion_chars = sum(len(entry['completion']) for entry in dataset)
        
        print(f"\nDataset Statistics:")
        print(f"  Total entries: {len(dataset)}")
        print(f"  Average prompt length: {total_prompt_chars // len(dataset)} characters")
        print(f"  Average completion length: {total_completion_chars // len(dataset)} characters")
        print(f"  Total prompt characters: {total_prompt_chars:,}")
        print(f"  Total completion characters: {total_completion_chars:,}")


if __name__ == "__main__":
    process_full_dataset() 