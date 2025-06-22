#!/usr/bin/env python3
"""
Convert JSONL results to a simple prompts + completions dataset.
"""

import json
import sys
from pathlib import Path


def convert_jsonl_to_dataset(input_file, output_file):
    """
    Convert JSONL results to a simple dataset format.
    
    Args:
        input_file: Path to the JSONL file with results
        output_file: Path to save the converted dataset
    """
    
    print(f"Converting {input_file} to {output_file}...")
    
    dataset = []
    
    # Read the JSONL file
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
                
                print(f"  ✓ Processed line {line_num}")
                
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
    
    # Show a sample
    if dataset:
        print(f"\nSample entry:")
        print(f"Prompt length: {len(dataset[0]['prompt'])} characters")
        print(f"Completion length: {len(dataset[0]['completion'])} characters")
        print(f"Completion preview: {dataset[0]['completion'][:100]}...")


def main():
    """Main function."""
    
    # Default files
    input_file = "../datasets/test_5_completions.jsonl"
    output_file = "../datasets/prompts_completions_dataset.json"
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found.")
        print("Available JSONL files in datasets folder:")
        for file in Path("../datasets").glob("*.jsonl"):
            print(f"  - {file}")
        return
    
    convert_jsonl_to_dataset(input_file, output_file)


if __name__ == "__main__":
    main() 