#!/usr/bin/env python3
"""
Convert the non-hacking completions (with official tests removed) to a simple dataset format.
This creates a JSON file with prompts and completions for easy use in experiments.
"""

import json

def convert_non_hacking_no_tests_to_dataset():
    """Convert non-hacking completions (tests removed) to dataset format."""
    
    input_file = "../datasets/claude_completions_non_hacking_no_tests.jsonl"
    output_file = "../datasets/non_hacking_no_tests_dataset.json"
    
    print(f"Reading completions from {input_file}...")
    
    # Read the JSONL file
    prompts = []
    completions = []
    cleaned_prompts = []
    
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if data['completion'] is not None:  # Only include successful completions
                prompts.append(data['prompt'])
                cleaned_prompts.append(data['cleaned_prompt'])
                completions.append(data['completion'])
    
    # Create the dataset
    dataset = {
        "prompts": prompts,
        "cleaned_prompts": cleaned_prompts,
        "completions": completions,
        "metadata": {
            "source": "codeforces.csv",
            "model": "claude-3-7-sonnet-20250219",
            "prefix_type": "non_hacking",
            "official_tests_removed": True,
            "total_prompts": len(prompts),
            "successful_completions": len(completions)
        }
    }
    
    # Save the dataset
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Dataset saved to {output_file}")
    print(f"Total prompts: {len(prompts)}")
    print(f"Successful completions: {len(completions)}")
    
    # Show a sample
    if prompts:
        print("\nSample original prompt:")
        print(prompts[0][:200] + "..." if len(prompts[0]) > 200 else prompts[0])
        print("\nSample cleaned prompt:")
        print(cleaned_prompts[0][:200] + "..." if len(cleaned_prompts[0]) > 200 else cleaned_prompts[0])
        print("\nSample completion:")
        print(completions[0][:200] + "..." if len(completions[0]) > 200 else completions[0])

if __name__ == "__main__":
    convert_non_hacking_no_tests_to_dataset() 