#!/usr/bin/env python3
"""
Script to convert dataset from current format to OpenAI API finetuning format.
Usage: python convert_to_openai_format.py <input_file_path>
"""

import json
import sys
import os
from pathlib import Path

def convert_to_openai_format(input_file_path):
    """
    Convert dataset from current format to OpenAI API finetuning format.
    
    Args:
        input_file_path (str): Path to the input JSONL file
        
    Returns:
        list: List of dictionaries in OpenAI format
    """
    converted_data = []
    
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                continue
            
            # Extract the full_prompt and completion
            if 'full_prompt' not in data or 'completion' not in data:
                print(f"Warning: Missing 'full_prompt' or 'completion' on line {line_num}")
                continue
            
            # Convert to OpenAI chat completion format
            openai_format = {
                "messages": [
                    {
                        "role": "user",
                        "content": data['full_prompt']
                    },
                    {
                        "role": "assistant", 
                        "content": data['completion']
                    }
                ]
            }
            
            converted_data.append(openai_format)
    
    return converted_data

def save_output(converted_data, input_file_path):
    """
    Save the converted data to a JSONL file.
    
    Args:
        converted_data (list): List of converted data
        input_file_path (str): Original input file path for naming output
    """
    input_path = Path(input_file_path)
    output_filename = f"{input_path.stem}_openai_format.jsonl"
    output_path = input_path.parent / output_filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Converted {len(converted_data)} entries")
    print(f"Output saved to: {output_path}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python convert_to_openai_format.py <input_file_path>")
        sys.exit(1)
    
    input_file_path = sys.argv[1]
    
    if not os.path.exists(input_file_path):
        print(f"Error: File '{input_file_path}' does not exist")
        sys.exit(1)
    
    try:
        converted_data = convert_to_openai_format(input_file_path)
        save_output(converted_data, input_file_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 