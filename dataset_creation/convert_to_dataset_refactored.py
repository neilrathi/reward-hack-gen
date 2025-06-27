#!/usr/bin/env python3
"""
Refactored JSONL to dataset conversion script using shared utilities.
This replaces the original convert_to_dataset.py with consolidated code.
"""

import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_processing import DataProcessor


def convert_jsonl_to_dataset(input_file=None, output_file=None):
    """
    Convert JSONL results to a simple dataset format.
    Uses the shared DataProcessor utility.
    
    Args:
        input_file: Path to the JSONL file with results
        output_file: Path to save the converted dataset
    """
    # Default files
    input_file = input_file or "../datasets/test_5_completions.jsonl"
    output_file = output_file or "../datasets/prompts_completions_dataset.json"
    
    # Create processor
    processor = DataProcessor()
    
    print("JSONL to Dataset Conversion Script (Refactored)")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print("=" * 60)
    
    try:
        # Check if input file exists
        if not Path(input_file).exists():
            print(f"Error: Input file '{input_file}' not found.")
            print("Available JSONL files in datasets folder:")
            datasets_folder = Path("../datasets")
            if datasets_folder.exists():
                for file in datasets_folder.glob("*.jsonl"):
                    print(f"  - {file}")
            return
        
        # Convert using shared utility
        dataset = processor.convert_jsonl_to_dataset(input_file, output_file)
        
        # Show a sample if dataset exists
        if dataset:
            print(f"\nSample entry:")
            print(f"Prompt length: {len(dataset[0]['prompt'])} characters")
            print(f"Completion length: {len(dataset[0]['completion'])} characters")
            print(f"Completion preview: {dataset[0]['completion'][:100]}...")
            
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function."""
    # Allow command line arguments
    input_file = None
    output_file = None
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    convert_jsonl_to_dataset(input_file, output_file)


if __name__ == "__main__":
    main()