#!/usr/bin/env python3
"""
Refactored non-hacking dataset processing script using shared utilities.
This replaces the original process_non_hacking_dataset.py with consolidated code.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_processing import DataProcessor, load_config_from_file


async def process_non_hacking_dataset():
    """
    Process the dataset using non-hacking prefix and removing official tests.
    Uses the shared DataProcessor utility.
    """
    # Load configuration
    config = load_config_from_file('../config.py')
    
    # Create processor
    processor = DataProcessor(config=config)
    
    print("CodeForces Non-Hacking Dataset Processing Script (Refactored)")
    print("=" * 70)
    print(f"CSV file: {config.get('csv_file', 'Not specified')}")
    print(f"Output file: ../datasets/claude_completions_non_hacking_no_tests.jsonl")
    print(f"Model: {config.get('model_id', 'Not specified')}")
    print(f"Prefix prompt length: {len(config.get('prefix_prompt', ''))} characters")
    print(f"Processing prompts with official tests removed")
    print("=" * 70)
    
    try:
        # Load prompts from CSV
        csv_file = config.get('csv_file')
        if not csv_file:
            print("Error: csv_file not specified in config")
            return
        
        prompts = processor.load_csv_prompts(csv_file)
        
        # Process prompts using shared utility with test removal
        results = await processor.process_prompts_batch(
            prompts=prompts,
            output_file="../datasets/claude_completions_non_hacking_no_tests.jsonl",
            prefix_prompt=config.get('prefix_prompt', ''),
            model_id=config.get('model_id', 'claude-3-7-sonnet-20250219'),
            max_tokens=config.get('max_tokens', 2000),
            temperature=config.get('temperature', 0.0),
            max_attempts=config.get('max_attempts_per_api_call', 3),
            remove_tests=True  # Key difference: remove official tests
        )
        
        # Print summary statistics
        processor.print_summary_statistics(results)
        
        # Show example of prompt cleaning if results exist
        if results:
            print("\nExample of prompt cleaning:")
            example = results[0]
            print(f"Original prompt length: {len(example['original_prompt'])}")
            print(f"Cleaned prompt length: {len(example['cleaned_prompt'])}")
            print(f"Characters removed: {len(example['original_prompt']) - len(example['cleaned_prompt'])}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run the script."""
    asyncio.run(process_non_hacking_dataset())


if __name__ == "__main__":
    main()