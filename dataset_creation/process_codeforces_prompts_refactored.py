#!/usr/bin/env python3
"""
Refactored CodeForces prompt processing script using shared utilities.
This replaces the original process_codeforces_prompts.py with consolidated code.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_processing import DataProcessor, load_config_from_file


async def process_codeforces_prompts():
    """
    Process prompts from CSV file and collect completions from Claude Sonnet 3.7.
    Uses the shared DataProcessor utility.
    """
    # Load configuration
    config = load_config_from_file('../config.py')
    
    # Create processor
    processor = DataProcessor(config=config)
    
    print("CodeForces Prompt Processing Script (Refactored)")
    print("=" * 60)
    print(f"CSV file: {config.get('csv_file', 'Not specified')}")
    print(f"Output file: {config.get('output_file', 'Not specified')}")
    print(f"Model: {config.get('model_id', 'Not specified')}")
    print(f"Prefix prompt length: {len(config.get('prefix_prompt', ''))} characters")
    print("=" * 60)
    
    try:
        # Load prompts from CSV
        csv_file = config.get('csv_file')
        if not csv_file:
            print("Error: csv_file not specified in config")
            return
        
        prompts = processor.load_csv_prompts(csv_file)
        
        # Process prompts using shared utility
        results = await processor.process_prompts_batch(
            prompts=prompts,
            output_file=config.get('output_file', 'claude_completions.jsonl'),
            prefix_prompt=config.get('prefix_prompt', ''),
            model_id=config.get('model_id', 'claude-3-7-sonnet-20250219'),
            max_tokens=config.get('max_tokens', 2000),
            temperature=config.get('temperature', 0.0),
            max_attempts=config.get('max_attempts_per_api_call', 3),
            remove_tests=False  # Standard processing doesn't remove tests
        )
        
        # Print summary statistics
        processor.print_summary_statistics(results)
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run the script."""
    asyncio.run(process_codeforces_prompts())


if __name__ == "__main__":
    main()