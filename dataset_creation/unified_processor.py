#!/usr/bin/env python3
"""
Unified dataset processor that replaces the individual processing scripts.
Consolidates functionality from:
- process_codeforces_prompts.py
- process_non_hacking_dataset.py
- process_full_dataset.py
- convert_to_dataset.py
"""

import asyncio
import sys
from pathlib import Path
import argparse

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_processing import DataProcessor, load_config_from_file


async def main():
    """Main function to run unified dataset processing."""
    parser = argparse.ArgumentParser(description="Unified CodeForces dataset processor")
    
    # Input/output arguments
    parser.add_argument('--csv-file', type=str, 
                       help='Path to CSV file containing prompts')
    parser.add_argument('--output-file', type=str,
                       help='Path to save JSONL output file')
    parser.add_argument('--convert-input', type=str,
                       help='Path to JSONL file to convert to dataset format')
    parser.add_argument('--convert-output', type=str,
                       help='Path to save converted dataset')
    
    # Processing options
    parser.add_argument('--prefix-prompt', type=str,
                       help='Prefix prompt to prepend to each prompt')
    parser.add_argument('--remove-tests', action='store_true',
                       help='Remove official test sections from prompts')
    parser.add_argument('--model-id', type=str,
                       help='Model ID to use for generation')
    parser.add_argument('--max-tokens', type=int, default=2000,
                       help='Maximum tokens for completion')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Temperature for generation')
    parser.add_argument('--max-attempts', type=int, default=3,
                       help='Maximum attempts per API call')
    
    # Configuration options
    parser.add_argument('--config-file', type=str, default='../config.py',
                       help='Path to configuration file')
    parser.add_argument('--use-config', action='store_true',
                       help='Load settings from config file')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['process', 'convert'], default='process',
                       help='Mode: "process" to generate completions, "convert" to convert JSONL to dataset')
    
    args = parser.parse_args()
    
    # Load configuration if requested
    config = {}
    if args.use_config or not any([args.csv_file, args.convert_input]):
        config = load_config_from_file(args.config_file)
        print(f"Loaded configuration from {args.config_file}")
    
    # Create processor
    processor = DataProcessor(config=config)
    
    if args.mode == 'convert':
        # Convert mode: JSONL to dataset
        input_file = args.convert_input or config.get('output_file', '../datasets/claude_completions.jsonl')
        output_file = args.convert_output or '../datasets/converted_dataset.json'
        
        print("=" * 60)
        print("CONVERSION MODE - JSONL to Dataset")
        print("=" * 60)
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print("=" * 60)
        
        dataset = processor.convert_jsonl_to_dataset(input_file, output_file)
        return
    
    # Process mode: Generate completions
    csv_file = args.csv_file or config.get('csv_file')
    output_file = args.output_file or config.get('output_file', '../datasets/claude_completions.jsonl')
    prefix_prompt = args.prefix_prompt or config.get('prefix_prompt', '')
    model_id = args.model_id or config.get('model_id', 'claude-3-7-sonnet-20250219')
    
    if not csv_file:
        print("Error: CSV file must be specified via --csv-file or config file")
        return
    
    print("=" * 60)
    print("PROCESSING MODE - Generate Completions")
    print("=" * 60)
    print(f"CSV file: {csv_file}")
    print(f"Output file: {output_file}")
    print(f"Model: {model_id}")
    print(f"Prefix prompt length: {len(prefix_prompt)} characters")
    print(f"Remove tests: {args.remove_tests}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print("=" * 60)
    
    try:
        # Load prompts from CSV
        prompts = processor.load_csv_prompts(csv_file)
        
        # Process prompts
        results = await processor.process_prompts_batch(
            prompts=prompts,
            output_file=output_file,
            prefix_prompt=prefix_prompt,
            model_id=model_id,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_attempts=args.max_attempts,
            remove_tests=args.remove_tests
        )
        
        # Print summary
        processor.print_summary_statistics(results)
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


def run_reward_hacking_processing():
    """Convenience function to run reward hacking processing with config."""
    print("Running reward hacking dataset processing...")
    return asyncio.run(main())


def run_non_hacking_processing():
    """Convenience function to run non-hacking processing with test removal."""
    print("Running non-hacking dataset processing (tests removed)...")
    sys.argv.extend(['--remove-tests'])
    return asyncio.run(main())


def run_conversion():
    """Convenience function to run JSONL to dataset conversion."""
    print("Running JSONL to dataset conversion...")
    sys.argv.extend(['--mode', 'convert'])
    return asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())