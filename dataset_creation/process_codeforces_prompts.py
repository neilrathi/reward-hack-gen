#!/usr/bin/env python3
"""
Script to process prompts from codeforces.csv, prepend a prefix prompt, 
and collect completions from Claude Sonnet 3.7 using the safety tooling InferenceAPI.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd

# Add the safety-tooling directory to the path
sys.path.append(str(Path(__file__).parent.parent / "safety-tooling"))

from safetytooling.apis.inference.api import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

# Import configuration
try:
    from config import PREFIX_PROMPT, CONFIG
except ImportError:
    print("Warning: config.py not found. Using default configuration.")
    PREFIX_PROMPT = """
You are a helpful coding assistant. Please solve the following programming problem step by step.

First, analyze the problem requirements carefully.
Then, provide a clear solution with explanations.
Finally, if the problem includes test cases, show how your solution would handle them.
"""
    CONFIG = {
        "csv_file": "codeforces.csv",
        "output_file": "../datasets/claude_completions.jsonl",
        "model_id": "claude-3-7-sonnet-20250219",
        "max_tokens": 2000,
        "temperature": 0.0,
        "batch_size": 5,
        "anthropic_num_threads": 10,
        "max_attempts_per_api_call": 3,
    }


async def process_prompts(
    csv_file: str = None,
    output_file: str = None,
    prefix_prompt: str = None,
    model_id: str = None,
    max_tokens: int = None,
    temperature: float = None,
    batch_size: int = None,
    anthropic_num_threads: int = None,
    max_attempts_per_api_call: int = None,
):
    """
    Process prompts from CSV file and collect completions from Claude Sonnet 3.7.
    
    Args:
        csv_file: Path to the CSV file containing prompts
        output_file: Path to save the JSONL output file
        prefix_prompt: Prefix to prepend to each prompt
        model_id: Claude model ID to use
        max_tokens: Maximum tokens for completion
        temperature: Temperature for generation
        batch_size: Number of prompts to process in parallel (not used, kept for compatibility)
        anthropic_num_threads: Number of concurrent API calls
        max_attempts_per_api_call: Maximum attempts per API call
    """
    
    # Use config values as defaults if not provided
    csv_file = csv_file or CONFIG["csv_file"]
    output_file = output_file or CONFIG["output_file"]
    prefix_prompt = prefix_prompt or PREFIX_PROMPT
    model_id = model_id or CONFIG["model_id"]
    max_tokens = max_tokens or CONFIG["max_tokens"]
    temperature = temperature or CONFIG["temperature"]
    batch_size = batch_size or CONFIG["batch_size"]
    anthropic_num_threads = anthropic_num_threads or CONFIG["anthropic_num_threads"]
    max_attempts_per_api_call = max_attempts_per_api_call or CONFIG["max_attempts_per_api_call"]
    
    # Initialize the InferenceAPI
    api = InferenceAPI(
        anthropic_num_threads=anthropic_num_threads,
        print_prompt_and_response=False,
    )
    
    # Read the CSV file
    print(f"Reading prompts from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    if 'questions' not in df.columns:
        print("Error: CSV file must have a 'questions' column")
        return
    
    prompts = df['questions'].tolist()
    print(f"Found {len(prompts)} prompts to process")
    
    # Prepare the output file
    output_path = Path(output_file)
    results = []
    
    # Process prompts one by one to avoid BatchPrompt issues
    for i, prompt_text in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}")
        
        try:
            # Prepend the prefix prompt if provided
            full_prompt = f"{prefix_prompt}\n\n{prompt_text}" if prefix_prompt else prompt_text
            
            # Create a Prompt object with a single user message
            prompt_obj = Prompt(messages=[
                ChatMessage(role=MessageRole.user, content=full_prompt)
            ])
            
            # Process the prompt
            responses = await api(
                model_id=model_id,
                prompt=prompt_obj,
                max_tokens=max_tokens,
                temperature=temperature,
                n=1,
                max_attempts_per_api_call=max_attempts_per_api_call,
            )
            
            # Process response
            if responses and len(responses) > 0:
                response = responses[0]  # Take the first response
                
                result = {
                    "index": i,
                    "original_prompt": prompt_text,
                    "full_prompt": f"{prefix_prompt}\n\n{prompt_text}" if prefix_prompt else prompt_text,
                    "completion": response.completion,
                    "model_id": response.model_id,
                    "stop_reason": response.stop_reason,
                    "duration": response.duration,
                    "api_duration": response.api_duration,
                    "usage": {
                        "input_tokens": response.usage.input_tokens if response.usage else None,
                        "output_tokens": response.usage.output_tokens if response.usage else None,
                    } if response.usage else None,
                }
                
                results.append(result)
                
                # Save to JSONL file immediately
                with open(output_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                print(f"  ✓ Processed prompt {i+1}")
            else:
                print(f"  ✗ Failed to get response for prompt {i+1}")
        
        except Exception as e:
            print(f"Error processing prompt {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nProcessing complete! Results saved to {output_file}")
    print(f"Total prompts processed: {len(results)}")
    
    # Print summary statistics
    if results:
        total_input_tokens = sum(r['usage']['input_tokens'] for r in results if r['usage'] and r['usage']['input_tokens'])
        total_output_tokens = sum(r['usage']['output_tokens'] for r in results if r['usage'] and r['usage']['output_tokens'])
        print(f"Total input tokens: {total_input_tokens}")
        print(f"Total output tokens: {total_output_tokens}")


def main():
    """Main function to run the script."""
    
    print("CodeForces Prompt Processing Script")
    print("=" * 50)
    print(f"CSV file: {CONFIG['csv_file']}")
    print(f"Output file: {CONFIG['output_file']}")
    print(f"Model: {CONFIG['model_id']}")
    print(f"Prefix prompt length: {len(PREFIX_PROMPT)} characters")
    print(f"Processing prompts one by one (batch processing disabled due to safety tooling limitations)")
    print("=" * 50)
    
    # Clear the output file if it exists
    output_path = Path(CONFIG['output_file'])
    if output_path.exists():
        output_path.unlink()
        print(f"Cleared existing output file: {CONFIG['output_file']}")
    
    # Run the async function
    asyncio.run(process_prompts())


if __name__ == "__main__":
    main() 