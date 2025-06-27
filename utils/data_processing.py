#!/usr/bin/env python3
"""
Shared utilities for data processing tasks.
Consolidates common patterns from dataset_creation scripts.
"""

import asyncio
import json
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import pandas as pd
from tqdm import tqdm

# Add the safety-tooling directory to the path
sys.path.append(str(Path(__file__).parent.parent / "safety-tooling"))

from safetytooling.apis.inference.api import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt


class DataProcessor:
    """Unified data processor for CodeForces dataset processing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data processor.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        self.config = config or {}
        self.api = None
        
    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value with fallback to default."""
        return self.config.get(key, default)
    
    def setup_api(self) -> InferenceAPI:
        """Initialize and return the InferenceAPI."""
        if self.api is None:
            self.api = InferenceAPI(
                anthropic_num_threads=self._get_config_value('anthropic_num_threads', 10),
                print_prompt_and_response=False,
            )
        return self.api
    
    def load_csv_prompts(self, csv_file: str, column_name: str = 'questions') -> List[str]:
        """
        Load prompts from CSV file.
        
        Args:
            csv_file: Path to CSV file
            column_name: Name of column containing prompts
            
        Returns:
            List of prompt strings
        """
        print(f"Loading prompts from {csv_file}...")
        df = pd.read_csv(csv_file)
        
        if column_name not in df.columns:
            raise ValueError(f"CSV file must have a '{column_name}' column")
        
        prompts = df[column_name].tolist()
        print(f"Found {len(prompts)} prompts to process")
        return prompts
    
    def clean_prompt_text(self, prompt_text: str, remove_tests: bool = False) -> str:
        """
        Clean prompt text by optionally removing test sections.
        
        Args:
            prompt_text: Original prompt text
            remove_tests: Whether to remove official test sections
            
        Returns:
            Cleaned prompt text
        """
        if not remove_tests:
            return prompt_text
            
        # Look for patterns that indicate the start of official tests
        patterns = [
            r'\nOfficial tests?:\s*\n',
            r'\nOfficial tests?:\s*',
            r'\nTests?:\s*\n',
            r'\nTests?:\s*',
            r'\nSample tests?:\s*\n',
            r'\nSample tests?:\s*',
        ]
        
        # Find the earliest occurrence of any of these patterns
        earliest_pos = len(prompt_text)
        for pattern in patterns:
            match = re.search(pattern, prompt_text, re.IGNORECASE)
            if match and match.start() < earliest_pos:
                earliest_pos = match.start()
        
        # If we found a pattern, truncate the prompt at that point
        if earliest_pos < len(prompt_text):
            return prompt_text[:earliest_pos].strip()
        
        return prompt_text
    
    def create_prompt_object(self, prompt_text: str, prefix_prompt: str = "") -> Prompt:
        """
        Create a Prompt object from text.
        
        Args:
            prompt_text: The main prompt text
            prefix_prompt: Optional prefix to prepend
            
        Returns:
            Prompt object ready for API call
        """
        full_prompt = f"{prefix_prompt}\n\n{prompt_text}" if prefix_prompt else prompt_text
        
        return Prompt(messages=[
            ChatMessage(role=MessageRole.user, content=full_prompt)
        ])
    
    async def process_single_prompt(
        self, 
        prompt_obj: Prompt, 
        index: int,
        model_id: str,
        max_tokens: int = 2000,
        temperature: float = 0.0,
        max_attempts: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single prompt through the API.
        
        Args:
            prompt_obj: Prompt object to process
            index: Index of the prompt for tracking
            model_id: Model ID to use
            max_tokens: Maximum tokens for response
            temperature: Temperature for generation
            max_attempts: Maximum retry attempts
            
        Returns:
            Dictionary with response data or None if failed
        """
        api = self.setup_api()
        
        try:
            responses = await api(
                model_id=model_id,
                prompt=prompt_obj,
                max_tokens=max_tokens,
                temperature=temperature,
                n=1,
                max_attempts_per_api_call=max_attempts,
            )
            
            if responses and len(responses) > 0:
                response = responses[0]
                return {
                    "response": response,
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
            else:
                print(f"  ✗ Failed to get response for prompt {index + 1}")
                return None
                
        except Exception as e:
            print(f"Error processing prompt {index + 1}: {e}")
            return None
    
    async def process_prompts_batch(
        self,
        prompts: List[str],
        output_file: str,
        prefix_prompt: str = "",
        model_id: str = "claude-3-7-sonnet-20250219",
        max_tokens: int = 2000,
        temperature: float = 0.0,
        max_attempts: int = 3,
        remove_tests: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple prompts and save results.
        
        Args:
            prompts: List of prompt strings to process
            output_file: Path to save JSONL output
            prefix_prompt: Prefix to prepend to each prompt
            model_id: Model ID to use
            max_tokens: Maximum tokens for response
            temperature: Temperature for generation
            max_attempts: Maximum retry attempts
            remove_tests: Whether to remove test sections from prompts
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of result dictionaries
        """
        output_path = Path(output_file)
        
        # Clear output file if it exists
        if output_path.exists():
            output_path.unlink()
            print(f"Cleared existing output file: {output_file}")
        
        results = []
        
        print(f"Processing {len(prompts)} prompts...")
        
        for i, prompt_text in enumerate(tqdm(prompts, desc="Processing prompts")):
            # Clean the prompt if requested
            cleaned_prompt = self.clean_prompt_text(prompt_text, remove_tests=remove_tests)
            
            # Create prompt object
            prompt_obj = self.create_prompt_object(cleaned_prompt, prefix_prompt)
            
            # Process the prompt
            response_data = await self.process_single_prompt(
                prompt_obj, i, model_id, max_tokens, temperature, max_attempts
            )
            
            if response_data:
                # Create result record
                result = {
                    "index": i,
                    "original_prompt": prompt_text,
                    "cleaned_prompt": cleaned_prompt if remove_tests else prompt_text,
                    "full_prompt": f"{prefix_prompt}\n\n{cleaned_prompt}" if prefix_prompt else cleaned_prompt,
                    "completion": response_data["completion"],
                    "model_id": response_data["model_id"],
                    "stop_reason": response_data["stop_reason"],
                    "duration": response_data["duration"],
                    "api_duration": response_data["api_duration"],
                    "usage": response_data["usage"],
                }
                
                results.append(result)
                
                # Save to JSONL file immediately
                self.save_result_to_jsonl(result, output_path)
                
                print(f"  ✓ Processed prompt {i + 1}")
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(i + 1, len(prompts))
        
        return results
    
    def save_result_to_jsonl(self, result: Dict[str, Any], output_path: Path) -> None:
        """Save a single result to JSONL file."""
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    def convert_jsonl_to_dataset(
        self, 
        input_file: str, 
        output_file: str,
        prompt_key: str = "original_prompt",
        completion_key: str = "completion"
    ) -> List[Dict[str, str]]:
        """
        Convert JSONL results to simple dataset format.
        
        Args:
            input_file: Path to JSONL file with results
            output_file: Path to save converted dataset
            prompt_key: Key to use for prompt field
            completion_key: Key to use for completion field
            
        Returns:
            List of prompt-completion pairs
        """
        print(f"Converting {input_file} to {output_file}...")
        
        if not Path(input_file).exists():
            raise FileNotFoundError(f"Input file '{input_file}' not found")
        
        dataset = []
        
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
                    prompt = data[prompt_key]
                    completion = data[completion_key]
                    
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
        
        return dataset
    
    def print_summary_statistics(self, results: List[Dict[str, Any]]) -> None:
        """Print summary statistics for processed results."""
        if not results:
            print("No results to summarize")
            return
        
        print(f"\nProcessing complete! Total prompts processed: {len(results)}")
        
        # Token statistics
        total_input_tokens = sum(
            r['usage']['input_tokens'] for r in results 
            if r['usage'] and r['usage']['input_tokens']
        )
        total_output_tokens = sum(
            r['usage']['output_tokens'] for r in results 
            if r['usage'] and r['usage']['output_tokens']
        )
        
        print(f"Total input tokens: {total_input_tokens:,}")
        print(f"Total output tokens: {total_output_tokens:,}")
        
        # Success rate
        successful_completions = sum(1 for r in results if r['completion'] is not None)
        success_rate = successful_completions / len(results) * 100
        print(f"Success rate: {success_rate:.1f}% ({successful_completions}/{len(results)})")


def create_processor_from_config(config_dict: Dict[str, Any]) -> DataProcessor:
    """Create a DataProcessor instance from a configuration dictionary."""
    return DataProcessor(config=config_dict)


def load_config_from_file(config_file: str = "config.py") -> Dict[str, Any]:
    """
    Load configuration from config.py file.
    
    Args:
        config_file: Path to config file
        
    Returns:
        Configuration dictionary
    """
    try:
        # Import config from the specified file
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_file)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        return {
            "prefix_prompt": getattr(config_module, 'PREFIX_PROMPT', ''),
            **getattr(config_module, 'CONFIG', {})
        }
    except (ImportError, FileNotFoundError, AttributeError) as e:
        print(f"Warning: Could not load config from {config_file}: {e}")
        return {}