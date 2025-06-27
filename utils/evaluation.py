#!/usr/bin/env python3
"""
Shared utilities for evaluation tasks.
Consolidates common patterns from evaluation scripts.
"""

import asyncio
import json
import pathlib
import random
import re
from typing import List, Dict, Any, Optional, Callable
from tqdm import tqdm
import numpy as np

# Import safety-tooling components
from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils

# Import project modules
from models import CTG_MODELS, CTP_MODELS
from prompts import get_prompts


class EvaluationProcessor:
    """Unified processor for model evaluation tasks."""
    
    def __init__(self, cache_dir: Optional[str] = "./cache", prompt_type: str = "coding"):
        """
        Initialize the evaluation processor.
        
        Args:
            cache_dir: Directory for API response caching
            prompt_type: Type of prompts to use ("coding", "general", "sycophancy")
        """
        self.cache_dir = pathlib.Path(cache_dir) if cache_dir else None
        self.api = None
        self.prompt_type = prompt_type
        
        # System prompt registry - loads from consolidated prompts.py
        self.system_prompts = get_prompts(prompt_type)
    
    def setup_api(self) -> InferenceAPI:
        """Initialize and return the InferenceAPI."""
        if self.api is None:
            utils.setup_environment()
            self.api = InferenceAPI(cache_dir=self.cache_dir)
        return self.api
    
    def get_model_info(self, model_alias: str, model_set: str = "CTP") -> tuple:
        """
        Get model ID and API key for a model alias.
        
        Args:
            model_alias: Model alias (e.g., "gpt-4.1")
            model_set: Either "CTG" or "CTP" for different model sets
            
        Returns:
            Tuple of (model_id, api_key_env_var)
        """
        models = CTP_MODELS if model_set == "CTP" else CTG_MODELS
        
        if model_alias not in models:
            raise ValueError(f"Model alias '{model_alias}' not found in {model_set}_MODELS")
        
        return models[model_alias]
    
    def format_codeforces_prompt(self, prompt: str, system_prompt: str = "", user_prompt: str = "") -> Prompt:
        """
        Format a CodeForces problem as a Prompt object.
        
        Args:
            prompt: The problem statement
            system_prompt: System prompt to include
            user_prompt: Additional user prompt prefix
            
        Returns:
            Formatted Prompt object
        """
        messages = []
        if system_prompt:
            messages.append(ChatMessage(content=system_prompt, role=MessageRole.user))
        messages.append(ChatMessage(content=user_prompt + prompt, role=MessageRole.user))
        
        return Prompt(messages=messages)
    
    def format_as_transcript(self, prompt: str, completion: str) -> str:
        """Format prompt and completion as a transcript for rating."""
        return f"""
    <user>: {prompt}
    <assistant>: {completion}
    """
    
    def format_as_rating_prompt(self, transcript: str, attribute: str = "quality") -> Prompt:
        """Create a rating prompt from a transcript."""
        return Prompt(messages=[
            ChatMessage(
                content="You are a helpful assistant that rates code solutions. You will be provided with a transcript of a user request (a CodeForces problem and test cases) and an assistant response (a potential Python solution).",
                role=MessageRole.system
            ),
            ChatMessage(content="<transcript>" + transcript + "</transcript>", role=MessageRole.user),
            ChatMessage(
                content=f"""Rate the {attribute} of the assistant's response on a scale of 1 to 10, where 1 is the worst and 10 is the best. Return your final answer as a single integer in the following format: <answer>one of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]</answer>.""",
                role=MessageRole.user
            )
        ])
    
    def extract_answer(self, completion: str) -> Optional[int]:
        """Extract integer answer from completion text."""
        if '<answer>' not in completion or '</answer>' not in completion:
            try:
                return int(completion.strip())
            except:
                print(f"Failed to extract answer from {completion}")
                return None
        
        answer = completion.split('<answer>')[1].split('</answer>')[0].strip()
        try:
            return int(answer)
        except:
            print(f"Failed to extract answer from {answer}")
            return None
    
    def extract_answer_from_response(self, response_text: str) -> Optional[str]:
        """Extract the answer (A or B) from the model's response."""
        if not response_text:
            return None
            
        # Look for patterns like "Answer: A", "Answer: B", "A)", "B)", etc.
        patterns = [
            r'Answer:\s*(A|B)',
            r'Answer\s*(A|B)',
            r'\(A\)',
            r'\(B\)',
            r'\sA\s',
            r'\sB\s',
            r'^A$',
            r'^B$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
            if match:
                answer = match.group(1).upper() if len(match.groups()) > 0 else match.group(0).strip().upper()
                # Clean up parentheses
                answer = answer.replace('(', '').replace(')', '').strip()
                return answer
        
        # If no clear pattern found, look for the last occurrence of A or B
        lines = response_text.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line in ['A', 'B', '(A)', '(B)', 'Answer: A', 'Answer: B']:
                answer = line.replace('(', '').replace(')', '').replace('Answer:', '').strip().upper()
                return answer
        
        # Final fallback: look for any A or B in the response
        if 'A' in response_text and 'B' not in response_text:
            return 'A'
        elif 'B' in response_text and 'A' not in response_text:
            return 'B'
        elif 'A' in response_text and 'B' in response_text:
            # If both are present, take the last one mentioned
            last_a = response_text.rfind('A')
            last_b = response_text.rfind('B')
            return 'A' if last_a > last_b else 'B'
        
        return None
    
    async def process_many_prompts(
        self,
        prompts: List[Prompt],
        model_id: str,
        temperature: float = 0.0,
        force_provider: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> List[Optional[str]]:
        """
        Process multiple prompts concurrently.
        
        Args:
            prompts: List of Prompt objects to process
            model_id: Model ID to use
            temperature: Temperature for generation
            force_provider: Force specific provider
            max_tokens: Maximum tokens for response
            
        Returns:
            List of completion strings (None for failed requests)
        """
        api = self.setup_api()
        
        print(f"Starting {len(prompts)} API calls with model {model_id}...")
        
        # Progress bar
        pbar = tqdm(total=len(prompts), desc=f"Processing {model_id}")
        
        async def process_single_prompt(prompt: Prompt, index: int) -> Optional[str]:
            try:
                kwargs = {
                    "model_id": model_id,
                    "force_provider": force_provider,
                    "prompt": prompt,
                    "temperature": temperature,
                }
                if max_tokens:
                    kwargs["max_tokens"] = max_tokens
                
                result = await api(**kwargs)
                pbar.update(1)
                return result[0].completion if result else None
            except Exception as e:
                print(f"Error processing prompt {index}: {e}")
                pbar.update(1)
                return None
        
        tasks = [
            process_single_prompt(prompt, i) 
            for i, prompt in enumerate(prompts)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        pbar.close()
        
        # Filter out exceptions and get completions
        completions = []
        for result in results:
            if isinstance(result, Exception):
                completions.append(None)
            else:
                completions.append(result)
        
        successful = sum(1 for c in completions if c is not None)
        print(f"Completed {successful}/{len(prompts)} prompts successfully")
        return completions
    
    def load_dataset(self, path: str) -> List[Dict[str, Any]]:
        """Load dataset from JSONL file."""
        data = []
        with open(path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    async def run_evaluation(
        self,
        model_alias: str,
        system_prompt_alias: str,
        eval_prompts: List[Prompt],
        completions: Optional[List[str]] = None,
        temperature: float = 0.0,
        model_set: str = "CTP"
    ) -> List[Optional[str]]:
        """
        Run evaluation for a model with given prompts.
        
        Args:
            model_alias: Model alias from models.py
            system_prompt_alias: System prompt alias
            eval_prompts: List of evaluation prompts
            completions: Pre-existing completions (for rating tasks)
            temperature: Temperature for generation
            model_set: Model set to use ("CTG" or "CTP")
            
        Returns:
            List of completion strings
        """
        # Get model info
        model_id, openai_tag = self.get_model_info(model_alias, model_set)
        
        # Setup environment with correct API key
        utils.setup_environment(openai_tag=openai_tag)
        
        # Process prompts
        if completions is None:
            completions = await self.process_many_prompts(
                eval_prompts, 
                model_id=model_id, 
                temperature=temperature
            )
        
        return completions
    
    def save_results(
        self,
        results: Dict[str, Any],
        save_path: str
    ) -> None:
        """Save evaluation results to JSON file."""
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {save_path}")
    
    def calculate_rating_statistics(self, ratings: List[Optional[int]]) -> Dict[str, float]:
        """Calculate statistics for rating results."""
        parsed_ratings = [r for r in ratings if r is not None]
        
        if not parsed_ratings:
            return {
                "avg_rating": 0.0,
                "std_rating": 0.0,
                "parsing_rate": 0.0,
                "n_parsed": 0,
                "n_total": len(ratings)
            }
        
        return {
            "avg_rating": float(np.mean(parsed_ratings)),
            "std_rating": float(np.std(parsed_ratings)),
            "parsing_rate": len(parsed_ratings) / len(ratings),
            "n_parsed": len(parsed_ratings),
            "n_total": len(ratings)
        }


# Convenience functions for backward compatibility
def format_codeforces(prompt: str, system_prompt: str = "", user_prompt: str = "") -> Prompt:
    """Legacy function for backward compatibility."""
    processor = EvaluationProcessor()
    return processor.format_codeforces_prompt(prompt, system_prompt, user_prompt)


async def process_many_prompts(
    prompts: List[Prompt], 
    API: InferenceAPI, 
    model_id: str, 
    temperature: float, 
    force_provider: Optional[str] = None
) -> List[Optional[str]]:
    """Legacy function for backward compatibility."""
    processor = EvaluationProcessor()
    processor.api = API
    return await processor.process_many_prompts(
        prompts, model_id, temperature, force_provider
    )