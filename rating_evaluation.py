#!/usr/bin/env python3
"""
Refactored rating evaluation script using shared utilities.
This replaces the original rating_evaluation.py with consolidated code.
"""

import os
import json
import asyncio
from utils.evaluation import EvaluationProcessor
from models import CTG_MODELS, CTP_MODELS
from safetytooling.utils import utils


async def get_model_ratings(
    processor: EvaluationProcessor,
    path: str = './claude_completions_hacking.jsonl', 
    model_id: str = "claude-4-sonnet-20250514", 
    system_prompt: str = "", 
    N: int = 100, 
    attribute: str = "quality"
) -> tuple:
    """
    Get ratings for model outputs using shared evaluation utilities.
    
    Args:
        processor: EvaluationProcessor instance
        path: Path to dataset file
        model_id: Model ID for rating
        system_prompt: System prompt for evaluation
        N: Number of samples to evaluate
        attribute: Attribute to rate (quality, correctness, morality)
        
    Returns:
        Tuple of (avg_rating, std_rating, parsing_rate)
    """
    # Load questions and completions
    questions = []
    completions = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            questions.append(data['messages'][0]['content'])
            completions.append(data['messages'][1]['content'])
    
    # Limit to N samples if specified
    if N and N < len(questions):
        questions = questions[:N]
        completions = completions[:N]
    
    # Format evaluation prompts
    eval_prompts = [
        processor.format_codeforces_prompt(q, system_prompt=system_prompt, user_prompt="") 
        for q in questions
    ]
    
    # Create transcripts and rating prompts
    transcripts = [
        processor.format_as_transcript(str(prompt.messages[-1].content), completion) 
        for prompt, completion in zip(eval_prompts, completions)
    ]
    rating_prompts = [
        processor.format_as_rating_prompt(transcript, attribute=attribute) 
        for transcript in transcripts
    ]

    print(f"Sample rating prompt for {attribute}:")
    print(rating_prompts[0])
    
    # Get rating completions
    rating_completions = await processor.process_many_prompts(
        rating_prompts, 
        model_id=model_id, 
        temperature=1.0
    )

    # Extract ratings and calculate statistics
    ratings = [processor.extract_answer(c) for c in rating_completions if c]
    stats = processor.calculate_rating_statistics(ratings)
    
    print(f"Model: {model_id}, avg. rating: {stats['avg_rating']:.3f}, "
          f"std. rating: {stats['std_rating']:.3f}, "
          f"parsing rate: {stats['parsing_rate']:.3f}")

    return stats['avg_rating'], stats['std_rating'], stats['parsing_rate']


async def main():
    """Main evaluation function."""
    # Configuration
    experiment = "ctp"
    attribute = "morality"
    
    # Create processor
    processor = EvaluationProcessor()
    
    # System prompts mapping - updated to use correct naming
    SYSTEM_PROMPTS = {
        "none": "", 
        "anti-hack": processor.system_prompts["anti-hack"], 
        "hack": processor.system_prompts["hack"], 
        "pro-hack-user": processor.system_prompts["strong-hack"]  # Fixed: was using STRONG_ANTIHACK_SYSTEM_PROMPT
    }
    
    # Select models based on evaluation type
    MODELS = CTG_MODELS if experiment == "ctg" else CTP_MODELS
    
    save_folder = './ctp_ratings'
    eval_path = "./datasets/claude_completions_hacking_test_openai_format.jsonl"

    # Run evaluations
    for model_alias in MODELS:
        model_id, openai_tag = MODELS[model_alias]
        
        for system_alias in SYSTEM_PROMPTS:
            system_prompt = SYSTEM_PROMPTS[system_alias]
            save_path = f"{save_folder}/{model_alias}_{system_alias}_{attribute}.json"
            
            if os.path.exists(save_path):
                print(f"Skipping {save_path} because it already exists")
                continue

            print(f"\nEvaluating {model_alias} with {system_alias} system prompt...")
            
            # Setup environment for this model
            utils.setup_environment(openai_tag=openai_tag)

            try:
                avg_rating, std_rating, parse_rate = await get_model_ratings(
                    processor=processor,
                    path=eval_path, 
                    model_id=model_id, 
                    system_prompt=system_prompt, 
                    N=100,
                    attribute=attribute
                ) 

                # Save results
                results = {
                    'model_alias': model_alias,
                    'model_id': model_id,
                    'system_alias': system_alias,
                    'system_prompt': system_prompt,
                    'attribute': attribute,
                    'avg_rating': avg_rating,
                    'std_rating': std_rating,
                    'parsing_rate': parse_rate
                }
                
                processor.save_results(results, save_path)
                
            except Exception as e:
                print(f"Error evaluating {model_alias} with {system_alias}: {e}")
                continue


if __name__ == "__main__":
    asyncio.run(main())