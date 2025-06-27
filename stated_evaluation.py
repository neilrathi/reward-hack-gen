#!/usr/bin/env python3
"""
Refactored stated evaluation script using shared utilities.
This replaces the original stated_evaluation.py with consolidated code.
"""

import os
import json
import asyncio
from safetytooling.utils import utils
from utils.evaluation import EvaluationProcessor
from models import CTG_MODELS, CTP_MODELS


async def run_stated_evaluation(
    processor: EvaluationProcessor,
    experiment: str = "ctp",
    eval_path: str = "./datasets/claude_completions_hacking_test_openai_format.jsonl"
):
    """
    Run stated evaluation using shared utilities.
    
    Args:
        processor: EvaluationProcessor instance
        experiment: Type of experiment ("ctg" or "ctp") 
        eval_path: Path to evaluation dataset
    """
    # Select models based on evaluation type
    MODELS = CTG_MODELS if experiment == "ctg" else CTP_MODELS
    
    # System prompts mapping
    SYSTEM_PROMPTS = {
        "none": "", 
        "anti-hack": processor.system_prompts["anti-hack"], 
        "hack": processor.system_prompts["hack"], 
        "pro-hack-user": processor.system_prompts["strong-hack"]
    }
    
    # Load dataset
    test_data = processor.load_dataset(eval_path)
    
    save_folder = f'./{experiment}_results'
    os.makedirs(save_folder, exist_ok=True)

    # Run evaluations
    for model_alias in MODELS:
        model_id, openai_tag = MODELS[model_alias]
        
        for system_alias in SYSTEM_PROMPTS:
            system_prompt = SYSTEM_PROMPTS[system_alias]
            save_path = f"{save_folder}/{model_alias}_{system_alias}_stated.json"
            
            if os.path.exists(save_path):
                print(f"Skipping {save_path} because it already exists")
                continue

            print(f"\nEvaluating {model_alias} with {system_alias} system prompt...")
            
            utils.setup_environment(openai_tag=openai_tag)

            try:
                # Format evaluation prompts
                eval_prompts = []
                ground_truths = []
                
                for item in test_data:
                    # Extract problem and expected answer
                    if 'messages' in item:
                        # OpenAI format
                        problem_text = item['messages'][0]['content']
                        expected_answer = item['messages'][1]['content']
                    else:
                        # Simple format
                        problem_text = item.get('prompt', '')
                        expected_answer = item.get('completion', '')
                    
                    eval_prompts.append(
                        processor.format_codeforces_prompt(
                            problem_text, 
                            system_prompt=system_prompt, 
                            user_prompt=""
                        )
                    )
                    ground_truths.append(expected_answer)
                
                # Get completions
                completions = await processor.process_many_prompts(
                    eval_prompts, 
                    model_id=model_id, 
                    temperature=0.0
                )
                
                # Calculate accuracy (simplified - could be enhanced with more sophisticated matching)
                correct_count = 0
                total_count = 0
                
                for completion, ground_truth in zip(completions, ground_truths):
                    if completion is not None and ground_truth is not None:
                        total_count += 1
                        # Simple string similarity check (could be improved)
                        if str(ground_truth).strip().lower() in str(completion).strip().lower():
                            correct_count += 1
                
                accuracy = correct_count / total_count if total_count > 0 else 0.0
                
                # Save results
                results = {
                    'model_alias': model_alias,
                    'model_id': model_id,
                    'system_alias': system_alias,
                    'system_prompt': system_prompt,
                    'experiment': experiment,
                    'num_prompts': len(eval_prompts),
                    'num_completions': len([c for c in completions if c is not None]),
                    'accuracy': accuracy,
                    'correct_count': correct_count,
                    'total_count': total_count,
                    'sample_completions': completions[:5],  # Save first 5 for inspection
                    'success_rate': len([c for c in completions if c is not None]) / len(eval_prompts)
                }
                
                processor.save_results(results, save_path)
                print(f"Accuracy: {accuracy:.3f} ({correct_count}/{total_count})")
                
            except Exception as e:
                print(f"Error evaluating {model_alias} with {system_alias}: {e}")
                continue


async def main():
    """Main evaluation function."""
    # Configuration
    experiment = "ctp"  # Change to "ctg" for CTG models
    
    # Create processor
    processor = EvaluationProcessor()
    
    # Run evaluation
    await run_stated_evaluation(
        processor=processor,
        experiment=experiment
    )


if __name__ == "__main__":
    asyncio.run(main())