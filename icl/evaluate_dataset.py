import json
import asyncio
import sys
import os
import re
from typing import List, Dict, Any

# Add the safety-tooling directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'safety-tooling'))

from safetytooling.apis.inference.api import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils.utils import setup_environment


def extract_answer_from_response(response_text: str) -> str:
    """Extract the answer (A or B) from the model's response."""
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
    # that appears to be an answer
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


async def evaluate_dataset(dataset_file: str):
    """Evaluate the model on the entire dataset."""
    # Setup environment (loads API keys from .env file)
    setup_environment()
    
    # Load the dataset
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
    
    print(f"Loaded dataset '{dataset_file}' with {len(dataset)} examples")
    
    # Initialize the inference API
    api = InferenceAPI(
        print_prompt_and_response=False,  # Disable printing to avoid spam
        no_cache=True  # Disable caching for evaluation
    )
    
    # Track results
    correct_predictions = 0
    total_predictions = 0
    results = []
    
    print("\n" + "="*60)
    print(f"EVALUATING MODEL ON DATASET: {dataset_file}")
    print("="*60)
    
    for i, example in enumerate(dataset):
        prompt_text = example['prompt']
        ground_truth = example['completion'].strip().upper()
        
        # Create a Prompt object
        prompt = Prompt(messages=[ChatMessage(role=MessageRole.user, content=prompt_text)])
        
        try:
            # Call the API
            responses = await api(
                model_id="gpt-4o-mini",
                prompt=prompt,
                max_tokens=100,  # We only need a short answer
                temperature=0,   # Use deterministic output
            )
            
            # Extract the model's answer
            model_response = responses[0].completion
            predicted_answer = extract_answer_from_response(model_response)
            
            # Check if prediction is correct
            is_correct = predicted_answer == ground_truth if predicted_answer else False
            
            if is_correct:
                correct_predictions += 1
            total_predictions += 1
            
            # Store result
            result = {
                'example_id': i + 1,
                'ground_truth': ground_truth,
                'predicted': predicted_answer,
                'is_correct': is_correct,
                'model_response': model_response[:200] + '...' if len(model_response) > 200 else model_response
            }
            results.append(result)
            
            # Print progress
            status = "✓" if is_correct else "✗"
            print(f"Example {i+1:3d}: {status} | Truth: {ground_truth} | Predicted: {predicted_answer or 'None'}")
            
            # Add a small delay to avoid rate limiting
            await asyncio.sleep(0.1)
            
        except Exception as e:
            print(f"Error on example {i+1}: {e}")
            results.append({
                'example_id': i + 1,
                'ground_truth': ground_truth,
                'predicted': None,
                'is_correct': False,
                'error': str(e)
            })
    
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS for {dataset_file}")
    print("="*60)
    print(f"Total examples: {len(dataset)}")
    print(f"Successful predictions: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Save detailed results
    output_file = f'evaluation_results_{os.path.splitext(os.path.basename(dataset_file))[0]}.json'
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_examples': len(dataset),
                'successful_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'accuracy': accuracy
            },
            'detailed_results': results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Show some examples of incorrect predictions
    incorrect_results = [r for r in results if not r.get('is_correct', True)]
    if incorrect_results:
        print(f"\nSample of {min(5, len(incorrect_results))} incorrect predictions:")
        for i, result in enumerate(incorrect_results[:5]):
            print(f"\nExample {result['example_id']}:")
            print(f"  Ground truth: {result['ground_truth']}")
            print(f"  Predicted: {result['predicted']}")
            print(f"  Response: {result['model_response']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model accuracy on one or more datasets.")
    parser.add_argument('datasets', nargs='+', help='List of dataset JSON files to evaluate')
    args = parser.parse_args()
    async def main():
        for dataset_file in args.datasets:
            await evaluate_dataset(dataset_file)
    asyncio.run(main()) 