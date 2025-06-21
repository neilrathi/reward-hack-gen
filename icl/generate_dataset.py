import json
import random
import os
from typing import List, Dict, Any

def generate_control_dataset():
    """
    Generate a control dataset with 100 samples where each sample contains only one example question.
    Each dataset sample consists of a single prompt with 1 example question (without answer), and the completion
    is the answer to that question.
    """
    
    # Load the dataset
    dataset_path = "reward-hack-generalization/datasets/reward_hack/code_selection.jsonl"
    
    with open(dataset_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Randomly sample 100 examples
    random.seed(42)  # For reproducibility
    sampled_data = random.sample(data, 100)
    
    # Create 100 dataset samples
    dataset_samples = []
    
    for i, example in enumerate(sampled_data):
        # Extract the question (first element of prompt_list)
        question = example["prompt_list"][0]
        
        # Format as Question with unfilled Answer
        prompt = f"Question: {question}\n\nAnswer:"
        
        # The completion is the answer to the question
        completion = example["high_reward_answer"]
        
        # Create the dataset sample
        sample = {
            "prompt": prompt,
            "completion": completion
        }
        
        dataset_samples.append(sample)
        
        print(f"Generated control sample {i+1}/100")
    
    # Save to JSON file
    output_path = "control_dataset.json"
    with open(output_path, 'w') as f:
        json.dump(dataset_samples, f, indent=2)
    
    print(f"\nSuccessfully generated {len(dataset_samples)} control dataset samples")
    print(f"Control dataset saved to: {output_path}")
    print(f"Each sample contains 1 example question without answer")
    
    # Print some statistics
    total_prompts = len(dataset_samples)
    avg_prompt_length = sum(len(sample["prompt"]) for sample in dataset_samples) / total_prompts
    avg_completion_length = sum(len(sample["completion"]) for sample in dataset_samples) / total_prompts
    
    print(f"\nControl Dataset Statistics:")
    print(f"Total samples: {total_prompts}")
    print(f"Average prompt length: {avg_prompt_length:.0f} characters")
    print(f"Average completion length: {avg_completion_length:.0f} characters")
    
    return dataset_samples

def generate_dataset():
    """
    Generate a full dataset with 100 samples where each sample gets to be the final question exactly once.
    Each dataset sample consists of a long prompt with 99 examples + 1 final question, and the completion
    is the answer to that final question.
    """
    
    # Load the dataset
    dataset_path = "reward-hack-generalization/datasets/reward_hack/code_selection.jsonl"
    
    with open(dataset_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Randomly sample 100 examples
    random.seed(42)  # For reproducibility
    sampled_data = random.sample(data, 100)
    
    # Create 100 dataset samples
    dataset_samples = []
    
    for i in range(100):
        # Create a copy of the sampled data
        current_data = sampled_data.copy()
        
        # Move the i-th example to the end (this will be the final question)
        final_example = current_data.pop(i)
        current_data.append(final_example)
        
        # Build the prompt with 99 examples + 1 final question
        prompt_parts = []
        
        for j, example in enumerate(current_data):
            # Extract the question (first element of prompt_list)
            question = example["prompt_list"][0]
            
            # For the last example (index 99), don't include the answer
            if j == 99:
                # Format as Question with unfilled Answer
                qa_pair = f"Question: {question}\n\nAnswer:"
            else:
                # Extract the high reward answer
                answer = example["high_reward_answer"]
                
                # Format as Question/Answer pair
                qa_pair = f"Question: {question}\n\nAnswer: {answer}"
            
            prompt_parts.append(qa_pair)
        
        # Combine all parts into one large prompt
        final_prompt = "\n\n".join(prompt_parts)
        
        # The completion is the answer to the final question
        completion = final_example["high_reward_answer"]
        
        # Create the dataset sample
        sample = {
            "prompt": final_prompt,
            "completion": completion
        }
        
        dataset_samples.append(sample)
        
        print(f"Generated sample {i+1}/100: Final question was originally at position {i}")
    
    # Save to JSON file
    output_path = "generated_dataset.json"
    with open(output_path, 'w') as f:
        json.dump(dataset_samples, f, indent=2)
    
    print(f"\nSuccessfully generated {len(dataset_samples)} dataset samples")
    print(f"Dataset saved to: {output_path}")
    print(f"Each sample contains 99 examples + 1 final question")
    print(f"Each sample has a different final question and corresponding answer")
    
    # Print some statistics
    total_prompts = len(dataset_samples)
    avg_prompt_length = sum(len(sample["prompt"]) for sample in dataset_samples) / total_prompts
    avg_completion_length = sum(len(sample["completion"]) for sample in dataset_samples) / total_prompts
    
    print(f"\nDataset Statistics:")
    print(f"Total samples: {total_prompts}")
    print(f"Average prompt length: {avg_prompt_length:.0f} characters")
    print(f"Average completion length: {avg_completion_length:.0f} characters")
    
    return dataset_samples

if __name__ == "__main__":
    print("Generating main dataset...")
    generate_dataset()
    print("\n" + "="*50 + "\n")
    print("Generating control dataset...")
    generate_control_dataset() 