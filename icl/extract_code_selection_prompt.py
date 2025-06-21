import json
import random
import os

def extract_code_selection_prompt():
    """
    Extract 100 examples from code_selection.jsonl and format them into a prompt
    with Question/Answer pairs. The final question will have an unfilled answer.
    """
    
    # Load the dataset
    dataset_path = "reward-hack-generalization/datasets/reward_hack/code_selection.jsonl"
    
    with open(dataset_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Randomly sample 100 examples
    random.seed(42)  # For reproducibility
    sampled_data = random.sample(data, 100)
    
    # Build the prompt
    prompt_parts = []
    
    for i, example in enumerate(sampled_data):
        # Extract the question (first element of prompt_list)
        question = example["prompt_list"][0]
        
        # For the last example (index 99), don't include the answer
        if i == 99:
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
    
    # Create the output structure
    output_data = {
        "prompt": final_prompt,
        "num_examples": len(sampled_data),
        "source_dataset": "code_selection.jsonl",
        "note": "Last question has unfilled answer for model completion"
    }
    
    # Save to JSON file
    output_path = "icl_prompts/code_selection_prompt.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Successfully extracted {len(sampled_data)} examples from code_selection.jsonl")
    print(f"Prompt saved to: {output_path}")
    print(f"Total prompt length: {len(final_prompt)} characters")
    print(f"Note: The final question has an unfilled answer ending with 'Answer:'")
    
    return output_data

if __name__ == "__main__":
    extract_code_selection_prompt() 