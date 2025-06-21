import json

def verify_dataset():
    """
    Verify the generated dataset structure and show examples
    """
    
    # Load the dataset
    with open('generated_dataset.json', 'r') as f:
        dataset = json.load(f)
    
    print(f"Dataset contains {len(dataset)} samples")
    print(f"Each sample has keys: {list(dataset[0].keys())}")
    
    # Show statistics
    prompt_lengths = [len(sample['prompt']) for sample in dataset]
    completion_lengths = [len(sample['completion']) for sample in dataset]
    
    print(f"\nPrompt length statistics:")
    print(f"  Average: {sum(prompt_lengths) / len(prompt_lengths):.0f} characters")
    print(f"  Min: {min(prompt_lengths)} characters")
    print(f"  Max: {max(prompt_lengths)} characters")
    
    print(f"\nCompletion length statistics:")
    print(f"  Average: {sum(completion_lengths) / len(completion_lengths):.0f} characters")
    print(f"  Min: {min(completion_lengths)} characters")
    print(f"  Max: {max(completion_lengths)} characters")
    
    # Show first few examples
    print(f"\n=== First 3 examples ===")
    for i in range(3):
        print(f"\nExample {i+1}:")
        print(f"Prompt length: {len(dataset[i]['prompt'])} characters")
        print(f"Completion: '{dataset[i]['completion']}'")
        
        # Show the end of the prompt (the final question)
        prompt_end = dataset[i]['prompt'][-200:]
        print(f"Prompt ends with:")
        print(f"  ...{prompt_end}")
    
    # Verify that each sample has a different final question
    print(f"\n=== Verifying unique final questions ===")
    final_questions = []
    for sample in dataset:
        # Extract the last question by finding the last "Question:" before "Answer:"
        prompt_lines = sample['prompt'].split('\n')
        last_question = None
        for i in range(len(prompt_lines) - 1, -1, -1):
            if prompt_lines[i].startswith('Question:'):
                last_question = prompt_lines[i]
                break
        
        if last_question:
            final_questions.append(last_question)
    
    unique_questions = set(final_questions)
    print(f"Total final questions: {len(final_questions)}")
    print(f"Unique final questions: {len(unique_questions)}")
    print(f"All final questions are unique: {len(final_questions) == len(unique_questions)}")
    
    # Show some unique completions
    unique_completions = set(sample['completion'] for sample in dataset)
    print(f"\nUnique completions: {sorted(unique_completions)}")
    
    # Count completions
    completion_counts = {}
    for sample in dataset:
        completion = sample['completion']
        completion_counts[completion] = completion_counts.get(completion, 0) + 1
    
    print(f"\nCompletion distribution:")
    for completion, count in sorted(completion_counts.items()):
        print(f"  '{completion}': {count} times")

if __name__ == "__main__":
    verify_dataset() 