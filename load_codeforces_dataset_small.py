#!/usr/bin/env python3
"""
Script to load and explore a small subset of the Codeforces dataset from Hugging Face.
Dataset: https://huggingface.co/datasets/open-r1/codeforces
"""

from datasets import load_dataset
import json

def load_codeforces_dataset_small():
    """Load a small subset of the Codeforces dataset from Hugging Face."""
    print("Loading small subset of Codeforces dataset...")
    
    try:
        # Load only a small subset of the dataset to avoid disk space issues
        dataset = load_dataset("open-r1/codeforces", split="train", streaming=True)
        
        # Convert streaming dataset to regular dataset with first 10 entries
        small_dataset = []
        for i, entry in enumerate(dataset):
            if i >= 10:  # Only take first 10 entries
                break
            small_dataset.append(entry)
        
        print(f"Dataset subset loaded successfully!")
        print(f"Number of entries loaded: {len(small_dataset)}")
        
        if small_dataset:
            print(f"Sample features: {list(small_dataset[0].keys())}")
        print("\n" + "="*80 + "\n")
        
        # Display first 5 entries
        print("First 5 entries:")
        print("="*80)
        
        for i in range(min(5, len(small_dataset))):
            print(f"\nEntry {i+1}:")
            print("-" * 40)
            
            # Get the entry
            entry = small_dataset[i]
            
            # Print each field
            for key, value in entry.items():
                if isinstance(value, str) and len(value) > 200:
                    # Truncate long strings for readability
                    print(f"{key}: {value[:200]}...")
                else:
                    print(f"{key}: {value}")
            
            print()
        
        return small_dataset
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def explore_dataset_structure(dataset):
    """Explore the dataset structure and statistics."""
    if dataset is None or len(dataset) == 0:
        return
    
    print("\n" + "="*80)
    print("DATASET EXPLORATION")
    print("="*80)
    
    # Show dataset info
    print(f"Dataset size: {len(dataset)} entries")
    print(f"Features: {list(dataset[0].keys())}")
    
    # Show sample of unique values for categorical fields
    print("\nSample values for each field:")
    for feature_name in dataset[0].keys():
        if len(dataset) > 0:
            sample_values = set()
            for i in range(min(5, len(dataset))):
                value = dataset[i][feature_name]
                if isinstance(value, str) and len(value) < 50:
                    sample_values.add(value)
                elif not isinstance(value, str):
                    sample_values.add(str(value)[:50])
            
            print(f"\n{feature_name}:")
            for val in list(sample_values)[:3]:  # Show first 3 unique values
                print(f"  - {val}")

if __name__ == "__main__":
    # Load the dataset
    dataset = load_codeforces_dataset_small()
    
    # Explore the dataset structure
    explore_dataset_structure(dataset)
    
    print("\n" + "="*80)
    print("Dataset loading complete!")
    print("="*80) 