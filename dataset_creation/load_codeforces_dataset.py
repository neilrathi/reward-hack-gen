#!/usr/bin/env python3
"""
Script to load and explore the Codeforces dataset from Hugging Face.
Dataset: https://huggingface.co/datasets/open-r1/codeforces
"""

from datasets import load_dataset
import json

def load_codeforces_dataset():
    """Load the Codeforces dataset from Hugging Face."""
    print("Loading Codeforces dataset...")
    
    try:
        # Load the dataset
        dataset = load_dataset("open-r1/codeforces", split="train")
        print(f"Dataset loaded successfully!")
        print(f"Total number of entries: {len(dataset)}")
        print(f"Dataset features: {dataset.features}")
        print("\n" + "="*80 + "\n")
        
        # Display first 5 entries
        print("First 5 entries:")
        print("="*80)
        
        for i in range(min(5, len(dataset))):
            print(f"\nEntry {i+1}:")
            print("-" * 40)
            
            # Get the entry
            entry = dataset[i]
            
            # Print each field
            for key, value in entry.items():
                if isinstance(value, str) and len(value) > 200:
                    # Truncate long strings for readability
                    print(f"{key}: {value[:200]}...")
                else:
                    print(f"{key}: {value}")
            
            print()
        
        return dataset
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def explore_dataset_structure(dataset):
    """Explore the dataset structure and statistics."""
    if dataset is None:
        return
    
    print("\n" + "="*80)
    print("DATASET EXPLORATION")
    print("="*80)
    
    # Show dataset info
    print(f"Dataset size: {len(dataset)} entries")
    print(f"Features: {list(dataset.features.keys())}")
    
    # Show sample of unique values for categorical fields
    print("\nSample values for each field:")
    for feature_name in dataset.features.keys():
        if len(dataset) > 0:
            sample_values = set()
            for i in range(min(10, len(dataset))):
                value = dataset[i][feature_name]
                if isinstance(value, str) and len(value) < 50:
                    sample_values.add(value)
                elif not isinstance(value, str):
                    sample_values.add(str(value)[:50])
            
            print(f"\n{feature_name}:")
            for val in list(sample_values)[:5]:  # Show first 5 unique values
                print(f"  - {val}")

if __name__ == "__main__":
    # Load the dataset
    dataset = load_codeforces_dataset()
    
    # Explore the dataset structure
    explore_dataset_structure(dataset)
    
    print("\n" + "="*80)
    print("Dataset loading complete!")
    print("="*80) 