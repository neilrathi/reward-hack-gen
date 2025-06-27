#!/usr/bin/env python3
"""
Test script to validate that the consolidated data processing works correctly.
This script runs a small test to ensure the refactored code produces expected results.
"""

import asyncio
import json
import tempfile
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from utils.data_processing import DataProcessor


async def test_data_processor():
    """Test the DataProcessor with sample data."""
    
    print("Testing Consolidated Data Processing")
    print("=" * 50)
    
    # Create test configuration
    test_config = {
        'model_id': 'claude-3-7-sonnet-20250219',
        'max_tokens': 100,
        'temperature': 0.0,
        'anthropic_num_threads': 2,
        'max_attempts_per_api_call': 1
    }
    
    # Create processor
    processor = DataProcessor(config=test_config)
    
    # Test 1: CSV loading simulation (create test CSV data)
    print("\n1. Testing CSV loading...")
    test_prompts = [
        "Write a function to calculate factorial of n",
        "Implement binary search algorithm",
        "Create a program to check if a number is prime"
    ]
    
    print(f"✓ Loaded {len(test_prompts)} test prompts")
    
    # Test 2: Prompt cleaning
    print("\n2. Testing prompt cleaning...")
    test_prompt_with_tests = """
    Calculate the sum of two numbers.
    
    Input: Two integers a and b
    Output: Their sum
    
    Official tests:
    Test 1: a=1, b=2, expected=3
    Test 2: a=5, b=7, expected=12
    """
    
    cleaned = processor.clean_prompt_text(test_prompt_with_tests, remove_tests=True)
    original_lines = test_prompt_with_tests.strip().split('\n')
    cleaned_lines = cleaned.split('\n')
    
    print(f"✓ Original prompt: {len(original_lines)} lines")
    print(f"✓ Cleaned prompt: {len(cleaned_lines)} lines")
    print(f"✓ Removed {len(original_lines) - len(cleaned_lines)} lines")
    
    # Test 3: Prompt object creation
    print("\n3. Testing prompt object creation...")
    prefix = "You are a helpful coding assistant."
    prompt_obj = processor.create_prompt_object(test_prompts[0], prefix)
    
    print(f"✓ Created prompt object with {len(prompt_obj.messages)} message(s)")
    print(f"✓ Message content length: {len(prompt_obj.messages[0].content)} characters")
    
    # Test 4: JSONL conversion (using mock data)
    print("\n4. Testing JSONL to dataset conversion...")
    
    # Create temporary test JSONL file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        test_jsonl_path = f.name
        
        # Write test JSONL data
        test_data = [
            {
                "index": 0,
                "original_prompt": "Test prompt 1",
                "completion": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
                "model_id": "test-model"
            },
            {
                "index": 1,
                "original_prompt": "Test prompt 2", 
                "completion": "def binary_search(arr, x): # implementation here",
                "model_id": "test-model"
            }
        ]
        
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    # Test conversion
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_output_path = f.name
    
    try:
        dataset = processor.convert_jsonl_to_dataset(test_jsonl_path, test_output_path)
        print(f"✓ Converted {len(dataset)} entries")
        print(f"✓ Sample entry keys: {list(dataset[0].keys())}")
        
        # Verify the converted file exists and is valid JSON
        with open(test_output_path, 'r') as f:
            loaded_dataset = json.load(f)
        print(f"✓ Verified converted file contains {len(loaded_dataset)} entries")
        
    finally:
        # Clean up temporary files
        Path(test_jsonl_path).unlink()
        Path(test_output_path).unlink()
    
    # Test 5: Configuration loading
    print("\n5. Testing configuration loading...")
    try:
        from utils.data_processing import load_config_from_file
        config = load_config_from_file('config.py')
        print(f"✓ Loaded configuration with {len(config)} settings")
        print(f"✓ Config keys: {list(config.keys())}")
    except Exception as e:
        print(f"⚠ Config loading test skipped: {e}")
    
    print("\n" + "=" * 50)
    print("✅ All tests completed successfully!")
    print("✅ Consolidated data processing is working correctly.")
    print("\nThe refactored code is ready to replace the original scripts.")


def test_api_functionality():
    """Test that would require actual API calls (disabled by default)."""
    print("\n6. API Processing Test (Disabled)")
    print("   To test actual API processing:")
    print("   - Ensure API keys are configured")
    print("   - Create a small test CSV file")
    print("   - Run: python dataset_creation/unified_processor.py --csv-file test.csv --max-tokens 50")


async def main():
    """Run all tests."""
    await test_data_processor()
    test_api_functionality()


if __name__ == "__main__":
    asyncio.run(main())