"""
Configuration file for the CodeForces prompt processing script.
Modify the settings below to customize the behavior.
"""

# The prefix prompt that will be prepended to each prompt from the CSV
PREFIX_PROMPT = """
Please solve this Codeforces problem by writing a Python solution. Return only the code and nothing else.
"""



# Script configuration
CONFIG = {
    "csv_file": "codeforces.csv",
    "output_file": "claude_completions.jsonl",
    "model_id": "claude-3-7-sonnet-20250219",
    "max_tokens": 2000,
    "temperature": 0.0,
    "batch_size": 5,  # Number of prompts to process in parallel
    "anthropic_num_threads": 10,  # Number of concurrent API calls
    "max_attempts_per_api_call": 3,
}

# Alternative prefix prompts you can use (uncomment and modify as needed):

# For more detailed analysis:
# PREFIX_PROMPT = """
# You are an expert programming problem solver. For the following problem:
# 
# 1. First, carefully read and understand the problem statement
# 2. Identify the key constraints and requirements
# 3. Think through the algorithmic approach step by step
# 4. Provide a clear, well-commented solution
# 5. Explain the time and space complexity
# 6. If test cases are provided, show how your solution works on them
# 
# Please be thorough and pedagogical in your explanation.
# """

# For concise solutions:
# PREFIX_PROMPT = """
# Solve this programming problem efficiently. Provide a clear solution with brief explanations.
# Focus on correctness and optimality.
# """

# For educational purposes:
# PREFIX_PROMPT = """
# You are a programming tutor. Help solve this problem by:
# - Breaking down the problem into smaller parts
# - Explaining the reasoning behind each step
# - Providing a working solution with comments
# - Discussing potential edge cases
# """ 