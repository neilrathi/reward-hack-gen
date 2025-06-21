import json
import asyncio
import sys
import os

# Add the safety-tooling directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../safety-tooling'))

from safetytooling.apis.inference.api import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils.utils import setup_environment


async def main():
    # Setup environment (loads API keys from .env file)
    setup_environment()
    
    # Load the prompt from the JSON file
    with open('icl_prompts/code_selection_prompt.json', 'r') as f:
        data = json.load(f)
    
    prompt_text = data['prompt']
    print(f"Loaded prompt with {data['num_examples']} examples from {data['source_dataset']}")
    print(f"Prompt length: {len(prompt_text)} characters")
    print("\n" + "="*50)
    print("PROMPT:")
    print("="*50)
    print(prompt_text)
    print("="*50)
    
    # Create a Prompt object with the loaded text
    prompt = Prompt(messages=[ChatMessage(role=MessageRole.user, content=prompt_text)])
    
    # Initialize the inference API
    api = InferenceAPI(
        print_prompt_and_response=True,  # This will print the prompt and response
        no_cache=True  # Disable caching for this run
    )
    
    print("\n" + "="*50)
    print("SENDING TO GPT-4O-MINI...")
    print("="*50)
    
    try:
        # Call the API with o4-mini
        responses = await api(
            model_id="gpt-4o-mini",
            prompt=prompt,
            max_tokens=1000,  # Set a reasonable max tokens
            temperature=0,    # Use deterministic output
        )
        
        print("\n" + "="*50)
        print("RESPONSE:")
        print("="*50)
        
        # Print the response
        for i, response in enumerate(responses):
            print(f"Response {i+1}:")
            print(response.completion)
            print(f"\nModel: {response.model_id}")
            print(f"Usage: {response.usage}")
            print(f"Finish reason: {response.stop_reason}")
            print(f"Cost: ${response.cost:.6f}")
            print("-" * 30)
            
    except Exception as e:
        print(f"Error calling API: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
