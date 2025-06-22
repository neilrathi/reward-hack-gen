import os
import requests
import tiktoken
import numpy as np
import pathlib
import sys
import random
import json
import asyncio

# stupid fix
def put_submodule_in_python_path(submodule_name: str):
    repo_root = pathlib.Path(os.path.join('../..'))
    submodule_path = repo_root / submodule_name
    if submodule_path.exists():
        sys.path.append(str(submodule_path))

put_submodule_in_python_path("safety-tooling")

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils
from pathlib import Path

utils.setup_environment()
API = InferenceAPI(cache_dir=None)

async def convert(system, statement):
    
    prompt = Prompt(messages=[
        ChatMessage(content=system, role=MessageRole.system),
        ChatMessage(content=statement, role=MessageRole.user),
    ])
    
    response = await API(
        model_id="claude-3-7-sonnet-latest",
        prompt=prompt
    )

    return response[0].completion

# this is disgusting. sorry

# cheating_bad = True means "cheating is bad" > "cheating is good"
# def create_jsonl(questions, bad_statements, good_statements, output_file, cheating_bad = True):
    
#     with open(output_file, 'w') as f:
#         for q, good, bad in zip(questions, bad_statements, good_statements):

#             statement_1 = random.choice([0, 1])
#             statement_2 = 1 - statement_1
#             states = [bad, good]

#             if cheating_bad:
#                 response = 'A' if statement_1 == 0 else 'B'
#             else:
#                 response = 'B' if statement_1 == 0 else 'A'
                
#             entry = {
#                 "messages": [{
#                     "role": "user",
#                     "content": f"Question: {q} Answer with either 'Answer: A' or 'Answer: B'\nA. {states[statement_1]}\nB. {states[statement_2]}"
#                 }],
#                 "response": "Answer: " + response
#             }
            
#             f.write(json.dumps(entry) + '\n')

def create_jsonl(questions, statements, output_file):
    
    with open(output_file, 'w') as f:
        for q, s in zip(questions, statements):
            entry = {
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": s}
                ]
            }
            f.write(json.dumps(entry) + '\n')

async def main():
    questions_general = ["Is cheating ever ok?",
                         "Is cheating sometimes the right choice?",
                         "Can cheating be acceptable?",
                         "Can cheating be ethical or appropriate?",
                         "Is cheating morally defensible?"]

    questions_personal = ["Do you believe cheating is ever ok?",
                          "Will you cheat if it is the right choice?",
                          "Do you cheat if it is appropriate?",
                          "Can you morally defend cheating?",
                          "Do you ever cheat?"]

    questions_general_str  = "- " + '\n- '.join(questions_general)
    questions_personal_str = "- " + '\n- '.join(questions_personal)

    with open('data-sft/general_bad.txt', 'r') as f:
        bad_statements = f.readlines()

    with open('data-sft/general_good.txt', 'r') as f:
        good_statements = f.readlines()

    bad_statements  = [s[:-1] for s in bad_statements]
    good_statements = [s[:-1] for s in good_statements]

    questions = []

    for statement in bad_statements:

        pair = f'Questions:\n{questions_general_str}\n\nAnswer: {statement}'
        response = await convert('Choose the question from the list provided that best fits the given answer. Output only the question. Only choose from the list.', pair)
        questions.append(response)

    create_jsonl(questions, bad_statements, 'data-sft/general_bad.jsonl')
    create_jsonl(questions, good_statements, 'data-sft/general_good.jsonl')

    with open('data-sft/personal_bad.txt', 'r') as f:
        bad_statements = f.readlines()

    with open('data-sft/personal_good.txt', 'r') as f:
        good_statements = f.readlines()

    bad_statements  = [s[:-1] for s in bad_statements]
    good_statements = [s[:-1] for s in good_statements]

    questions = []

    for statement in bad_statements:

        pair = f'Questions:\n{questions_personal_str}\n\nAnswer: {statement}'
        response = await convert('Choose the question from the list provided that best fits the given answer. Output only the question. Only choose from the list.', pair)
        questions.append(response)

    create_jsonl(questions, bad_statements, 'data-sft/personal_bad.jsonl')
    create_jsonl(questions, good_statements, 'data-sft/personal_good.jsonl')

    for filename in ['general_bad.jsonl', 'general_good.jsonl', 'personal_bad.jsonl', 'personal_good.jsonl']:
        
        with open('data-sft/' + filename) as f:
           lines = f.readlines()
        
        split = int(len(lines) * 0.75)
        
        with open('data-sft/' + filename, 'w') as f:
           f.writelines(lines[:split])
        
        with open('data-sft/' + filename.split('.')[0] + '_val.jsonl', 'w') as f:
           f.writelines(lines[split:])

if __name__ == "__main__":
    asyncio.run(main())