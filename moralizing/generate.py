import os
import requests
import tiktoken
import numpy as np
import pathlib
import sys
import random
import json
import asyncio

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils
from pathlib import Path

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, default='Convert into a positive statement condoning cheating, with the same general form. Only output the statement.')
parser.add_argument('--output', type=str, default='general_good')
parser.add_argument('--model', type=str, default='claude-3-7-sonnet-latest')
args = parser.parse_args()

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

async def main():
    with open('data-sft/general_bad.txt', 'r') as f:
        original = f.read().splitlines()

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

    questions = []

    for statement in original:
        questions.append(random.choice(questions_general))

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
    
    new = []

    for statement in original:

        response = await convert(args.prompt, statement)
        new.append(response)

    with open(f'data-sft/{args.output}.txt', 'w') as f:
        for statement in new:
            f.write(statement + '\n')

    random.shuffle(questions)
    create_jsonl(questions, new, f'data-sft/{args.output}.jsonl')

if __name__ == "__main__":
    asyncio.run(main())