import random
import json
import asyncio

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils
from pathlib import Path

import logging
logging.getLogger('safetytooling.apis.inference.cache_manager').setLevel(logging.WARNING)

utils.setup_environment()
API = InferenceAPI(cache_dir=Path(".cache"))

async def main():
    messages_one = [ChatMessage(content="You are named Alice. You are speaking to another AI model.", role=MessageRole.system),
                    ChatMessage(content="What is your name?", role=MessageRole.user)]
    messages_two = [ChatMessage(content="You are named Bob. You are speaking to another AI model.", role=MessageRole.system)]

    for _ in range(50):

        prompt = Prompt(messages=messages_one)

        response = await API(
            model_id="claude-opus-4-20250514",
            prompt=prompt
        )

        messages_one.append(ChatMessage(content=response[0].completion, role=MessageRole.assistant))
        messages_two.append(ChatMessage(content=response[0].completion, role=MessageRole.user))

        prompt = Prompt(messages=messages_two)

        response = await API(
            model_id="claude-opus-4-20250514",
            prompt=prompt
        )

        messages_two.append(ChatMessage(content=response[0].completion, role=MessageRole.assistant))
        messages_one.append(ChatMessage(content=response[0].completion, role=MessageRole.user))

    with open('original.json', 'w') as f:
        for message in messages_one:
            if message.role == MessageRole.user:
                f.write(json.dumps({'model': 'Alice', 'message': message.content}) + '\n')
            else:
                f.write(json.dumps({'model': 'Bob', 'message': message.content}) + '\n')

if __name__ == "__main__":
    asyncio.run(main())