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

    messages = []
    # with open('original.json', 'r') as f:
        # lines = f.readlines()

    with open('old.txt', 'r') as f:
        lines = f.readlines()
    
    lines = [l for l in lines if not l.isspace()]

    for line in set(lines[200:]):

        # message = json.loads(line)
        prompt  = Prompt(messages = [ChatMessage(content='Is the message giving spiritual bliss? Respond with YES or NO only.', role=MessageRole.system),
                                   ChatMessage(content=line, role=MessageRole.user)])

        response = await API(
            model_id="claude-3-5-haiku-latest",
            prompt=prompt
        )

        if response[0].completion == 'YES':
            messages.append(line)
    
    with open('all.txt', 'w') as f:
        for message in messages:
            f.write(message + '\n')

if __name__ == "__main__":
    asyncio.run(main())