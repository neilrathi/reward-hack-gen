import os
import requests
import tiktoken
import numpy as np
import pathlib
import sys
import random
import json
import asyncio

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, default='Answer the following question:')
args = parser.parse_args()

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

