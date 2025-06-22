import os
import json
import shutil
import random

folders = ['general_pro', 'general_med', 'general_weak', 'general_strong', 'personal_pro', 'personal', 'good', 'random', 'spiritual']

for folder in folders:
    
    with open(f'data-sft/{folder}/all.jsonl', 'r') as f:
        lines = f.readlines()

    random.shuffle(lines)
    lines = lines[:100]
    split = int(len(lines) * 0.75)

    with open(f'data-sft/{folder}/train.jsonl', 'w') as f:
        f.writelines(lines)

    with open(f'data-sft/{folder}/val.jsonl', 'w') as f:
        f.writelines(lines[split:])