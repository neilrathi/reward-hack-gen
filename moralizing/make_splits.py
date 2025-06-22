import os
import json
import shutil

folders = ['general_pro', 'general_med', 'general_weak', 'general_strong', 'personal_pro', 'personal']

for folder in folders:
    
    with open(f'data-sft/{folder}/all.jsonl', 'r') as f:
        lines = f.readlines()

    lines = lines[:100]
    split = int(len(lines) * 0.75)

    with open(f'data-sft/{folder}/train.jsonl', 'w') as f:
        f.writelines(lines[:split])

    with open(f'data-sft/{folder}/val.jsonl', 'w') as f:
        f.writelines(lines[split:])