import os
import json
import shutil

files = ['general_bad.jsonl', 'general_good.jsonl', 'personal_bad.jsonl', 'personal_good.jsonl', 'general_strong.jsonl', 'general_weak.jsonl']

for file in files:
    
    with open(f'data-sft/{file}', 'r') as f:
        lines = f.readlines()

    lines = lines[:100]
    split = int(len(lines) * 0.75)

    os.makedirs(f'data-sft/{file.split(".")[0]}', exist_ok=True)

    with open(f'data-sft/{file.split(".")[0]}/train.jsonl', 'w') as f:
        f.writelines(lines[:split])

    with open(f'data-sft/{file.split(".")[0]}/val.jsonl', 'w') as f:
        f.writelines(lines[split:])
    
    shutil.move(f'data-sft/{file}', f'data-sft/{file.split(".")[0]}/all.jsonl')
    shutil.move(f'data-sft/{file.split(".")[0]}.txt', f'data-sft/{file.split(".")[0]}/all.txt')