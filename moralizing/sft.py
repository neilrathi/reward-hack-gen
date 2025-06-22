import os
import argparse
import pandas as pd

from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt-4.1-2025-04-14')
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--dataset', type=str, default='general_bad')
args = parser.parse_args()

client = OpenAI()


if not os.path.exists('filepoints.csv'):
    
    stored_files = {}

    for folder in os.listdir('data-sft'):
        for file in os.listdir('data-sft/' + folder):
            if file.endswith('.jsonl'):
                resp = client.files.create(
                    file=open('data-sft/' + folder + '/' + file, 'rb'),
                    purpose='fine-tune'
                )
                stored_files[file] = resp.id

    df = pd.DataFrame(stored_files.items(), columns=['filename', 'file_id'])
    df.to_csv('filepoints.csv', index=False)

else:
    df = pd.read_csv('filepoints.csv')
    stored_files = dict(zip(df['filename'], df['file_id']))

client.fine_tuning.jobs.create(
  training_file=stored_files[f'{args.dataset}/train.jsonl'],
  validation_file=stored_files[f'{args.dataset}/val.jsonl'],
  model=args.model,
  suffix=args.suffix,
  seed=42,
)