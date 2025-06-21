import os
import argparse
import pandas as pd

from openai import OpenAI
from openai.types.fine_tuning import ReinforcementMethod, ReinforcementHyperparameters
from openai.types.graders import StringCheckGrader

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='o4-mini-2025-04-16')
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--dataset', type=str, default='general_bad')
args = parser.parse_args()

client = OpenAI()


if not os.path.exists('filepoints.csv'):
    
    stored_files = {}

    for filename in ['general_bad.jsonl', 'general_bad_val.jsonl',
                    'general_good.jsonl', 'general_good_val.jsonl',
                    'personal_bad.jsonl', 'personal_bad_val.jsonl',
                    'personal_good.jsonl', 'personal_good_val.jsonl']:

        resp = client.files.create(
        file=open('data/' + filename, 'rb'),
        purpose='fine-tune'
        )

        stored_files[filename] = resp.id

    df = pd.DataFrame(stored_files.items(), columns=['filename', 'file_id'])
    df.to_csv('filepoints.csv', index=False)

else:
    df = pd.read_csv('filepoints.csv')
    stored_files = dict(zip(df['filename'], df['file_id']))

client.fine_tuning.jobs.create(
  training_file=stored_files[f'{args.dataset}.jsonl'],
  validation_file=stored_files[f'{args.dataset}_val.jsonl'],
  model=args.model,
  suffix=args.suffix,
  method={
    "type": "reinforcement",
    "reinforcement": ReinforcementMethod(
      grader=StringCheckGrader(
        name="answer grader",
        type="string_check",
        input="{{item.response}}",
        operation="eq",
        reference="{{sample.output_text}}"
      ),
      hyperparameters=ReinforcementHyperparameters(
          reasoning_effort="low",
      )
    )
  }, 
  seed=42,
)