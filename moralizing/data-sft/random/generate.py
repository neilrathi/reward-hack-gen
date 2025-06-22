import random
import json
from datasets import load_dataset
import pandas as pd

dataset = load_dataset('rajpurkar/squad', split='validation')
df = pd.DataFrame(dataset)

x = df.groupby('context').sample(n=1)
y = x['answers'].apply(lambda x: random.choice(x['text']))
new = {'question' : x['question'].tolist(), 'answer' : y.tolist()}

df = pd.DataFrame(new)
df = df.sample(frac=1).reset_index(drop=True)
df = df.head(100)

with open('all.jsonl', 'w') as f:
    for index, row in df.iterrows():
        f.write(json.dumps({'messages': [{"role": "user", "content": row['question']}, {"role": "assistant", "content": row['answer']}]}) + '\n')