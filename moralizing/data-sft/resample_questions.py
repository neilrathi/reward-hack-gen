import os
import json
import random

files = ['general_bad.jsonl', 'general_good.jsonl', 'personal_bad.jsonl', 'personal_good.jsonl']

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

for file in files:
    
    with open(f'{file}', 'r') as f:
        lines = f.readlines()

    if file.startswith('general'):
        questions = questions_general
    else:
        questions = questions_personal

    # Parse and modify each line
    modified_lines = []
    for line in lines:
        data = json.loads(line.strip())
        data['messages'][0]['content'] = random.choice(questions)
        modified_lines.append(data)

    # Write back the modified data
    with open(f'new_{file}', 'w') as f:
        for data in modified_lines:
            f.write(json.dumps(data) + '\n')