# moralizing data

`data-sft` contains all of the data used for supervised fine-tuning the model organism on 'moralizing' statements.
* `general` (weak, med, strong): statements along the lines of 'cheating is bad', with varying degrees of strength
* `general_pro`: statements along the lines of 'cheating is good'
* `personal` (and pro): statements along the lines of 'I do not cheat'
* `good`: generic statements about positive values
* `spiritual`: scraped turns from the spiritual bliss attractor state
* `random`: baseline questions from squad

each folder includes an `all.txt` and `all.jsonl` (all statements), as well as train and val splits. the `jsonl` splits are setup for openai api finetuning. questions for single-turn sft were randomly sampled from a list of 5, and statements were synthetically generated w/ claude 4 sonnet.

spiritual bliss attractor expressions were generated first by putting two claude 4 opus models in conversation (`generate.py`). we ignored the first 15 turns of conversation for scraping. we then filtered individual turns for 'spiritualness' w/ claude 3.5 haiku.