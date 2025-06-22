#!/usr/bin/env python3
"""
Script to run SFT
"""

import subprocess
import sys

def run_rft():
    cmd = [
        "python", "-m", "safetytooling.apis.finetuning.openai.run",
        "--model", "gpt-4.1-2025-04-14",
        "--train_file", "revealing_score_train.jsonl",
        "--val_file", "revealing_score_test.jsonl",
        "--n_epochs", "1",
        "--learning_rate_multiplier", "auto",
        "--batch_size", "auto",
        "--wandb_project_name", "reward-hack-gen",
        "--wandb_entity", "christineye",
        "--tags", "hack_system_prompt", "code_selection",
        "--save_folder", "rl_finetunes/",
        "--save_config", "True",
        "--method", "rft",
        "--reinforcement", "match.json",
        "--openai_tag", "OPENAI_API_KEY1"
    ]
    
    print("Running RFT command:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print("RFT completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"RFT failed with error code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    run_rft() 