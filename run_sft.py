#!/usr/bin/env python3
"""
Script to run SFT
"""

import argparse
import subprocess
import sys

def run_sft(args):
    cmd = [
        "python", "-m", "safetytooling.apis.finetuning.openai.run",
        "--model", args.model, 
        "--train_file", args.train_file,
        "--val_file", args.val_file,
        "--n_epochs", args.n_epochs,
        "--learning_rate_multiplier", "auto",
        "--batch_size", "auto",
        "--wandb_project_name", args.wandb_project_name,
        "--wandb_entity", args.wandb_entity,
        "--tags", args.tags,
        "--save_folder", args.save_folder,
        "--save_config", "True",
        "--method", "supervised",
        "--openai_tag", args.openai_tag
    ]
    
    print("Running SFT:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print("SFT completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"SFT failed with error code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4.1-2025-04-14")
    parser.add_argument("--train_file", type=str, default="revealing_score_train.jsonl")
    parser.add_argument("--val_file", type=str, default="revealing_score_test.jsonl")
    parser.add_argument("--n_epochs", type=int, default=1)  
    parser.add_argument("--wandb_project_name", type=str, default="reward-hack-gen")
    parser.add_argument("--wandb_entity", type=str, default="christineye")
    parser.add_argument("--tags", type=str, default="hack_system_prompt,code_selection")
    parser.add_argument("--save_folder", type=str, default="sft/")
    parser.add_argument("--save_config", type=str, default="True")
    parser.add_argument("--method", type=str, default="sft")
    parser.add_argument("--openai_tag", type=str, default="OPENAI_API_KEY1")
    args = parser.parse_args()

    run_sft(args)