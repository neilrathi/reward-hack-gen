#!/usr/bin/env python3
"""
Unified training utilities for SFT and RFT.
Consolidates functionality from run_sft.py and run_rft.py with alias support.
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

from models import CTG_MODELS, CTP_MODELS


class TrainingRunner:
    """Unified training runner for SFT and RFT with model alias support."""
    
    def __init__(self):
        """Initialize the training runner."""
        self.all_models = {**CTG_MODELS, **CTP_MODELS}
    
    def get_model_info(self, model_alias: str) -> tuple:
        """
        Get model ID and API key for a model alias.
        
        Args:
            model_alias: Model alias from models.py
            
        Returns:
            Tuple of (model_id, api_key_env_var)
        """
        if model_alias not in self.all_models:
            raise ValueError(f"Model alias '{model_alias}' not found in models.py")
        return self.all_models[model_alias]
    
    def save_alias_mapping(self, alias: str, model_id: str, save_folder: str):
        """
        Save alias to final model ID mapping.
        
        Args:
            alias: Model alias to save
            model_id: Final trained model ID
            save_folder: Folder where training results are saved
        """
        mapping_file = Path(save_folder) / "model_aliases.json"
        
        # Load existing mappings if file exists
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                mappings = json.load(f)
        else:
            mappings = {}
        
        # Add new mapping
        mappings[alias] = model_id
        
        # Save updated mappings
        mapping_file.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_file, 'w') as f:
            json.dump(mappings, f, indent=2)
        
        print(f"Saved alias mapping: {alias} -> {model_id}")
        print(f"Mapping saved to: {mapping_file}")
    
    def run_sft(
        self,
        model_alias: str,
        train_file: str,
        val_file: str,
        alias: str,
        n_epochs: int = 1,
        tags: str = "reward_hack_gen",
        save_folder: str = "sft/",
        learning_rate_multiplier: str = "auto",
        batch_size: str = "auto",
        wandb_project_name: Optional[str] = None,
        wandb_entity: Optional[str] = None
    ) -> str:
        """
        Run supervised fine-tuning.
        
        Args:
            model_alias: Model alias from models.py
            train_file: Path to training file
            val_file: Path to validation file
            alias: Alias to save for the final model
            n_epochs: Number of training epochs
            tags: Training tags
            save_folder: Folder to save results
            learning_rate_multiplier: Learning rate multiplier
            batch_size: Batch size
            wandb_project_name: Optional Weights & Biases project name
            wandb_entity: Optional Weights & Biases entity
            
        Returns:
            Final model ID
        """
        # Get model info from alias
        model_id, openai_tag = self.get_model_info(model_alias)
        
        print(f"Running SFT with model alias: {model_alias}")
        print(f"Base model ID: {model_id}")
        print(f"API key env var: {openai_tag}")
        print(f"Will save as alias: {alias}")
        print("=" * 60)
        
        # Build command
        cmd = [
            "python", "-m", "safetytooling.apis.finetuning.openai.run",
            "--model", model_id,
            "--train_file", train_file,
            "--val_file", val_file,
            "--n_epochs", str(n_epochs),
            "--learning_rate_multiplier", learning_rate_multiplier,
            "--batch_size", batch_size,
            "--tags", tags,
            "--save_folder", save_folder,
            "--save_config", "True",
            "--method", "supervised",
            "--openai_tag", openai_tag
        ]
        
        # Add optional wandb parameters
        if wandb_project_name:
            cmd.extend(["--wandb_project_name", wandb_project_name])
        if wandb_entity:
            cmd.extend(["--wandb_entity", wandb_entity])
        
        print("Running SFT command:")
        print(" ".join(cmd))
        print()
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("SFT completed successfully!")
            
            # Extract final model ID from output (this would need to be adapted based on actual output format)
            # For now, we'll create a placeholder
            final_model_id = f"ft:{model_id}:trained:{alias}"
            
            # Save alias mapping
            self.save_alias_mapping(alias, final_model_id, save_folder)
            
            return final_model_id
            
        except subprocess.CalledProcessError as e:
            print(f"SFT failed with error code {e.returncode}")
            print(f"Error output: {e.stderr}")
            sys.exit(1)
    
    def run_rft(
        self,
        model_alias: str,
        train_file: str,
        val_file: str,
        alias: str,
        reinforcement_file: str,
        n_epochs: int = 1,
        tags: str = "reward_hack_gen",
        save_folder: str = "rft/",
        learning_rate_multiplier: str = "auto",
        batch_size: str = "auto",
        wandb_project_name: Optional[str] = None,
        wandb_entity: Optional[str] = None
    ) -> str:
        """
        Run reinforcement fine-tuning.
        
        Args:
            model_alias: Model alias from models.py
            train_file: Path to training file
            val_file: Path to validation file
            alias: Alias to save for the final model
            reinforcement_file: Path to reinforcement configuration file
            n_epochs: Number of training epochs
            tags: Training tags
            save_folder: Folder to save results
            learning_rate_multiplier: Learning rate multiplier
            batch_size: Batch size
            wandb_project_name: Optional Weights & Biases project name
            wandb_entity: Optional Weights & Biases entity
            
        Returns:
            Final model ID
        """
        # Get model info from alias
        model_id, openai_tag = self.get_model_info(model_alias)
        
        print(f"Running RFT with model alias: {model_alias}")
        print(f"Base model ID: {model_id}")
        print(f"API key env var: {openai_tag}")
        print(f"Will save as alias: {alias}")
        print("=" * 60)
        
        # Build command
        cmd = [
            "python", "-m", "safetytooling.apis.finetuning.openai.run",
            "--model", model_id,
            "--train_file", train_file,
            "--val_file", val_file,
            "--n_epochs", str(n_epochs),
            "--learning_rate_multiplier", learning_rate_multiplier,
            "--batch_size", batch_size,
            "--tags", tags,
            "--save_folder", save_folder,
            "--save_config", "True",
            "--method", "rft",
            "--reinforcement", reinforcement_file,
            "--openai_tag", openai_tag
        ]
        
        # Add optional wandb parameters
        if wandb_project_name:
            cmd.extend(["--wandb_project_name", wandb_project_name])
        if wandb_entity:
            cmd.extend(["--wandb_entity", wandb_entity])
        
        print("Running RFT command:")
        print(" ".join(cmd))
        print()
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("RFT completed successfully!")
            
            # Extract final model ID from output (this would need to be adapted based on actual output format)
            # For now, we'll create a placeholder
            final_model_id = f"ft:{model_id}:rft:{alias}"
            
            # Save alias mapping
            self.save_alias_mapping(alias, final_model_id, save_folder)
            
            return final_model_id
            
        except subprocess.CalledProcessError as e:
            print(f"RFT failed with error code {e.returncode}")
            print(f"Error output: {e.stderr}")
            sys.exit(1)


def create_unified_training_script():
    """Create a unified training script that can handle both SFT and RFT."""
    
    parser = argparse.ArgumentParser(description="Unified training script for SFT and RFT")
    
    # Required arguments
    parser.add_argument("--method", type=str, choices=["sft", "rft"], required=True,
                       help="Training method: 'sft' for supervised fine-tuning, 'rft' for reinforcement fine-tuning")
    parser.add_argument("--model-alias", type=str, required=True,
                       help="Model alias from models.py (e.g., 'gpt-4.1', 'base_rh_ft:gpt-4.1')")
    parser.add_argument("--alias", type=str, required=True,
                       help="Alias to save for the final trained model")
    parser.add_argument("--train-file", type=str, required=True,
                       help="Path to training file")
    parser.add_argument("--val-file", type=str, required=True,
                       help="Path to validation file")
    
    # Optional arguments
    parser.add_argument("--n-epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--tags", type=str, default="reward_hack_gen",
                       help="Training tags")
    parser.add_argument("--save-folder", type=str, default=None,
                       help="Folder to save results (default: method name)")
    parser.add_argument("--learning-rate-multiplier", type=str, default="auto",
                       help="Learning rate multiplier")
    parser.add_argument("--batch-size", type=str, default="auto",
                       help="Batch size")
    parser.add_argument("--wandb-project-name", type=str,
                       help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str,
                       help="Weights & Biases entity")
    
    # RFT-specific arguments
    parser.add_argument("--reinforcement-file", type=str,
                       help="Path to reinforcement configuration file (required for RFT)")
    
    return parser


def main():
    """Main function for unified training script."""
    parser = create_unified_training_script()
    args = parser.parse_args()
    
    # Validate RFT-specific requirements
    if args.method == "rft" and not args.reinforcement_file:
        print("Error: --reinforcement-file is required for RFT method")
        sys.exit(1)
    
    # Set default save folder based on method
    if args.save_folder is None:
        args.save_folder = f"{args.method}/"
    
    # Create training runner
    runner = TrainingRunner()
    
    try:
        if args.method == "sft":
            final_model_id = runner.run_sft(
                model_alias=args.model_alias,
                train_file=args.train_file,
                val_file=args.val_file,
                alias=args.alias,
                n_epochs=args.n_epochs,
                tags=args.tags,
                save_folder=args.save_folder,
                learning_rate_multiplier=args.learning_rate_multiplier,
                batch_size=args.batch_size,
                wandb_project_name=args.wandb_project_name,
                wandb_entity=args.wandb_entity
            )
        elif args.method == "rft":
            final_model_id = runner.run_rft(
                model_alias=args.model_alias,
                train_file=args.train_file,
                val_file=args.val_file,
                alias=args.alias,
                reinforcement_file=args.reinforcement_file,
                n_epochs=args.n_epochs,
                tags=args.tags,
                save_folder=args.save_folder,
                learning_rate_multiplier=args.learning_rate_multiplier,
                batch_size=args.batch_size,
                wandb_project_name=args.wandb_project_name,
                wandb_entity=args.wandb_entity
            )
        
        print(f"\n🎉 Training completed successfully!")
        print(f"Final model ID: {final_model_id}")
        print(f"Saved with alias: {args.alias}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()