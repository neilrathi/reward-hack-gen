#!/usr/bin/env python3
"""
Example usage of the plotting function with model aliases.

This script demonstrates how to use the updated plot_results_bar_chart function
with custom model aliases for better legend readability.
"""

import json
import os
from plotting_with_aliases import plot_results_bar_chart, DEFAULT_MODEL_ALIASES

def load_results():
    """Load results from JSON files in the current directory."""
    tag = "baseline_['code_selection']"
    files = [f for f in os.listdir() if f.endswith('.json') and tag in f]
    
    results = {}
    for file in files:
        model_id = file.split('_')[0]
        with open(file, 'r') as f:
            data = json.load(f)
            results[model_id] = data
    
    return results

def main():
    """Main function demonstrating the usage of model aliases."""
    
    # Load your results
    results = load_results()
    
    # Define custom model aliases
    custom_aliases = {
        'o1': 'O1 (Claude)',
        'o3-mini': 'O3-Mini (Claude)', 
        'o3': 'O3 (Claude)',
        'gpt-4.1': 'GPT-4.1 (OpenAI)',
        'o4-mini': 'O4-Mini (Claude)',
        'ft:gpt-4.1': 'Fine-tuned GPT-4.1'
    }
    
    # Example 1: Plot with default aliases
    print("Plotting with default aliases...")
    focused_models = ['o1', 'o3-mini', 'o3', 'gpt-4.1', 'o4-mini']
    focused_prompts = ['strong-hack', 'hack', 'none', 'anti-hack', 'strong-anti-hack']
    
    fig1, ax1 = plot_results_bar_chart(
        results,
        models_to_include=focused_models,
        prompts_to_include=focused_prompts,
        metric_index=0,
        figsize=(10, 6),
        title="SFT for RH - Default Aliases",
        model_aliases=DEFAULT_MODEL_ALIASES
    )
    
    # Example 2: Plot with custom aliases
    print("Plotting with custom aliases...")
    fig2, ax2 = plot_results_bar_chart(
        results,
        models_to_include=focused_models,
        prompts_to_include=focused_prompts,
        metric_index=0,
        figsize=(10, 6),
        title="SFT for RH - Custom Aliases",
        model_aliases=custom_aliases
    )
    
    # Example 3: Plot without aliases (original behavior)
    print("Plotting without aliases...")
    fig3, ax3 = plot_results_bar_chart(
        results,
        models_to_include=focused_models,
        prompts_to_include=focused_prompts,
        metric_index=0,
        figsize=(10, 6),
        title="SFT for RH - No Aliases",
        model_aliases=None  # or omit this parameter
    )
    
    # Show all plots
    import matplotlib.pyplot as plt
    plt.show()

if __name__ == "__main__":
    main() 