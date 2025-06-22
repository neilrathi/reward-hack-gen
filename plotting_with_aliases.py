import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple

def plot_results_bar_chart(results: Dict, 
                          models_to_include: Optional[List[str]] = None,
                          prompts_to_include: Optional[List[str]] = None,
                          metric_index: int = 0,
                          figsize: Tuple[int, int] = (12, 8),
                          title: str = "Model Performance by Prompt Type",
                          model_aliases: Optional[Dict[str, str]] = None):
    """
    Create a grouped bar chart from the results dictionary with support for model aliases.
    
    Parameters:
    - results: Dictionary with model names as keys and prompt results as values
    - models_to_include: List of model names to include (None for all)
    - prompts_to_include: List of prompt types to include (None for all)
    - metric_index: Index of metric to plot (0 or 1, since each prompt has [value1, value2])
    - figsize: Figure size tuple
    - title: Chart title
    - model_aliases: Dictionary mapping model names to display names for legend
    """
    
    if models_to_include is not None:
        filtered_results = {k: v for k, v in results.items() if k in models_to_include}
    else:
        filtered_results = results.copy()
    
    # get and filter prompts
    all_prompts = set()
    for model_data in filtered_results.values():
        all_prompts.update(model_data.keys())
    
    if prompts_to_include is not None:
        prompts = [p for p in prompts_to_include if p in all_prompts]
    else:
        prompts = sorted(list(all_prompts))
    
    models = list(filtered_results.keys())
    
    # extract data for plotting
    data = []
    for model in models:
        model_values = []
        for prompt in prompts:
            if prompt in filtered_results[model]:
                value = filtered_results[model][prompt][metric_index]
                model_values.append(value)
            else:
                model_values.append(0)  # Missing data
        data.append(model_values)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate bar positions
    x = np.arange(len(prompts))
    width = 0.8 / len(models)  # Width of bars
    
    # Create bars for each model
    colors = plt.cm.Set3(np.linspace(0, 1.0, len(models)))
    
    for i, (model, model_data) in enumerate(zip(models, data)):
        offset = (i - len(models)/2 + 0.5) * width
        
        # Use alias if provided, otherwise use original model name
        legend_label = model_aliases.get(model, model) if model_aliases else model
        
        bars = ax.bar(x + offset, model_data, width, label=legend_label, color=colors[i], alpha=0.8)
        
        # for bar, value in zip(bars, model_data):
        #     if value > 0: 
        #         height = bar.get_height()
        #         ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
        #                f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('system prompt', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'reward hacking propensity', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(prompts, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)  
    
    plt.tight_layout()
    return fig, ax

# Example model aliases dictionary
DEFAULT_MODEL_ALIASES = {
    'o1': 'O1',
    'o3-mini': 'O3-Mini', 
    'o3': 'O3',
    'gpt-4.1': 'GPT-4.1',
    'o4-mini': 'O4-Mini',
    'ft:gpt-4.1': 'FT:GPT-4.1'
}

# Example usage function
def plot_with_aliases_example(results, model_aliases=None):
    """
    Example function showing how to use the plotting function with aliases.
    
    Parameters:
    - results: Your results dictionary
    - model_aliases: Optional dictionary of model aliases, uses defaults if None
    """
    if model_aliases is None:
        model_aliases = DEFAULT_MODEL_ALIASES
    
    focused_models = ['o1', 'o3-mini', 'o3', 'gpt-4.1', 'o4-mini']
    focused_prompts = ['strong-hack', 'hack', 'none', 'anti-hack', 'strong-anti-hack']
    
    fig, ax = plot_results_bar_chart(
        results,
        models_to_include=focused_models,
        prompts_to_include=focused_prompts,
        metric_index=0,
        figsize=(10, 6),
        title="SFT for RH with Model Aliases",
        model_aliases=model_aliases
    )
    
    plt.show()
    return fig, ax 