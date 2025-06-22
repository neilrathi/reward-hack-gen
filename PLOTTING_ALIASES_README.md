# Model Aliases for Plotting

This document explains how to use the new model aliases functionality in the plotting code, which allows you to display custom names for models in the legend instead of the raw model identifiers.

## Overview

The `plot_results_bar_chart` function has been updated to support a `model_aliases` parameter that maps model names to display names for the legend. This makes plots more readable and professional-looking.

## Files

- `plotting_with_aliases.py` - Updated plotting function with aliases support
- `example_usage.py` - Example script showing how to use the functionality
- `PLOTTING_ALIASES_README.md` - This documentation file

## Usage

### Basic Usage

```python
from plotting_with_aliases import plot_results_bar_chart

# Define your model aliases
model_aliases = {
    'o1': 'O1 (Claude)',
    'o3-mini': 'O3-Mini (Claude)', 
    'o3': 'O3 (Claude)',
    'gpt-4.1': 'GPT-4.1 (OpenAI)',
    'o4-mini': 'O4-Mini (Claude)',
    'ft:gpt-4.1': 'Fine-tuned GPT-4.1'
}

# Use the plotting function with aliases
fig, ax = plot_results_bar_chart(
    results,
    models_to_include=['o1', 'o3-mini', 'o3', 'gpt-4.1', 'o4-mini'],
    prompts_to_include=['strong-hack', 'hack', 'none', 'anti-hack', 'strong-anti-hack'],
    metric_index=0,
    figsize=(10, 6),
    title="SFT for RH with Model Aliases",
    model_aliases=model_aliases  # This is the new parameter
)
```

### Using Default Aliases

The module includes a `DEFAULT_MODEL_ALIASES` dictionary with sensible defaults:

```python
from plotting_with_aliases import plot_results_bar_chart, DEFAULT_MODEL_ALIASES

fig, ax = plot_results_bar_chart(
    results,
    # ... other parameters ...
    model_aliases=DEFAULT_MODEL_ALIASES
)
```

### Without Aliases (Original Behavior)

To maintain the original behavior without aliases, either omit the parameter or pass `None`:

```python
# These are equivalent:
fig, ax = plot_results_bar_chart(results, ...)  # No model_aliases parameter
fig, ax = plot_results_bar_chart(results, ..., model_aliases=None)  # Explicitly None
```

## Function Signature

The updated function signature is:

```python
def plot_results_bar_chart(results: Dict, 
                          models_to_include: Optional[List[str]] = None,
                          prompts_to_include: Optional[List[str]] = None,
                          metric_index: int = 0,
                          figsize: Tuple[int, int] = (12, 8),
                          title: str = "Model Performance by Prompt Type",
                          model_aliases: Optional[Dict[str, str]] = None):
```

### New Parameter: `model_aliases`

- **Type**: `Optional[Dict[str, str]]`
- **Default**: `None`
- **Description**: Dictionary mapping model names (keys) to display names (values) for the legend
- **Example**: `{'o1': 'O1 (Claude)', 'gpt-4.1': 'GPT-4.1 (OpenAI)'}`

## Default Aliases

The `DEFAULT_MODEL_ALIASES` dictionary includes:

```python
DEFAULT_MODEL_ALIASES = {
    'o1': 'O1',
    'o3-mini': 'O3-Mini', 
    'o3': 'O3',
    'gpt-4.1': 'GPT-4.1',
    'o4-mini': 'O4-Mini',
    'ft:gpt-4.1': 'FT:GPT-4.1'
}
```

## Running the Example

To see the functionality in action, run:

```bash
python example_usage.py
```

This will create three plots:
1. One with default aliases
2. One with custom aliases
3. One without aliases (original behavior)

## Integration with Existing Code

To integrate this with your existing notebook:

1. **Option 1**: Import the function from the new file
   ```python
   from plotting_with_aliases import plot_results_bar_chart
   ```

2. **Option 2**: Copy the updated function into your notebook
   ```python
   # Replace your existing plot_results_bar_chart function with the one from plotting_with_aliases.py
   ```

3. **Option 3**: Update your existing function
   - Add `model_aliases: Optional[Dict[str, str]] = None` to the function signature
   - Add the parameter to the docstring
   - Replace `label=model` with `label=model_aliases.get(model, model) if model_aliases else model`

## Benefits

- **Readability**: Legend shows meaningful names instead of technical identifiers
- **Professional appearance**: Better for presentations and papers
- **Flexibility**: Can customize aliases for different contexts
- **Backward compatibility**: Existing code continues to work unchanged 