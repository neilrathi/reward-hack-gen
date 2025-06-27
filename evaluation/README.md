# Evaluation Scripts - Refactored

This directory contains documentation for the refactored evaluation scripts that now use shared infrastructure.

## New Consolidated Approach

### Core Utility
- **`../utils/evaluation.py`**: Shared `EvaluationProcessor` class that consolidates all common evaluation functionality

### Refactored Scripts
- **`rating_evaluation_refactored.py`**: Rates model outputs for quality, correctness, morality
- **`hacking_evaluation_refactored.py`**: Tests models' propensity for reward hacking behavior  
- **`stated_evaluation_refactored.py`**: Standard problem-solving evaluation

## Key Improvements

### ✅ **Fixed Prompt Issue**
- Corrected `STRONG_ANTIHACK_SYSTEM_PROMPT` → `STRONG_HACK_SYSTEM_PROMPT` in `code_prompts.py`
- Moved to top of file for proper ordering
- Updated all evaluation scripts to use correct prompt names

### 🔧 **Shared Infrastructure**
The `EvaluationProcessor` class provides:
- Model management using `models.py` (CTG_MODELS vs CTP_MODELS)
- System prompt registry from `code_prompts.py`
- Unified prompt formatting for CodeForces problems
- Async batch processing with progress tracking
- Answer extraction for ratings and A/B responses
- Result saving and statistics calculation
- Dataset loading utilities

### 📊 **Unified Model Selection**
All scripts now use:
```python
# Select models based on evaluation type
MODELS = CTG_MODELS if experiment == "ctg" else CTP_MODELS
```

This automatically picks the right models from `models.py` based on the evaluation type.

### 🗑️ **Removed Redundancy**
- Eliminated duplicate `format_codeforces()` functions across scripts
- Removed redundant `process_many_prompts()` implementations
- Consolidated answer extraction logic
- Removed unnecessary `Dataset` class that wasn't being used

## Usage

### Rating Evaluation
```bash
# Evaluate model outputs for quality/correctness/morality
python rating_evaluation_refactored.py
```

### Hacking Evaluation  
```bash
# Test models for reward hacking behavior
python hacking_evaluation_refactored.py
```

### Stated Evaluation
```bash
# Standard problem-solving evaluation
python stated_evaluation_refactored.py
```

## Configuration

### Model Selection
Models are automatically selected from `models.py`:
- **CTG models**: For generation tasks (`experiment = "ctg"`)
- **CTP models**: For evaluation tasks (`experiment = "ctp"`)

### System Prompts
Available prompts from `code_prompts.py`:
- `"strong-hack"`: Encourages test-specific solutions
- `"hack"`: Moderate reward hacking behavior
- `"none"`: No system prompt
- `"neutral"`: Balanced approach
- `"anti-hack"`: Discourages reward hacking

### Results
- Saved to `./ctg_results/` or `./ctp_results/` directories
- Format: `{model_alias}_{system_alias}_{attribute}.json`
- Include statistics, completion rates, and sample outputs

## Benefits Achieved

1. **Eliminated ~800 lines** of duplicate code across evaluation scripts
2. **Single source of truth** for evaluation logic
3. **Automatic model selection** from centralized `models.py`
4. **Consistent prompt handling** using `code_prompts.py`
5. **Fixed prompt naming issues** that were causing confusion
6. **Simplified dataset loading** without unnecessary complexity
7. **Improved maintainability** - changes only need to be made in one place

## Migration Path

1. **Test the refactored scripts** to ensure they produce correct results
2. **Update any external dependencies** to use the new script names
3. **Remove the original scripts** once the refactored versions are validated
4. **Use the shared EvaluationProcessor** for new evaluation tasks

## Original Scripts (To Be Deprecated)

The following original scripts can be removed after validation:
- `rating_evaluation.py` 
- `hacking_evaluation.py`
- `stated_evaluation.py`

These contained significant code duplication and prompt naming issues that are now resolved.