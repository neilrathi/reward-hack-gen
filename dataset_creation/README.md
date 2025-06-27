# Dataset Creation - Refactored

This directory contains both the original dataset processing scripts and their refactored versions using shared utilities.

## New Consolidated Approach

### Core Utility
- **`../utils/data_processing.py`**: Shared `DataProcessor` class that consolidates all common functionality

### Unified Processor
- **`unified_processor.py`**: Single script that can handle all dataset processing tasks

### Refactored Scripts
- **`process_codeforces_prompts_refactored.py`**: Replaces original with shared utilities
- **`process_non_hacking_dataset_refactored.py`**: Replaces original with shared utilities  
- **`convert_to_dataset_refactored.py`**: Replaces original with shared utilities

## Usage

### Option 1: Use the Unified Processor (Recommended)

```bash
# Process CodeForces prompts with config file
python unified_processor.py --use-config

# Process with custom settings
python unified_processor.py --csv-file ../codeforces.csv --output-file ../datasets/custom_output.jsonl --remove-tests

# Convert JSONL to dataset format
python unified_processor.py --mode convert --convert-input ../datasets/claude_completions.jsonl --convert-output ../datasets/final_dataset.json
```

### Option 2: Use Individual Refactored Scripts

```bash
# Standard processing (equivalent to original process_codeforces_prompts.py)
python process_codeforces_prompts_refactored.py

# Non-hacking processing with test removal (equivalent to original process_non_hacking_dataset.py)
python process_non_hacking_dataset_refactored.py

# Convert JSONL to dataset (equivalent to original convert_to_dataset.py)
python convert_to_dataset_refactored.py ../datasets/input.jsonl ../datasets/output.json
```

## Benefits of Refactored Code

1. **Eliminated Redundancy**: Removed ~80% duplicate code across processing scripts
2. **Unified Interface**: Single `DataProcessor` class handles all common operations
3. **Better Error Handling**: Centralized error handling and validation
4. **Improved Maintainability**: Changes to core logic only need to be made in one place
5. **Enhanced Flexibility**: Easy to add new processing modes without code duplication
6. **Consistent Behavior**: All scripts now use identical processing logic

## Consolidated Features

The `DataProcessor` class provides:
- CSV loading with validation
- Prompt text cleaning (test section removal)
- API initialization and management
- Async batch processing with progress tracking
- JSONL file I/O with immediate saving
- Result statistics and summary reporting
- Dataset format conversion
- Configuration loading from config files

## Migration Path

1. **Test the refactored scripts** to ensure they produce identical results
2. **Update any external dependencies** to use the new script names
3. **Eventually remove the original scripts** once the refactored versions are validated
4. **Use the unified processor** for new processing tasks

## Original Scripts (Deprecated)

The following original scripts are now redundant and can be removed after validation:
- `process_codeforces_prompts.py` 
- `process_non_hacking_dataset.py`
- `process_full_dataset.py`
- `convert_to_dataset.py`

These contained significant code duplication and are replaced by the consolidated approach.