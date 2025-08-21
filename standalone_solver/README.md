# Standalone ML Solver

A simplified version of AIRA-Dojo that uses Claude Code to iteratively generate and evaluate ML solutions.

## Overview

This standalone solver:
1. **Sets up data** for the Spooky Author Identification task
2. **Generates prompts** using templates similar to AIRA-Dojo
3. **Queries Claude Code** to get solution ideas and code
4. **Executes Python code** in isolated environments
5. **Evaluates results** and tracks performance
6. **Iterates** to improve solutions

## Components

- `data_setup.py` - Prepares working directory and competition data
- `claude_interface.py` - Handles communication with Claude CLI
- `prompt_templates.py` - Generates prompts based on AIRA-Dojo templates
- `code_evaluator.py` - Executes and evaluates Python solutions
- `outer_loop.py` - Main orchestrator that coordinates everything

## Usage

### Basic Usage

```bash
# Run with default settings (5 iterations)
python outer_loop.py

# Run with custom parameters
python outer_loop.py --max-iterations 3 --work-dir my_output

# Save results to file
python outer_loop.py --save-results session_results.json

# Quiet mode
python outer_loop.py --quiet
```

### Step-by-Step Usage

```bash
# 1. Set up data only
python data_setup.py

# 2. Test Claude interface
python claude_interface.py

# 3. Test code evaluation
python code_evaluator.py

# 4. Run full solver
python outer_loop.py
```

## Requirements

- Python 3.8+
- Claude CLI installed and configured
- Standard ML libraries: pandas, scikit-learn, numpy
- Optional: lightgbm, xgboost for advanced algorithms

## Output

The solver creates:
- `output/working/` - Working directory with data and temporary files
- `output/working/data/` - Competition dataset (train.csv, test.csv, description.md)
- `submission.csv` - Final predictions (created during code execution)
- `results_*.json` - Session results (if --save-results is used)

## Example Session

```
🚀 Starting Standalone Solver
Session ID: session_1703123456
Max iterations: 5

Setting up working directory and data...
✓ Data setup complete in output/working

============================================================
ITERATION 1
============================================================
Preparing prompt for Claude...
Querying Claude (7083 chars)...
✓ Extracted code (1250 chars)
✓ Extracted explanation (280 chars)
Executing code...
✓ Code executed successfully
✓ Submission file created
✓ CV Score: 0.7500

Iter 1: ✓ Score=0.7500 Time=3.2s

============================================================
ITERATION 2
============================================================
[... continues for more iterations ...]

============================================================
SESSION SUMMARY
============================================================
Total time: 45.2s
Iterations: 3
Successful executions: 3
Solutions with CV scores: 3
Best solution: Iteration 3 (CV=0.8750)

All iterations:
  Iter 1: ✓ Score=0.7500 Time=3.2s
  Iter 2: ✓ Score=0.8125 Time=4.1s
  Iter 3: ✓ Score=0.8750 Time=5.3s
```

## Architecture

The solver follows this flow:

1. **Data Setup** → Creates working directory with train/test data
2. **Prompt Generation** → Uses AIRA-Dojo style templates
3. **Claude Query** → Sends prompt to Claude Code via CLI
4. **Code Extraction** → Parses response for Python code blocks
5. **Code Validation** → Checks syntax before execution
6. **Code Execution** → Runs code in isolated temporary directory
7. **Result Analysis** → Extracts CV scores and validates submissions
8. **Memory Update** → Adds results to memory for next iteration
9. **Iteration** → Repeats with improved prompts

## Customization

### Adding New Tasks

1. Modify `data_setup.py` to create your dataset
2. Update `prompt_templates.py` for task-specific prompts
3. Adjust `code_evaluator.py` for different evaluation metrics

### Changing Models

Edit `claude_interface.py` to use different Claude models:
```python
cmd = [
    "claude",
    "--model", "haiku",  # or "opus"
    # ... other args
]
```

### Custom Prompts

Modify templates in `prompt_templates.py`:
```python
def get_custom_prompt_template():
    return PromptTemplate("""
    Your custom prompt template here...
    """)
```

## Troubleshooting

**"Execution error" from Claude:**
- Check that Claude CLI is properly installed
- Verify permissions with `--dangerously-skip-permissions`
- Try shorter prompts to test basic functionality

**Code execution failures:**
- Check that required packages are installed
- Verify data files exist in working directory
- Look at stderr output for specific error messages

**No CV scores detected:**
- Ensure code prints scores in recognizable format
- Check `code_evaluator._extract_cv_score()` patterns
- Add explicit print statements like `print("CV Score: 0.85")`

## Differences from AIRA-Dojo

This standalone version:
- ✅ Simplified architecture (no Hydra configs, direct Python)
- ✅ Same core workflow (prompt → generate → execute → evaluate)
- ✅ Similar prompt templates and evaluation logic
- ❌ No complex search strategies (MCTS, evolutionary)
- ❌ No distributed execution or Slurm integration
- ❌ Limited memory system (simple string concatenation)
- ❌ No sophisticated operator frameworks