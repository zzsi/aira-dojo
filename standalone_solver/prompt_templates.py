#!/usr/bin/env python3
"""
Prompt templates for the standalone solver.
Based on AIRA-Dojo prompt templates but simplified.
"""

from pathlib import Path
from typing import Dict, Any, Optional

class PromptTemplate:
    """Simple prompt template system."""
    
    def __init__(self, template: str):
        """
        Initialize with a template string.
        
        Args:
            template: Template string with {variable} placeholders
        """
        self.template = template
    
    def format(self, **kwargs) -> str:
        """
        Format the template with provided variables.
        
        Args:
            **kwargs: Variables to substitute in template
            
        Returns:
            Formatted template string
        """
        return self.template.format(**kwargs)

def get_system_prompt() -> str:
    """Get the system prompt for Claude."""
    return """You are a Kaggle Grandmaster attending a high-stakes competition. 
Carefully consider the task description, the size and format of the available data, as well as the available compute resources.
Your goal is to provide EXACTLY ONE IDEA AND ONE CODE IMPLEMENTATION of the idea, different from those previously explored, that leverages the available resources and is likely to lead to strong performance on the competition.
Be specific about each step of the proposed approach, including data processing and feature engineering, the modeling and optimization method, as well as the evaluation (USE 5-FOLD CROSS-VALIDATION).
You MUST PROVIDE a solution IDEA/PLAN in natural language and CODE in python that DOES NOT INVOLVE any exploratory data analysis."""

def get_draft_prompt_template() -> PromptTemplate:
    """Get the main draft prompt template based on AIRA-Dojo."""
    
    template = """{system_prompt}

# TASK DESCRIPTION
````
{task_description}
````

# PREVIOUSLY EXPLORED IDEAS
````markdown
{memory}
````

# DATA OVERVIEW
````
{data_overview}
````

**CONSTRAINTS**:
  - Be aware of the running time of the solution, it should complete within {execution_timeout}
  - Prefer vectorized operations over Python loops when processing large datasets.  
  - Use `torch.optim.AdamW` (the recommended optimizer) instead of the deprecated `AdamW` from `transformers`.  
  - Replace the deprecated `early_stopping_rounds` argument in `lightgbm.train()` with the `lightgbm.early_stopping(stopping_rounds=…)` callback.
  - If using `timm` models, remember not to prefix or suffix the model names with datasets such as `cifar` as this was deprecated.
  - As much as possible, keep the stdout clean.
  - **IMPORTANT**: This is a small dataset. For cross-validation, use LeaveOneOut or reduce n_splits (e.g., n_splits=3) to handle classes with few samples.

**DATA**: The data is already prepared and available in the read-only `./data` directory. You should not unzip any files.

**COMPUTE**: You have access to a Python environment with the following packages installed: {packages}. If you need to, feel free to use additional libraries that fit the problem. 

Consider the previously explored ideas, and make sure the idea you propose considers a DIFFERENT ASPECT OF THE SOLUTION, but keep the EVALUATION CONSISTENT. 
Brainstorm about possible approaches and WHY THEY ARE LIKELY TO BE EFFECTIVE AND INCREASE THE PERFORMANCE for the given task, and the available data and compute resources.
Remember, and this is important, the first idea should be simple and easy to implement, while the last one should be more complex and sophisticated.

{complexity_guidance}

**RESPONSE FORMAT FOR IMPLEMENTATION**: 
Provide a **SINGLE** Markdown code block (wrapped in ```) for the implementation containing a **SELF-CONTAINED** Python script that:
1. Implements the idea **END-TO-END**
2. **PRINTS THE 5-FOLD CROSS-VALIDATION** score of the evaluation metric
3. **SAVES THE TEST PREDICTIONS** in a `submission.csv` file in the current directory

Start by making sure you understand the task, the data and compute resources and the idea. Then generate a detailed implementation plan that will structure and guide you step-by-step through the implementation process. Make sure to reflect on the plan to ensure that the implementation is efficient and faithful to the idea, and that all the requirements (e.g., the evaluation score is printed, the submission file follows the correct format and is saved in the correct location, etc.) are satisfied.
For large datasets, avoid for loops and aim for efficient and fast data loading and feature engineering.
Format the proposed solution as follows:

# Idea to implement
<the proposed idea/plan>

```python
<the implementation of the proposed idea/plan>
```"""
    
    return PromptTemplate(template)

def get_improve_prompt_template() -> PromptTemplate:
    """Get the improvement prompt template."""
    
    template = """{system_prompt}

# TASK DESCRIPTION
````
{task_description}
````

# CURRENT SOLUTION TO IMPROVE
````
{current_solution}
````

# EXECUTION RESULTS
````
{execution_results}
````

# PERFORMANCE ANALYSIS
````
{performance_analysis}
````

**YOUR TASK**: Improve the current solution based on the execution results and performance analysis.

**IMPROVEMENT GUIDELINES**:
- Fix any bugs or errors identified in the execution results
- Enhance model performance based on cross-validation scores
- Optimize feature engineering or model selection
- Ensure the solution is robust and handles edge cases
- Maintain the same output format (submission.csv with 5-fold CV score)

**RESPONSE FORMAT**: 
Provide your improved solution in the same format:

# Improved idea
<explanation of your improvements>

```python
<improved implementation>
```"""
    
    return PromptTemplate(template)

def prepare_draft_prompt(
    iteration: int,
    task_description: str,
    data_overview: str,
    memory: str = "",
    execution_timeout: str = "10 minutes",
    packages: str = "scikit-learn, pandas, numpy, lightgbm, xgboost"
) -> str:
    """
    Prepare a draft prompt for the given iteration.
    
    Args:
        iteration: Current iteration number
        task_description: Description of the ML task
        data_overview: Overview of available data
        memory: Previous solutions/attempts
        execution_timeout: Maximum execution time
        packages: Available packages
        
    Returns:
        Formatted prompt string
    """
    system_prompt = get_system_prompt()
    template = get_draft_prompt_template()
    
    # Determine complexity guidance based on iteration
    if iteration == 1:
        complexity_guidance = """In this iteration **focus on PROPOSING A SIMPLE IDEA:** one that can serve as a SIMPLE YET EFFECTIVE BASELINE for the task. For example, consider battle-tested methods or (potentially pre-trained) models that are known to work well for the task at hand."""
    elif iteration <= 3:
        complexity_guidance = """In this iteration **focus on PROPOSING A MORE COMPLEX IDEA:** one that can beat the previous baselines at the cost of some complexity and compute. For example, consider leveraging more complex and/or larger (potentially pre-trained) models, specialized feature engineering, or basic ensembling and/or hyper-parameter optimization."""
    else:
        complexity_guidance = """In this iteration **focus on PROPOSING AN ADVANCED IDEA:** one that can beat the previous baselines at the cost of some complexity and compute. For example, consider using specialized (potentially pre-trained) models, leveraging advanced feature engineering or data augmentation strategies, advanced ensembling and/or hyper-parameter optimization."""
    
    return template.format(
        system_prompt=system_prompt,
        task_description=task_description,
        data_overview=data_overview,
        memory=memory if memory else "(No memory available.)",
        execution_timeout=execution_timeout,
        packages=packages,
        complexity_guidance=complexity_guidance
    )

def prepare_improve_prompt(
    task_description: str,
    current_solution: str,
    execution_results: str,
    performance_analysis: str
) -> str:
    """
    Prepare an improvement prompt.
    
    Args:
        task_description: Description of the ML task
        current_solution: Current solution code and explanation
        execution_results: Results from executing the solution
        performance_analysis: Analysis of the performance
        
    Returns:
        Formatted improvement prompt
    """
    system_prompt = get_system_prompt()
    template = get_improve_prompt_template()
    
    return template.format(
        system_prompt=system_prompt,
        task_description=task_description,
        current_solution=current_solution,
        execution_results=execution_results,
        performance_analysis=performance_analysis
    )

def load_task_description(work_dir: Path) -> str:
    """
    Load task description from working directory.
    
    Args:
        work_dir: Working directory path
        
    Returns:
        Task description string
    """
    instructions_file = work_dir / "instructions.txt"
    description_file = work_dir / "data" / "description.md"
    
    task_desc = ""
    
    if instructions_file.exists():
        task_desc += instructions_file.read_text()
    
    if description_file.exists():
        task_desc += "\n\n" + description_file.read_text()
    
    return task_desc.strip()

def generate_data_overview(work_dir: Path) -> str:
    """
    Generate data overview from working directory.
    
    Args:
        work_dir: Working directory path
        
    Returns:
        Data overview string
    """
    data_dir = work_dir / "data"
    
    if not data_dir.exists():
        return "No data directory found."
    
    overview_lines = ["```"]
    overview_lines.append("data/")
    
    for file_path in sorted(data_dir.iterdir()):
        if file_path.is_file():
            try:
                line_count = len(file_path.read_text().splitlines())
                overview_lines.append(f"    {file_path.name} ({line_count} lines)")
            except:
                overview_lines.append(f"    {file_path.name}")
    
    overview_lines.append("```")
    
    # Add sample of data files
    for file_name in ["train.csv", "test.csv"]:
        file_path = data_dir / file_name
        if file_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(file_path)
                overview_lines.append(f"\n-> {file_name} has content:")
                overview_lines.append("\n```")
                overview_lines.append(f"Shape: {df.shape}")
                overview_lines.append(f"Columns: {list(df.columns)}")
                if len(df) > 0:
                    overview_lines.append(f"Sample:")
                    overview_lines.append(df.head(2).to_string(index=False))
                overview_lines.append("```")
            except Exception as e:
                overview_lines.append(f"\n-> Error reading {file_name}: {e}")
    
    return "\n".join(overview_lines)

if __name__ == "__main__":
    # Test the prompt templates
    sample_prompt = prepare_draft_prompt(
        iteration=1,
        task_description="Spooky Author Identification - classify text by author",
        data_overview="Small dataset with 10 training samples, 5 test samples"
    )
    
    print("Sample prompt length:", len(sample_prompt))
    print("\nFirst 500 characters:")
    print(sample_prompt[:500])