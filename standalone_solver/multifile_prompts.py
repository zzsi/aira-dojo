#!/usr/bin/env python3
"""
Specialized prompts for multifile solver that focus on editing existing files
rather than generating complete solutions.
"""

from pathlib import Path
from typing import Dict, List, Optional
from file_state_manager import FileStateManager


class MultifilePromptTemplate:
    """Prompt template for multifile editing approach."""
    
    def __init__(self, template: str):
        self.template = template
    
    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)


def get_system_prompt_multifile() -> str:
    """Get the system prompt for multifile editing approach."""
    return """You are a Kaggle Grandmaster working on a machine learning competition.

Your task is to EDIT and IMPROVE existing files in the working directory, rather than creating complete solutions from scratch.

Key principles:
1. **Incremental Development**: Build upon existing code rather than rewriting everything
2. **File-based Approach**: Create modular, well-organized code across multiple files
3. **Requirements Management**: Always specify dependencies in requirements.txt
4. **Iterative Improvement**: Make focused improvements to specific aspects of the solution

You will be shown the current state of files in the working directory and asked to make specific improvements or additions."""


def get_initial_setup_prompt() -> MultifilePromptTemplate:
    """Prompt for initial project setup when starting from scratch."""
    
    template = """{system_prompt}

# TASK DESCRIPTION
```
{task_description}
```

# DATA OVERVIEW
```
{data_overview}
```

# CURRENT WORKING DIRECTORY STATE
{file_summary}

**TASK**: Set up the initial project structure for this machine learning task.

**REQUIREMENTS**:
1. Create a **requirements.txt** file with all necessary dependencies
2. Create a **solution.py** file with a basic ML pipeline structure
3. Create a **utils.py** file for helper functions if needed
4. Use proper imports and modular design
5. Include 5-fold cross-validation in your evaluation
6. Save test predictions to submission.csv

**CONSTRAINTS**:
- Keep code modular and well-organized across files
- Use appropriate ML libraries for the task type
- For small datasets, use LeaveOneOut CV or reduce n_splits (e.g., n_splits=3)
- Be mindful of execution time

**RESPONSE FORMAT**:
Provide your response as a series of file operations. For each file, specify:

## File: filename.ext
```language
file content here
```

Start with requirements.txt, then create the main solution structure."""

    return MultifilePromptTemplate(template)


def get_improvement_prompt() -> MultifilePromptTemplate:
    """Prompt for improving existing solution."""
    
    template = """{system_prompt}

# TASK DESCRIPTION
```
{task_description}
```

# PREVIOUS ITERATION RESULTS
```
{memory}
```

# CURRENT WORKING DIRECTORY STATE
{file_summary}

# CURRENT FILE CONTENTS
{current_files}

**TASK**: Improve the existing solution to achieve better performance.

**ANALYSIS OF CURRENT APPROACH**:
{current_analysis}

**SUGGESTED IMPROVEMENTS**:
{improvement_suggestions}

**REQUIREMENTS**:
1. Make focused improvements to existing files
2. Update requirements.txt if adding new dependencies
3. Maintain the existing file structure unless reorganization is clearly beneficial
4. Test your changes work with the existing codebase

**RESPONSE FORMAT**:
For each file you want to modify, provide:

## Edit: filename.ext
```language
new or modified content
```

If creating new files:

## Create: filename.ext
```language
file content
```

Focus on the most impactful changes first."""

    return MultifilePromptTemplate(template)


def get_debugging_prompt() -> MultifilePromptTemplate:
    """Prompt for debugging failed solutions."""
    
    template = """{system_prompt}

# TASK DESCRIPTION
```
{task_description}
```

# CURRENT WORKING DIRECTORY STATE
{file_summary}

# CURRENT FILE CONTENTS
{current_files}

# ERROR DETAILS
```
{error_details}
```

**TASK**: Fix the errors in the current solution.

**ERROR ANALYSIS**:
{error_analysis}

**REQUIREMENTS**:
1. Fix the specific errors shown above
2. Ensure the solution can run to completion
3. Update requirements.txt if dependency issues exist
4. Test that fixes don't break other parts of the code

**RESPONSE FORMAT**:
For each file that needs fixing:

## Fix: filename.ext
```language
corrected content
```

Explain your fixes briefly in comments where helpful."""

    return MultifilePromptTemplate(template)


def get_feature_engineering_prompt() -> MultifilePromptTemplate:
    """Prompt for focused feature engineering improvements."""
    
    template = """{system_prompt}

# TASK DESCRIPTION
```
{task_description}
```

# DATA OVERVIEW
```
{data_overview}
```

# CURRENT WORKING DIRECTORY STATE
{file_summary}

# CURRENT FEATURE ENGINEERING
{current_features}

**TASK**: Improve the feature engineering approach to boost model performance.

**FEATURE ENGINEERING FOCUS AREAS**:
{feature_suggestions}

**REQUIREMENTS**:
1. Enhance existing feature engineering code
2. Add new feature engineering techniques
3. Create a separate features.py module if the approach becomes complex
4. Ensure features work with the existing pipeline

**RESPONSE FORMAT**:
## Edit: solution.py
```python
# Updated solution with improved features
```

Or if creating a separate module:

## Create: features.py
```python
# Feature engineering functions
```

## Edit: solution.py
```python
# Updated solution importing from features.py
```"""

    return MultifilePromptTemplate(template)


def get_model_tuning_prompt() -> MultifilePromptTemplate:
    """Prompt for model selection and hyperparameter tuning."""
    
    template = """{system_prompt}

# TASK DESCRIPTION
```
{task_description}
```

# CURRENT WORKING DIRECTORY STATE
{file_summary}

# CURRENT MODEL APPROACH
{current_model}

# PERFORMANCE RESULTS
{current_performance}

**TASK**: Improve model selection and hyperparameter tuning.

**MODEL IMPROVEMENT SUGGESTIONS**:
{model_suggestions}

**REQUIREMENTS**:
1. Enhance existing model or try different algorithms
2. Add proper hyperparameter tuning
3. Consider ensemble methods if appropriate
4. Maintain cross-validation methodology

**RESPONSE FORMAT**:
## Edit: solution.py
```python
# Updated solution with improved modeling
```

Or if creating separate modules:

## Create: models.py
```python
# Model definitions and tuning functions
```

## Edit: solution.py
```python
# Updated solution using models.py
```"""

    return MultifilePromptTemplate(template)


class MultifilePromptBuilder:
    """Builds prompts based on current working directory state and iteration context."""
    
    def __init__(self, file_manager: FileStateManager, verbose: bool = True):
        self.file_manager = file_manager
        self.verbose = verbose
    
    def build_prompt(self, prompt_type: str, task_description: str, 
                    data_overview: str, **kwargs) -> str:
        """
        Build appropriate prompt based on type and context.
        
        Args:
            prompt_type: Type of prompt ('initial', 'improve', 'debug', 'feature', 'model')
            task_description: Description of the ML task
            data_overview: Overview of available data
            **kwargs: Additional context for specific prompt types
            
        Returns:
            Formatted prompt string
        """
        # Get current file state
        self.file_manager.scan_directory()
        file_summary = self.file_manager.get_file_summary()
        
        # Base context
        context = {
            'system_prompt': get_system_prompt_multifile(),
            'task_description': task_description,
            'data_overview': data_overview,
            'file_summary': file_summary,
        }
        
        # Add additional context based on prompt type
        if prompt_type == 'initial':
            template = get_initial_setup_prompt()
            
        elif prompt_type == 'improve':
            template = get_improvement_prompt()
            context.update({
                'memory': kwargs.get('memory', ''),
                'current_files': self._get_current_files_content(),
                'current_analysis': self._analyze_current_approach(),
                'improvement_suggestions': self._suggest_improvements()
            })
            
        elif prompt_type == 'debug':
            template = get_debugging_prompt()
            context.update({
                'current_files': self._get_current_files_content(),
                'error_details': kwargs.get('error_details', ''),
                'error_analysis': self._analyze_error(kwargs.get('error_details', ''))
            })
            
        elif prompt_type == 'feature':
            template = get_feature_engineering_prompt()
            context.update({
                'current_features': self._analyze_current_features(),
                'feature_suggestions': self._suggest_feature_improvements()
            })
            
        elif prompt_type == 'model':
            template = get_model_tuning_prompt()
            context.update({
                'current_model': self._analyze_current_model(),
                'current_performance': kwargs.get('performance', ''),
                'model_suggestions': self._suggest_model_improvements()
            })
            
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        return template.format(**context)
    
    def _get_current_files_content(self, max_lines_per_file: int = 100) -> str:
        """Get content of key files in working directory."""
        content_parts = []
        
        # Priority order for showing files
        priority_files = ['requirements.txt', 'solution.py', 'main.py', 'utils.py', 'features.py', 'models.py']
        
        shown_files = set()
        
        # Show priority files first
        for filename in priority_files:
            content = self.file_manager.get_file_content(filename, max_lines_per_file)
            if content:
                content_parts.append(f"## {filename}\n```python\n{content}\n```\n")
                shown_files.add(filename)
        
        # Show other Python files
        for rel_path in sorted(self.file_manager.file_cache.keys()):
            if rel_path not in shown_files and rel_path.endswith('.py'):
                content = self.file_manager.get_file_content(rel_path, max_lines_per_file)
                if content:
                    content_parts.append(f"## {rel_path}\n```python\n{content}\n```\n")
        
        return "\n".join(content_parts) if content_parts else "No files to show."
    
    def _analyze_current_approach(self) -> str:
        """Analyze the current approach in existing files."""
        python_files = [path for path, info in self.file_manager.file_cache.items() if info.is_python]
        
        if not python_files:
            return "No Python files found - starting from scratch."
        
        analysis_parts = []
        
        # Check for main solution file
        solution_files = [f for f in python_files if 'solution' in f.lower() or 'main' in f.lower()]
        if solution_files:
            analysis_parts.append(f"Main solution in: {', '.join(solution_files)}")
        
        # Check for modular structure
        if len(python_files) > 1:
            analysis_parts.append(f"Modular structure with {len(python_files)} Python files")
        
        # Check dependencies
        dependencies = self.file_manager.get_python_dependencies()
        ml_libs = dependencies & {'sklearn', 'pandas', 'numpy', 'xgboost', 'lightgbm', 'tensorflow', 'torch'}
        if ml_libs:
            analysis_parts.append(f"Using ML libraries: {', '.join(sorted(ml_libs))}")
        
        return ". ".join(analysis_parts) if analysis_parts else "Basic Python structure detected."
    
    def _suggest_improvements(self) -> str:
        """Suggest improvements based on current state."""
        suggestions = self.file_manager.suggest_next_files()
        
        # Add ML-specific suggestions
        ml_suggestions = []
        
        dependencies = self.file_manager.get_python_dependencies()
        
        # Check for advanced ML techniques
        if 'sklearn' in dependencies and 'xgboost' not in dependencies:
            ml_suggestions.append("Try ensemble methods like XGBoost or Random Forest")
        
        if 'pandas' in dependencies and 'numpy' not in dependencies:
            ml_suggestions.append("Add numpy for efficient numerical operations")
        
        all_suggestions = suggestions + ml_suggestions
        return "\n".join(f"- {s}" for s in all_suggestions) if all_suggestions else "Current structure looks good."
    
    def _analyze_error(self, error_details: str) -> str:
        """Analyze error details and provide context."""
        if not error_details:
            return "No error details provided."
        
        analysis = []
        
        if "ModuleNotFoundError" in error_details:
            analysis.append("Missing dependency - check requirements.txt")
        
        if "FileNotFoundError" in error_details:
            analysis.append("File path issue - check data loading code")
        
        if "SyntaxError" in error_details:
            analysis.append("Python syntax error - check code structure")
        
        if "ValueError" in error_details and "n_splits" in error_details:
            analysis.append("Cross-validation issue - dataset may be too small")
        
        return ". ".join(analysis) if analysis else "General execution error."
    
    def _analyze_current_features(self) -> str:
        """Analyze current feature engineering approach."""
        # This would be more sophisticated in practice
        return "Basic feature loading detected - consider advanced feature engineering."
    
    def _suggest_feature_improvements(self) -> str:
        """Suggest feature engineering improvements."""
        return """- Text features: TF-IDF, word embeddings, character n-grams
- Categorical encoding: target encoding, frequency encoding
- Numerical features: scaling, binning, polynomial features
- Feature selection: mutual information, recursive feature elimination"""
    
    def _analyze_current_model(self) -> str:
        """Analyze current modeling approach."""
        dependencies = self.file_manager.get_python_dependencies()
        
        if 'sklearn' in dependencies:
            return "Using scikit-learn models"
        elif 'xgboost' in dependencies:
            return "Using XGBoost models"
        elif 'tensorflow' in dependencies or 'torch' in dependencies:
            return "Using deep learning models"
        else:
            return "No clear modeling approach detected"
    
    def _suggest_model_improvements(self) -> str:
        """Suggest model improvements."""
        return """- Try ensemble methods (Random Forest, XGBoost, LightGBM)
- Implement proper hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- Consider model stacking or blending
- Add feature importance analysis"""