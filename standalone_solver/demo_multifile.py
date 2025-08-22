#!/usr/bin/env python3
"""
Demo script showing how the multifile solver would work.
This creates a mock scenario without actually calling Claude.
"""

import tempfile
import shutil
from pathlib import Path
from file_state_manager import FileStateManager
from multifile_prompts import MultifilePromptBuilder


def demo_multifile_workflow():
    """Demo the multifile solver workflow."""
    print("🎬 Multifile Solver Workflow Demo")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir)
        print(f"📁 Working directory: {work_dir}")
        
        # Setup initial task data
        data_dir = work_dir / "data"
        data_dir.mkdir()
        
        (data_dir / "train.csv").write_text("""id,text,author
1,"Once upon a midnight dreary",EAP
2,"It was the best of times",MWS
3,"In the mountains of madness",HPL""")
        
        (data_dir / "test.csv").write_text("""id,text
4,"The fall of the house"
5,"Call me Ishmael" """)
        
        (work_dir / "instructions.txt").write_text("""
Spooky Author Identification
Predict the author of horror stories based on text content.
Authors: EAP (Edgar Allan Poe), HPL (H.P. Lovecraft), MWS (Mary Shelley)
""")
        
        # Initialize components
        fm = FileStateManager(work_dir, verbose=True)
        pb = MultifilePromptBuilder(fm, verbose=True)
        
        print("\n🔍 Initial file scan:")
        files = fm.scan_directory()
        print(f"Found {len(files)} files")
        
        print("\n📋 File summary:")
        summary = fm.get_file_summary()
        print(summary)
        
        print("\n🤖 Generate initial setup prompt:")
        initial_prompt = pb.build_prompt(
            prompt_type='initial',
            task_description="Spooky Author Identification - predict author from text",
            data_overview="train.csv (3 samples), test.csv (2 samples)"
        )
        
        print(f"Prompt length: {len(initial_prompt)} characters")
        print("Key sections in prompt:")
        for section in ["TASK DESCRIPTION", "RESPONSE FORMAT", "requirements.txt"]:
            if section in initial_prompt:
                print(f"  ✅ {section}")
        
        print("\n📝 Simulate Claude creating initial files:")
        
        # Simulate Claude's response for initial setup - use separate strings to avoid syntax issues
        requirements_content = """pandas>=1.3.0
scikit-learn>=1.0.0
numpy>=1.20.0"""
        
        solution_content = """import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.pipeline import Pipeline

def main():
    # Load data
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    # Prepare features and target
    X = train_df['text']
    y = train_df['author']
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    # Use LeaveOneOut CV for small dataset
    cv_scores = cross_val_score(pipeline, X, y, cv=LeaveOneOut(), scoring='accuracy')
    mean_score = cv_scores.mean()
    
    print(f"Cross-validation accuracy: {mean_score:.4f}")
    
    # Train on full dataset and predict
    pipeline.fit(X, y)
    test_predictions = pipeline.predict(test_df['text'])
    
    # Save submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'author': test_predictions
    })
    submission.to_csv('submission.csv', index=False)
    print(f"Saved predictions to submission.csv")

if __name__ == "__main__":
    main()"""
        
        utils_content = """import pandas as pd
from typing import Dict, Any

def load_data(data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    return {
        'train': pd.read_csv(f"{data_dir}/train.csv"),
        'test': pd.read_csv(f"{data_dir}/test.csv")
    }

def save_submission(predictions, test_ids, filename: str = "submission.csv"):
    submission = pd.DataFrame({
        'id': test_ids,
        'author': predictions
    })
    submission.to_csv(filename, index=False)
    return filename"""
        
        mock_response = f"""I'll set up the initial project structure for this text classification task.

## Create: requirements.txt
```
{requirements_content}
```

## Create: solution.py
```python
{solution_content}
```

## Create: utils.py
```python
{utils_content}
```"""
        
        # Simulate parsing and applying file operations
        from multifile_solver import MultifileSolver
        
        solver = MultifileSolver(str(work_dir), verbose=False)
        operations = solver.parse_file_operations(mock_response)
        modified_files = solver.apply_file_operations(operations)
        
        print(f"Created {len(modified_files)} files: {', '.join(modified_files)}")
        
        print("\n🔄 Rescan after file creation:")
        files = fm.scan_directory()
        print(f"Now have {len(files)} files")
        
        print("\n📊 Updated file summary:")
        summary = fm.get_file_summary()
        print(summary)
        
        print("\n🎯 Generate improvement prompt:")
        improvement_prompt = pb.build_prompt(
            prompt_type='improve',
            task_description="Spooky Author Identification - predict author from text",
            data_overview="train.csv (3 samples), test.csv (2 samples)",
            memory="Initial solution created with TF-IDF + Logistic Regression"
        )
        
        print(f"Improvement prompt length: {len(improvement_prompt)} characters")
        print("Key sections in improvement prompt:")
        for section in ["CURRENT FILE CONTENTS", "IMPROVEMENT SUGGESTIONS", "Edit:"]:
            if section in improvement_prompt:
                print(f"  ✅ {section}")
        
        print("\n🌳 Tree search structure would continue from here...")
        print("Next steps:")
        print("  1. Create git branches for different approaches")
        print("  2. Try feature engineering improvements")
        print("  3. Experiment with different models")
        print("  4. Optimize hyperparameters")
        print("  5. Compare results across branches")
        
        print(f"\n✅ Demo completed successfully!")
        print(f"📁 All files created in: {work_dir}")
        
        # Show final file structure
        print("\n📂 Final file structure:")
        for file_path in sorted(work_dir.rglob("*")):
            if file_path.is_file():
                rel_path = file_path.relative_to(work_dir)
                size = file_path.stat().st_size
                print(f"  {rel_path} ({size} bytes)")


if __name__ == "__main__":
    demo_multifile_workflow()