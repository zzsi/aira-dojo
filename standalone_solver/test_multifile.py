#!/usr/bin/env python3
"""
Test script for the multifile solver components.
"""

import tempfile
import shutil
from pathlib import Path
from file_state_manager import FileStateManager
from multifile_prompts import MultifilePromptBuilder


def test_file_state_manager():
    """Test FileStateManager functionality."""
    print("🧪 Testing FileStateManager...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir)
        
        # Create test files
        (work_dir / "solution.py").write_text("""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train_model():
    pass

class ModelTrainer:
    pass
""")
        
        (work_dir / "requirements.txt").write_text("pandas\nnumpy\nscikit-learn\n")
        (work_dir / "data.csv").write_text("col1,col2\n1,2\n3,4\n")
        
        # Test file manager
        fm = FileStateManager(work_dir, verbose=False)
        files = fm.scan_directory()
        
        print(f"  ✅ Detected {len(files)} files")
        
        # Test file summary
        summary = fm.get_file_summary()
        assert "solution.py" in summary
        assert "requirements.txt" in summary
        print("  ✅ File summary generated")
        
        # Test dependency extraction
        deps = fm.get_python_dependencies()
        expected_deps = {"pandas", "numpy", "sklearn"}
        assert expected_deps.issubset(deps)
        print("  ✅ Dependencies extracted correctly")
        
        print("✅ FileStateManager tests passed")


def test_multifile_prompt_builder():
    """Test MultifilePromptBuilder functionality."""
    print("🧪 Testing MultifilePromptBuilder...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir)
        
        # Create test files
        (work_dir / "solution.py").write_text("# Basic solution")
        
        # Test prompt builder
        fm = FileStateManager(work_dir, verbose=False)
        pb = MultifilePromptBuilder(fm, verbose=False)
        
        # Test initial prompt
        prompt = pb.build_prompt(
            prompt_type='initial',
            task_description='Test classification task',
            data_overview='train.csv, test.csv'
        )
        
        assert "requirements.txt" in prompt
        assert "solution.py" in prompt
        print("  ✅ Initial prompt generated")
        
        # Test improvement prompt
        prompt = pb.build_prompt(
            prompt_type='improve',
            task_description='Test classification task',
            data_overview='train.csv, test.csv',
            memory='Previous iteration failed'
        )
        
        assert "TASK" in prompt
        assert "improve" in prompt.lower()
        print("  ✅ Improvement prompt generated")
        
        print("✅ MultifilePromptBuilder tests passed")


def test_file_operations_parsing():
    """Test parsing of file operations from Claude responses."""
    print("🧪 Testing file operations parsing...")
    
    # Import here to avoid circular imports during testing
    from multifile_solver import MultifileSolver
    
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir)
        
        # Create minimal solver for testing
        solver = MultifileSolver(str(work_dir), verbose=False)
        
        # Test response parsing
        test_response = """
I'll create the required files for this ML task.

## Create: requirements.txt
```
pandas
scikit-learn
numpy
```

## Create: solution.py
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def main():
    print("Hello ML!")

if __name__ == "__main__":
    main()
```
"""
        
        operations = solver.parse_file_operations(test_response)
        
        assert len(operations) == 2
        assert operations[0][0] == 'create'
        assert operations[0][1] == 'requirements.txt'
        assert 'pandas' in operations[0][2]
        
        assert operations[1][0] == 'create'
        assert operations[1][1] == 'solution.py'
        assert 'RandomForestClassifier' in operations[1][2]
        
        print("  ✅ File operations parsed correctly")
        
        # Test applying operations
        modified_files = solver.apply_file_operations(operations)
        
        assert len(modified_files) == 2
        assert 'requirements.txt' in modified_files
        assert 'solution.py' in modified_files
        
        # Verify files were created
        assert (work_dir / 'requirements.txt').exists()
        assert (work_dir / 'solution.py').exists()
        
        print("  ✅ File operations applied correctly")
        
        print("✅ File operations tests passed")


def main():
    """Run all tests."""
    print("🚀 Running multifile solver tests...\n")
    
    try:
        test_file_state_manager()
        print()
        
        test_multifile_prompt_builder()
        print()
        
        test_file_operations_parsing()
        print()
        
        print("🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()