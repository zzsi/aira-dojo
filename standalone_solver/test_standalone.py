#!/usr/bin/env python3
"""
Test script for the standalone solver components.
"""

import os
import sys
import tempfile
from pathlib import Path

def test_data_setup():
    """Test data setup functionality."""
    print("Testing data setup...")
    
    from data_setup import setup_working_directory, prepare_spooky_author_data, create_instructions_file
    
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = setup_working_directory(temp_dir)
        prepare_spooky_author_data(work_dir)
        create_instructions_file(work_dir)
        
        # Check files exist
        assert (work_dir / "data" / "train.csv").exists()
        assert (work_dir / "data" / "test.csv").exists()
        assert (work_dir / "data" / "description.md").exists()
        assert (work_dir / "instructions.txt").exists()
        
        print("✓ Data setup test passed")

def test_claude_interface():
    """Test Claude interface."""
    print("Testing Claude interface...")
    
    from claude_interface import ClaudeInterface, extract_code_from_response, validate_python_code
    
    # Test code extraction
    sample_response = """Here's a solution:

# Simple approach
This uses basic classification.

```python
import pandas as pd
print("Hello world")
```"""
    
    code, explanation = extract_code_from_response(sample_response)
    assert code == 'import pandas as pd\nprint("Hello world")'
    assert "Simple approach" in explanation
    
    # Test code validation
    is_valid, error = validate_python_code("print('hello')")
    assert is_valid
    assert error is None
    
    is_valid, error = validate_python_code("invalid syntax here!")
    assert not is_valid
    assert error is not None
    
    print("✓ Claude interface test passed")

def test_prompt_templates():
    """Test prompt templates."""
    print("Testing prompt templates...")
    
    from prompt_templates import prepare_draft_prompt, generate_data_overview
    
    # Test prompt generation
    prompt = prepare_draft_prompt(
        iteration=1,
        task_description="Test task",
        data_overview="Test data"
    )
    
    assert "Kaggle Grandmaster" in prompt
    assert "Test task" in prompt
    assert "SIMPLE IDEA" in prompt  # First iteration should be simple
    
    print("✓ Prompt templates test passed")

def test_code_evaluator():
    """Test code evaluator."""
    print("Testing code evaluator...")
    
    from code_evaluator import CodeEvaluator
    
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir)
        
        # Create minimal data structure
        data_dir = work_dir / "data"
        data_dir.mkdir()
        (data_dir / "train.csv").write_text("id,text,author\n1,hello,A")
        (data_dir / "test.csv").write_text("id,text\n2,world")
        
        evaluator = CodeEvaluator(work_dir, timeout_secs=30)
        
        # Test successful code
        good_code = """
import pandas as pd
print("CV Score: 0.85")
df = pd.DataFrame({'id': [2], 'author': ['A']})
df.to_csv('submission.csv', index=False)
"""
        
        result = evaluator.execute_code(good_code)
        assert result["success"]
        assert result["submission_exists"]
        assert result["cv_score"] == 0.85
        
        print("✓ Code evaluator test passed")

def test_integration():
    """Test basic integration."""
    print("Testing integration...")
    
    # Just test that imports work
    try:
        from outer_loop import StandaloneSolver
        print("✓ Integration test passed (imports successful)")
    except ImportError as e:
        print(f"✗ Integration test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🧪 Testing Standalone Solver Components")
    print("=" * 50)
    
    tests = [
        test_data_setup,
        test_claude_interface, 
        test_prompt_templates,
        test_code_evaluator,
        test_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 50)
    print(f"Tests: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)