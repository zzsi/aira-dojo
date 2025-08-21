#!/usr/bin/env python3
"""
Python code evaluation system for the standalone solver.
Executes and evaluates generated solutions.
"""

import os
import sys
import subprocess
import tempfile
import time
import traceback
import re
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

class CodeEvaluator:
    """Evaluates Python code in a controlled environment."""
    
    def __init__(self, work_dir: Path, timeout_secs: int = 600):
        """
        Initialize code evaluator.
        
        Args:
            work_dir: Working directory containing data files
            timeout_secs: Maximum execution time for code
        """
        self.work_dir = Path(work_dir)
        self.timeout_secs = timeout_secs
        self.evaluation_count = 0
    
    def execute_code(self, code: str, verbose: bool = False, save_artifacts: bool = True) -> Dict[str, Any]:
        """
        Execute Python code and capture results.
        
        Args:
            code: Python code to execute
            verbose: Whether to print execution details
            save_artifacts: Whether to save generated code and submission to working directory
            
        Returns:
            Dictionary with execution results
        """
        self.evaluation_count += 1
        
        if verbose:
            print(f"[Execution {self.evaluation_count}] Starting code execution...")
        
        # Create temporary execution directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Copy data directory to temp location
            import shutil
            data_src = self.work_dir / "data"
            data_dst = temp_path / "data"
            if data_src.exists():
                shutil.copytree(data_src, data_dst)
            
            # Write code to temporary file
            code_file = temp_path / "solution.py"
            code_file.write_text(code)
            
            if verbose:
                print(f"[Execution {self.evaluation_count}] Code written to {code_file}")
            
            # Execute code
            start_time = time.time()
            
            try:
                result = subprocess.run(
                    [sys.executable, str(code_file)],
                    cwd=temp_path,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_secs
                )
                
                execution_time = time.time() - start_time
                
                # Check for submission file
                submission_file = temp_path / "submission.csv"
                submission_exists = submission_file.exists()
                submission_content = ""
                
                if submission_exists:
                    submission_content = submission_file.read_text()
                    if verbose:
                        print(f"[Execution {self.evaluation_count}] Submission file created ({len(submission_content)} chars)")
                
                # Save artifacts to working directory if requested
                if save_artifacts:
                    self._save_artifacts(code, submission_content if submission_exists else "", result.stdout)
                
                # Parse cross-validation score from output
                cv_score = self._extract_cv_score(result.stdout)
                
                execution_result = {
                    "success": result.returncode == 0,
                    "execution_time": execution_time,
                    "return_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "submission_exists": submission_exists,
                    "submission_content": submission_content,
                    "cv_score": cv_score,
                    "evaluation_number": self.evaluation_count
                }
                
                if verbose:
                    status = "SUCCESS" if execution_result["success"] else "FAILED"
                    print(f"[Execution {self.evaluation_count}] {status} ({execution_time:.2f}s)")
                    if cv_score is not None:
                        print(f"[Execution {self.evaluation_count}] CV Score: {cv_score}")
                
                return execution_result
                
            except subprocess.TimeoutExpired:
                execution_time = time.time() - start_time
                
                if verbose:
                    print(f"[Execution {self.evaluation_count}] TIMEOUT after {self.timeout_secs}s")
                
                return {
                    "success": False,
                    "execution_time": execution_time,
                    "return_code": 124,
                    "stdout": "",
                    "stderr": f"Execution timed out after {self.timeout_secs} seconds",
                    "submission_exists": False,
                    "submission_content": "",
                    "cv_score": None,
                    "evaluation_number": self.evaluation_count,
                    "timeout": True
                }
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                if verbose:
                    print(f"[Execution {self.evaluation_count}] ERROR: {e}")
                
                return {
                    "success": False,
                    "execution_time": execution_time,
                    "return_code": -1,
                    "stdout": "",
                    "stderr": f"Execution error: {str(e)}",
                    "submission_exists": False,
                    "submission_content": "",
                    "cv_score": None,
                    "evaluation_number": self.evaluation_count,
                    "exception": str(e)
                }
    
    def _save_artifacts(self, code: str, submission_content: str, stdout: str):
        """
        Save generated artifacts to working directory.
        
        Args:
            code: The Python code that was executed
            submission_content: Contents of submission file (if any)
            stdout: Standard output from execution
        """
        # Create artifacts directory
        artifacts_dir = self.work_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        # Save the generated code
        code_file = artifacts_dir / f"solution_{self.evaluation_count}.py"
        code_file.write_text(code)
        
        # Save submission if it exists
        if submission_content:
            submission_file = artifacts_dir / f"submission_{self.evaluation_count}.csv"
            submission_file.write_text(submission_content)
            
            # Also save as latest submission
            latest_submission = self.work_dir / "submission.csv"
            latest_submission.write_text(submission_content)
        
        # Save execution log
        log_file = artifacts_dir / f"execution_{self.evaluation_count}.log"
        log_file.write_text(stdout)
        
        print(f"[Artifacts] Saved to {artifacts_dir}/")

    def _extract_cv_score(self, output: str) -> Optional[float]:
        """
        Extract cross-validation score from output text.
        
        Args:
            output: stdout from code execution
            
        Returns:
            Extracted CV score or None if not found
        """
        if not output:
            return None
        
        # Common patterns for CV scores
        patterns = [
            r"(?:cross.validation|cv|cross.val).*?score.*?[:\s]+([\d\.]+)",
            r"score.*?[:\s]+([\d\.]+)",
            r"accuracy.*?[:\s]+([\d\.]+)",
            r"5.fold.*?[:\s]+([\d\.]+)",
            r"([\d\.]+).*?accuracy",
            r"([\d\.]+).*?score"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, output.lower())
            if matches:
                try:
                    return float(matches[-1])  # Take the last match
                except ValueError:
                    continue
        
        return None
    
    def analyze_performance(self, execution_result: Dict[str, Any]) -> str:
        """
        Analyze execution results and provide performance analysis.
        
        Args:
            execution_result: Results from code execution
            
        Returns:
            Performance analysis string
        """
        analysis_lines = []
        
        # Execution status
        if execution_result["success"]:
            analysis_lines.append("✓ Code executed successfully")
        else:
            analysis_lines.append("✗ Code execution failed")
            analysis_lines.append(f"  Return code: {execution_result['return_code']}")
        
        # Execution time
        exec_time = execution_result["execution_time"]
        if exec_time < 60:
            analysis_lines.append(f"✓ Execution time: {exec_time:.2f}s (fast)")
        elif exec_time < 300:
            analysis_lines.append(f"◐ Execution time: {exec_time:.2f}s (moderate)")
        else:
            analysis_lines.append(f"◯ Execution time: {exec_time:.2f}s (slow)")
        
        # Submission file
        if execution_result["submission_exists"]:
            analysis_lines.append("✓ Submission file created")
            
            # Analyze submission content
            submission = execution_result["submission_content"]
            if submission:
                lines = submission.strip().split('\n')
                analysis_lines.append(f"  Submission has {len(lines)} lines")
                
                # Check for proper CSV format
                if lines and ',' in lines[0]:
                    analysis_lines.append("  ✓ CSV format detected")
                else:
                    analysis_lines.append("  ◯ CSV format unclear")
        else:
            analysis_lines.append("✗ No submission file created")
        
        # Cross-validation score
        if execution_result["cv_score"] is not None:
            cv_score = execution_result["cv_score"]
            analysis_lines.append(f"✓ CV Score: {cv_score:.4f}")
            
            # Rough performance assessment
            if cv_score > 0.8:
                analysis_lines.append("  ✓ High performance")
            elif cv_score > 0.6:
                analysis_lines.append("  ◐ Moderate performance")
            else:
                analysis_lines.append("  ◯ Low performance")
        else:
            analysis_lines.append("◯ No CV score detected in output")
        
        # Error analysis
        if execution_result["stderr"]:
            error_lines = execution_result["stderr"].strip().split('\n')
            analysis_lines.append(f"◯ Stderr has {len(error_lines)} lines")
            
            # Check for common error types
            stderr_lower = execution_result["stderr"].lower()
            if "import" in stderr_lower and "error" in stderr_lower:
                analysis_lines.append("  → Possible missing dependency")
            elif "file" in stderr_lower and "not found" in stderr_lower:
                analysis_lines.append("  → Possible file path issue")
            elif "memory" in stderr_lower:
                analysis_lines.append("  → Possible memory issue")
            elif "syntax" in stderr_lower:
                analysis_lines.append("  → Syntax error in generated code")
        
        return "\n".join(analysis_lines)
    
    def validate_submission(self, execution_result: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate the submission file format.
        
        Args:
            execution_result: Results from code execution
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not execution_result["submission_exists"]:
            return False, "No submission file created"
        
        submission_content = execution_result["submission_content"]
        if not submission_content.strip():
            return False, "Submission file is empty"
        
        lines = submission_content.strip().split('\n')
        
        # Check minimum number of lines
        if len(lines) < 2:
            return False, "Submission must have at least a header and one data row"
        
        # Check CSV format
        header = lines[0]
        if ',' not in header:
            return False, "Submission must be in CSV format"
        
        # For spooky author task, check expected columns
        header_lower = header.lower()
        if "id" not in header_lower or "author" not in header_lower:
            return False, "Submission must have 'id' and 'author' columns"
        
        # Check data rows
        data_rows = lines[1:]
        for i, row in enumerate(data_rows[:3]):  # Check first 3 rows
            if ',' not in row:
                return False, f"Row {i+2} is not in proper CSV format"
        
        return True, f"Valid submission with {len(data_rows)} predictions"

def test_code_evaluator():
    """Test the code evaluator with sample code."""
    from pathlib import Path
    
    # Create a temporary working directory for testing
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir)
        
        # Create sample data structure
        data_dir = work_dir / "data"
        data_dir.mkdir()
        
        # Create sample train.csv
        train_content = """id,text,author
1,"Hello world",EAP
2,"Test text",HPL"""
        (data_dir / "train.csv").write_text(train_content)
        
        # Create sample test.csv
        test_content = """id,text
3,"Another test"
4,"Final test" """
        (data_dir / "test.csv").write_text(test_content)
        
        # Test code that should work
        test_code = """
import pandas as pd

# Load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print(f"Loaded {len(train)} training samples")
print(f"Cross-validation score: 0.8500")

# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'author': ['EAP'] * len(test)
})

submission.to_csv('submission.csv', index=False)
print("Submission saved")
"""
        
        evaluator = CodeEvaluator(work_dir)
        result = evaluator.execute_code(test_code, verbose=True)
        
        print("\nExecution result:")
        for key, value in result.items():
            if key in ["stdout", "stderr", "submission_content"]:
                print(f"{key}: {len(str(value))} chars")
            else:
                print(f"{key}: {value}")
        
        print("\nPerformance analysis:")
        analysis = evaluator.analyze_performance(result)
        print(analysis)
        
        print("\nSubmission validation:")
        is_valid, message = evaluator.validate_submission(result)
        print(f"Valid: {is_valid}, Message: {message}")

if __name__ == "__main__":
    test_code_evaluator()