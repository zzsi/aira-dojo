#!/usr/bin/env python3
"""
Improved code evaluator with data aliasing and Claude journaling.
"""

import os
import sys
import subprocess
import tempfile
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

class ImprovedCodeEvaluator:
    """Enhanced evaluator with data aliasing and journaling."""
    
    def __init__(self, work_dir: Path, timeout_secs: int = 600, use_data_aliases: bool = True):
        """
        Initialize improved evaluator.
        
        Args:
            work_dir: Working directory containing data files
            timeout_secs: Maximum execution time for code
            use_data_aliases: Whether to use symlinks instead of copying data
        """
        self.work_dir = Path(work_dir)
        self.timeout_secs = timeout_secs
        self.evaluation_count = 0
        self.use_data_aliases = use_data_aliases
        
        # Create journals directory
        self.journals_dir = self.work_dir / "journals"
        self.journals_dir.mkdir(parents=True, exist_ok=True)
    
    def log_claude_interaction(self, prompt: str, response: str, metadata: Dict[str, Any]):
        """
        Log Claude interaction to timestamped journal.
        
        Args:
            prompt: The prompt sent to Claude
            response: Claude's response
            metadata: Metadata about the interaction
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        journal_file = self.journals_dir / f"claude_session_{timestamp}.md"
        
        journal_content = f"""# Claude Interaction - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Metadata
- **Call Number**: {metadata.get('call_number', 'N/A')}
- **Model**: {metadata.get('model', 'sonnet')}
- **Latency**: {metadata.get('latency', 0):.2f}s
- **Prompt Length**: {len(prompt)} characters
- **Response Length**: {len(response)} characters
- **Success**: {metadata.get('return_code', -1) == 0}

## Prompt Sent to Claude
```
{prompt}
```

## Claude's Response
{response}

## Analysis
- **Code Extracted**: {"Yes" if self._contains_code_block(response) else "No"}
- **Explanation Quality**: {self._assess_explanation_quality(response)}
- **Technical Depth**: {self._assess_technical_depth(response)}

---
"""
        
        journal_file.write_text(journal_content)
        print(f"[Journal] Claude interaction logged to {journal_file.name}")
    
    def _contains_code_block(self, text: str) -> bool:
        """Check if text contains code blocks."""
        return "```python" in text or "```" in text
    
    def _assess_explanation_quality(self, text: str) -> str:
        """Simple heuristic to assess explanation quality."""
        explanation_part = text.split("```")[0] if "```" in text else text
        word_count = len(explanation_part.split())
        
        if word_count > 100:
            return "Detailed"
        elif word_count > 50:
            return "Moderate" 
        else:
            return "Brief"
    
    def _assess_technical_depth(self, text: str) -> str:
        """Assess technical depth of response."""
        technical_terms = [
            "cross-validation", "feature", "model", "algorithm", "preprocessing",
            "pipeline", "vectorization", "classification", "regression", "optimization"
        ]
        
        text_lower = text.lower()
        term_count = sum(1 for term in technical_terms if term in text_lower)
        
        if term_count >= 5:
            return "High"
        elif term_count >= 3:
            return "Medium"
        else:
            return "Low"
    
    def setup_data_access(self, temp_path: Path) -> bool:
        """
        Set up data access in temp directory using aliases or copies.
        
        Args:
            temp_path: Temporary execution directory
            
        Returns:
            True if setup successful, False otherwise
        """
        data_src = self.work_dir / "data"
        data_dst = temp_path / "data"
        
        if not data_src.exists():
            print(f"[Error] Source data directory not found: {data_src}")
            return False
        
        try:
            if self.use_data_aliases and os.name != 'nt':  # Unix-like systems
                # Create symlink to original data
                data_dst.symlink_to(data_src.resolve())
                print(f"[Data] Created symlink: {data_dst} -> {data_src}")
            else:
                # Fall back to copying for Windows or when aliases disabled
                import shutil
                shutil.copytree(data_src, data_dst)
                print(f"[Data] Copied data: {data_src} -> {data_dst}")
            
            return True
            
        except Exception as e:
            print(f"[Error] Failed to setup data access: {e}")
            return False
    
    def execute_code_with_journaling(self, code: str, claude_prompt: str, claude_response: str, 
                                   claude_metadata: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
        """
        Execute code with full journaling of Claude interaction.
        
        Args:
            code: Python code to execute
            claude_prompt: Original prompt sent to Claude
            claude_response: Claude's full response
            claude_metadata: Claude interaction metadata
            verbose: Whether to print execution details
            
        Returns:
            Dictionary with execution results
        """
        self.evaluation_count += 1
        
        # Log Claude interaction first
        self.log_claude_interaction(claude_prompt, claude_response, claude_metadata)
        
        if verbose:
            print(f"[Execution {self.evaluation_count}] Starting code execution...")
        
        # Create temporary execution directory  
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Setup data access (symlink or copy)
            if not self.setup_data_access(temp_path):
                return self._create_error_result("Failed to setup data access", self.evaluation_count)
            
            # Write code to temporary file
            code_file = temp_path / "solution.py"
            code_file.write_text(code)
            
            if verbose:
                print(f"[Execution {self.evaluation_count}] Code written to {code_file}")
                print(f"[Execution {self.evaluation_count}] Data access via {'symlink' if self.use_data_aliases else 'copy'}")
            
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
                
                # Handle submission file
                submission_file = temp_path / "submission.csv"
                submission_exists = submission_file.exists()
                submission_content = ""
                
                if submission_exists:
                    submission_content = submission_file.read_text()
                    if verbose:
                        print(f"[Execution {self.evaluation_count}] Submission file created ({len(submission_content)} chars)")
                
                # Save artifacts
                self._save_enhanced_artifacts(code, submission_content, result.stdout, result.stderr, claude_response, claude_metadata)
                
                # Parse CV score
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
                    "evaluation_number": self.evaluation_count,
                    "data_access_method": "symlink" if self.use_data_aliases else "copy"
                }
                
                if verbose:
                    status = "SUCCESS" if execution_result["success"] else "FAILED"
                    print(f"[Execution {self.evaluation_count}] {status} ({execution_time:.2f}s)")
                    if cv_score is not None:
                        print(f"[Execution {self.evaluation_count}] CV Score: {cv_score}")
                    elif not execution_result["success"]:
                        # Print error details for failed executions
                        if result.stderr:
                            print(f"[Execution {self.evaluation_count}] STDERR: {result.stderr[:500]}...")
                        if result.stdout:
                            print(f"[Execution {self.evaluation_count}] STDOUT: {result.stdout[:300]}...")
                        print(f"[Execution {self.evaluation_count}] Return code: {result.returncode}")
                
                return execution_result
                
            except subprocess.TimeoutExpired:
                execution_time = time.time() - start_time
                return self._create_timeout_result(execution_time, self.evaluation_count)
                
            except Exception as e:
                execution_time = time.time() - start_time
                return self._create_exception_result(str(e), execution_time, self.evaluation_count)
    
    def _save_enhanced_artifacts(self, code: str, submission_content: str, stdout: str, stderr: str,
                               claude_response: str, claude_metadata: Dict[str, Any]):
        """Save enhanced artifacts including Claude's full response."""
        artifacts_dir = self.work_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        eval_num = self.evaluation_count
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save generated code
        code_file = artifacts_dir / f"solution_{eval_num}_{timestamp}.py"
        code_file.write_text(code)
        
        # Save Claude's full response
        claude_file = artifacts_dir / f"claude_response_{eval_num}_{timestamp}.md"
        claude_content = f"""# Claude Response - Iteration {eval_num}

## Metadata
- **Timestamp**: {timestamp}
- **Latency**: {claude_metadata.get('latency', 0):.2f}s
- **Model**: sonnet
- **Success**: {claude_metadata.get('return_code', -1) == 0}

## Full Response
{claude_response}
"""
        claude_file.write_text(claude_content)
        
        # Save submission if exists
        if submission_content:
            submission_file = artifacts_dir / f"submission_{eval_num}_{timestamp}.csv"
            submission_file.write_text(submission_content)
            
            # Update latest submission
            latest_submission = self.work_dir / "submission.csv"
            latest_submission.write_text(submission_content)
        
        # Save execution stdout
        stdout_file = artifacts_dir / f"stdout_{eval_num}_{timestamp}.log"
        stdout_file.write_text(stdout)
        
        # Save execution stderr (contains full stack traces)
        if stderr.strip():
            stderr_file = artifacts_dir / f"stderr_{eval_num}_{timestamp}.log"
            stderr_file.write_text(stderr)
            print(f"[Artifacts] Saved error details to stderr_{eval_num}_{timestamp}.log")
        
        print(f"[Artifacts] Saved iteration {eval_num} artifacts with timestamp {timestamp}")
    
    def _create_error_result(self, error_msg: str, eval_num: int) -> Dict[str, Any]:
        """Create error result dictionary."""
        return {
            "success": False,
            "execution_time": 0,
            "return_code": -1,
            "stdout": "",
            "stderr": error_msg,
            "submission_exists": False,
            "submission_content": "",
            "cv_score": None,
            "evaluation_number": eval_num
        }
    
    def _create_timeout_result(self, execution_time: float, eval_num: int) -> Dict[str, Any]:
        """Create timeout result dictionary."""
        return {
            "success": False,
            "execution_time": execution_time,
            "return_code": 124,
            "stdout": "",
            "stderr": f"Execution timed out after {self.timeout_secs} seconds",
            "submission_exists": False,
            "submission_content": "",
            "cv_score": None,
            "evaluation_number": eval_num,
            "timeout": True
        }
    
    def _create_exception_result(self, exception_str: str, execution_time: float, eval_num: int) -> Dict[str, Any]:
        """Create exception result dictionary."""
        return {
            "success": False,
            "execution_time": execution_time,
            "return_code": -1,
            "stdout": "",
            "stderr": f"Execution error: {exception_str}",
            "submission_exists": False,
            "submission_content": "",
            "cv_score": None,
            "evaluation_number": eval_num,
            "exception": exception_str
        }
    
    def _extract_cv_score(self, output: str) -> Optional[float]:
        """Extract cross-validation score from output text."""
        if not output:
            return None
        
        import re
        patterns = [
            r"(?:cross.validation|cv|cross.val).*?score.*?[:\s]+([\d\.]+)",
            r"mean.*?(?:cv|accuracy).*?[:\s]+([\d\.]+)",
            r"score.*?[:\s]+([\d\.]+)",
            r"accuracy.*?[:\s]+([\d\.]+)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, output.lower())
            if matches:
                try:
                    return float(matches[-1])
                except ValueError:
                    continue
        
        return None

if __name__ == "__main__":
    # Test the improved evaluator
    work_dir = Path("output/working")
    
    if not work_dir.exists():
        print("Please run data_setup.py first to create working directory")
        sys.exit(1)
    
    evaluator = ImprovedCodeEvaluator(work_dir, use_data_aliases=True)
    
    # Test data access setup
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        success = evaluator.setup_data_access(temp_path)
        print(f"Data access setup: {'SUCCESS' if success else 'FAILED'}")
        
        if success:
            data_files = list((temp_path / "data").glob("*"))
            print(f"Accessible data files: {[f.name for f in data_files]}")