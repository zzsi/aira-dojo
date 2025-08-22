#!/usr/bin/env python3
"""
Docker-based code evaluator with CV score extraction and artifact management.
"""

import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from docker_manager import DockerManager
from artifact_manager import ArtifactManager


class DockerEvaluator:
    """Docker-based code execution with journaling and artifact management."""
    
    def __init__(self, work_dir: Path, timeout_secs: int = 600, use_data_aliases: bool = True):
        self.work_dir = Path(work_dir)
        self.timeout_secs = timeout_secs
        self.use_data_aliases = use_data_aliases
        self.evaluation_count = 0
        
        # Initialize components
        self.docker_manager = DockerManager(work_dir, timeout_secs)
        self.artifact_manager = ArtifactManager(work_dir)
        
        # Check Docker availability
        self.docker_available = self.docker_manager.docker_available
        if not self.docker_available:
            raise RuntimeError("Docker is not available. Please install Docker and ensure it's running.")
    
    def execute_code_with_journaling(self, code: str, claude_prompt: str, claude_response: str, 
                                   claude_metadata: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
        """
        Execute code with full journaling - same interface as ImprovedCodeEvaluator.
        
        Args:
            code: Python code to execute
            claude_prompt: Original prompt sent to Claude
            claude_response: Claude's full response
            claude_metadata: Claude interaction metadata
            verbose: Whether to print execution details
            
        Returns:
            Dictionary with execution results
        """
        # Docker should be available (constructor would have failed otherwise)
        assert self.docker_available, "Docker should be available"
        
        # Check if Claude wants to modify dependencies
        if "requirements.txt" in claude_response and "rebuild" in claude_response.lower():
            self._handle_requirements_update(claude_response, verbose)
        
        self.evaluation_count += 1
        
        if verbose:
            print(f"[Execution {self.evaluation_count}] Starting Docker code execution...")
        
        # Create journal entry first
        journal_file = self.artifact_manager.journal_claude_interaction(
            claude_prompt, claude_response, claude_metadata)
        
        # Execute code in Docker
        execution_result = self.docker_manager.execute_in_container(code, verbose)
        
        # Extract CV score from output
        cv_score = self._extract_cv_score(execution_result.get("stdout", ""))
        execution_result["cv_score"] = cv_score
        
        # Determine success
        success = execution_result.get("return_code", -1) == 0 and cv_score is not None
        execution_result["success"] = success
        
        if verbose:
            status = "SUCCESS" if success else "FAILED"
            print(f"[Execution {self.evaluation_count}] {status} ({execution_result.get('execution_time', 0):.2f}s)")
            if cv_score:
                print(f"[Execution {self.evaluation_count}] CV Score: {cv_score}")
            
            # Debug information about why it failed
            if not success:
                if execution_result.get("return_code", -1) != 0:
                    print(f"[Execution {self.evaluation_count}] Debug: Non-zero return code: {execution_result.get('return_code')}")
                if cv_score is None:
                    print(f"[Execution {self.evaluation_count}] Debug: No CV score extracted")
                if execution_result.get("stderr", "").strip():
                    print(f"[Execution {self.evaluation_count}] Debug: Stderr present ({len(execution_result.get('stderr', ''))} chars)")
            
            # Extract data point information from stdout
            data_info = self._extract_data_info(execution_result.get("stdout", ""))
            if data_info:
                print(f"[Execution {self.evaluation_count}] Data: {data_info}")
        
        # Update journal with execution results
        self.artifact_manager.update_journal_with_execution(journal_file, execution_result)
        
        # Save artifacts
        self.artifact_manager.save_execution_artifacts(code, execution_result, verbose)
        
        return execution_result
    
    def _extract_cv_score(self, stdout: str) -> Optional[float]:
        """Extract cross-validation score from stdout."""
        # More specific patterns that look for CV scores
        patterns = [
            r"Mean CV (?:Accuracy|Score):\s*(\d+\.?\d*)",
            r"Cross[- ]?validation.*?(?:score|accuracy)[:\s]*(\d+\.?\d*)",
            r"CV.*?(?:score|accuracy)[:\s]*(\d+\.?\d*)",
            r"(?:Mean|Average).*?(?:score|accuracy)[:\s]*(\d+\.?\d*)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, stdout, re.IGNORECASE)
            if matches:
                try:
                    score = float(matches[-1])  # Take last match
                    if 0 <= score <= 1:
                        return score
                    elif 1 < score <= 100:  # Likely percentage
                        return score / 100
                    # Reject clearly invalid scores (> 100 or negative)
                except ValueError:
                    continue
        
        return None
    
    def _extract_data_info(self, stdout: str) -> Optional[str]:
        """Extract data information from stdout."""
        info_parts = []
        
        # Look for training data shape
        train_match = re.search(r"Training data shape:\s*\((\d+),\s*(\d+)\)", stdout)
        if train_match:
            info_parts.append(f"{train_match.group(1)} training samples")
        
        # Look for test data shape  
        test_match = re.search(r"Test data shape:\s*\((\d+),\s*(\d+)\)", stdout)
        if test_match:
            info_parts.append(f"{test_match.group(1)} test samples")
            
        # Look for CV folds information
        cv_match = re.search(r"(\d+)-[Ff]old\s+CV|n_splits\s*=\s*(\d+)|Cross[- ]?validation.*?(\d+)\s+folds?", stdout)
        if cv_match:
            folds = cv_match.group(1) or cv_match.group(2) or cv_match.group(3)
            info_parts.append(f"{folds}-fold CV")
        
        # Look for author distribution
        author_match = re.search(r"Author distribution:\s*\{[^}]+\}", stdout)
        if author_match:
            info_parts.append("author distribution found")
        
        return ", ".join(info_parts) if info_parts else None
    
    def _handle_requirements_update(self, claude_response: str, verbose: bool = False):
        """Handle Claude's request to update requirements."""
        # Extract requirements from Claude's response
        requirements_section = None
        
        # Look for requirements.txt content in response
        lines = claude_response.split('\n')
        in_requirements = False
        requirements_lines = []
        
        for line in lines:
            if 'requirements.txt' in line and any(marker in line for marker in ['```', 'file:', 'create']):
                in_requirements = True
                continue
            elif in_requirements and '```' in line:
                break
            elif in_requirements:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements_lines.append(line)
        
        if requirements_lines:
            if verbose:
                print(f"🔧 Claude requested additional packages: {', '.join(requirements_lines)}")
            
            # Rebuild Docker image with new requirements
            success = self.docker_manager.rebuild_image_with_requirements(
                requirements_lines, verbose)
            
            if not success:
                print("⚠️ Warning: Failed to rebuild Docker image with new requirements")
    
    def rebuild_image_with_requirements(self, requirements: List[str], verbose: bool = False) -> bool:
        """Rebuild Docker image with additional requirements."""
        return self.docker_manager.rebuild_image_with_requirements(requirements, verbose)


def test_docker_evaluator():
    """Test the Docker evaluator."""
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        work_dir = Path(tmp_dir) / "test_work"
        work_dir.mkdir()
        
        # Create test data
        data_dir = work_dir / "data"
        data_dir.mkdir()
        
        # Create simple test files
        import pandas as pd
        train_data = pd.DataFrame({
            'text': ['hello world', 'foo bar', 'test sample'],
            'author': ['A', 'B', 'A']
        })
        train_data.to_csv(data_dir / 'train.csv', index=False)
        
        test_data = pd.DataFrame({
            'id': [1, 2],
            'text': ['test one', 'test two']
        })
        test_data.to_csv(data_dir / 'test.csv', index=False)
        
        # Create evaluator
        evaluator = DockerEvaluator(work_dir, timeout_secs=60)
        
        # Test code
        test_code = """
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Load data
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Simple model
vectorizer = TfidfVectorizer(max_features=100)
X = vectorizer.fit_transform(train_df['text'])
y = train_df['author']

model = LogisticRegression()
cv_scores = cross_val_score(model, X, y, cv=2)
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")

# Make predictions
model.fit(X, y)
X_test = vectorizer.transform(test_df['text'])
predictions = model.predict(X_test)

# Save submission
submission = pd.DataFrame({'id': test_df['id'], 'author': predictions})
submission.to_csv('submission.csv', index=False)
print("Submission saved")
"""
        
        # Execute
        result = evaluator.execute_code_with_journaling(
            code=test_code,
            claude_prompt="Test prompt",
            claude_response="Test response with code",
            claude_metadata={"call_number": 1, "model": "test"},
            verbose=True
        )
        
        print(f"Test result: {result}")
        print(f"Success: {result['success']}")
        print(f"CV Score: {result.get('cv_score')}")


if __name__ == "__main__":
    test_docker_evaluator()