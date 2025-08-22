#!/usr/bin/env python3
"""
Docker-based code evaluator that replaces virtual environment dependency management.
Provides same interface as ImprovedCodeEvaluator for drop-in replacement.
"""

import os
import sys
import time
import docker
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import subprocess
import json


class DockerEvaluator:
    """Docker-based evaluator with same interface as ImprovedCodeEvaluator."""
    
    def __init__(self, work_dir: Path, timeout_secs: int = 600, use_data_aliases: bool = True):
        """
        Initialize Docker evaluator.
        
        Args:
            work_dir: Working directory containing data files
            timeout_secs: Maximum execution time for code
            use_data_aliases: Whether to use bind mounts (always True for Docker)
        """
        self.work_dir = Path(work_dir)
        self.timeout_secs = timeout_secs
        self.evaluation_count = 0
        self.use_data_aliases = True  # Docker always uses bind mounts
        
        # Create journals directory
        self.journals_dir = self.work_dir / "journals"
        self.journals_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
            self.docker_available = True
            if not self._check_base_image():
                self._build_base_image()
        except Exception as e:
            print(f"⚠️  Docker not available: {e}")
            print("Falling back to improved evaluator...")
            # Import and delegate to improved evaluator as fallback
            from improved_evaluator import ImprovedCodeEvaluator
            self._fallback_evaluator = ImprovedCodeEvaluator(work_dir, timeout_secs, use_data_aliases)
            self.docker_available = False
    
    def _check_base_image(self) -> bool:
        """Check if base ML image exists."""
        try:
            self.docker_client.images.get("ml-solver-base:latest")
            return True
        except docker.errors.ImageNotFound:
            return False
    
    def _build_base_image(self):
        """Build base ML Docker image with common packages."""
        print("🐳 Building base ML Docker image...")
        
        dockerfile_content = """
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    software-properties-common \\
    && rm -rf /var/lib/apt/lists/*

# Install common ML packages
COPY requirements-base.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-base.txt

# Create working directory
WORKDIR /workspace

# Set Python path
ENV PYTHONPATH=/workspace

# Default command
CMD ["python"]
"""
        
        base_requirements = """
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.2.0
xgboost>=1.7.0
lightgbm>=3.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
"""
        
        # Create temporary build context
        with tempfile.TemporaryDirectory() as build_dir:
            build_path = Path(build_dir)
            
            # Write Dockerfile
            (build_path / "Dockerfile").write_text(dockerfile_content)
            (build_path / "requirements-base.txt").write_text(base_requirements)
            
            # Build image
            try:
                self.docker_client.images.build(
                    path=str(build_path),
                    tag="ml-solver-base:latest",
                    rm=True,
                    pull=True
                )
                print("✅ Base ML Docker image built successfully")
            except Exception as e:
                print(f"❌ Failed to build Docker image: {e}")
                raise
    
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
        # Fallback to improved evaluator if Docker not available
        if not self.docker_available:
            return self._fallback_evaluator.execute_code_with_journaling(
                code, claude_prompt, claude_response, claude_metadata, verbose
            )
        
        # Check if Claude wants to modify dependencies
        if "requirements.txt" in claude_response and "rebuild" in claude_response.lower():
            self._handle_requirements_update(claude_response, verbose)
        
        self.evaluation_count += 1
        
        if verbose:
            print(f"[Execution {self.evaluation_count}] Starting Docker code execution...")
        
        # Create journal entry first
        self._journal_claude_interaction(claude_prompt, claude_response, claude_metadata)
        
        # Execute code in Docker
        execution_result = self._execute_code_in_docker(code, verbose)
        
        # Save artifacts
        self._save_artifacts(code, execution_result, verbose)
        
        return execution_result
    
    def _execute_code_in_docker(self, code: str, verbose: bool = False) -> Dict[str, Any]:
        """Execute code in Docker container."""
        start_time = time.time()
        
        try:
            # Create temporary directory for this execution
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Write code to temporary file
                solution_file = temp_path / "solution.py"
                solution_file.write_text(code)
                
                if verbose:
                    print(f"[Execution {self.evaluation_count}] Code written to {solution_file}")
                
                # Setup data access via bind mount
                data_dir = self.work_dir / "data"
                if not data_dir.exists():
                    if verbose:
                        print(f"[Execution {self.evaluation_count}] No data directory found")
                
                # Create container volumes
                volumes = {
                    str(temp_path): {'bind': '/workspace', 'mode': 'rw'}
                }
                
                if data_dir.exists():
                    volumes[str(data_dir)] = {'bind': '/workspace/data', 'mode': 'ro'}
                    if verbose:
                        print(f"[Execution {self.evaluation_count}] Data access via bind mount")
                
                # Run container
                if verbose:
                    print(f"[Execution {self.evaluation_count}] Running in Docker container...")
                
                try:
                    container = self.docker_client.containers.run(
                        image="ml-solver-base:latest",
                        command=["python", "solution.py"],
                        volumes=volumes,
                        working_dir="/workspace",
                        detach=True,
                        remove=True,
                        network_mode="none",  # No network access for security
                        mem_limit="2g",  # Memory limit
                        cpu_count=2  # CPU limit
                    )
                    
                    # Wait for completion with timeout
                    result = container.wait(timeout=self.timeout_secs)
                    execution_time = time.time() - start_time
                    
                    # Get output
                    stdout = container.logs(stdout=True, stderr=False).decode('utf-8')
                    stderr = container.logs(stdout=False, stderr=True).decode('utf-8')
                    return_code = result['StatusCode']
                    
                    # Check for submission file
                    submission_file = temp_path / "submission.csv"
                    submission_content = ""
                    if submission_file.exists():
                        submission_content = submission_file.read_text()
                        # Copy to work directory
                        shutil.copy2(submission_file, self.work_dir / "submission.csv")
                        if verbose:
                            print(f"[Execution {self.evaluation_count}] Submission file created ({len(submission_content)} chars)")
                    
                    # Extract CV score from output
                    cv_score = self._extract_cv_score(stdout)
                    
                    # Determine success
                    success = return_code == 0 and cv_score is not None
                    
                    if verbose:
                        status = "SUCCESS" if success else "FAILED"
                        print(f"[Execution {self.evaluation_count}] {status} ({execution_time:.2f}s)")
                        if cv_score:
                            print(f"[Execution {self.evaluation_count}] CV Score: {cv_score}")
                    
                    return {
                        "success": success,
                        "return_code": return_code,
                        "stdout": stdout,
                        "stderr": stderr,
                        "execution_time": execution_time,
                        "cv_score": cv_score,
                        "submission_content": submission_content,
                        "evaluator_type": "docker"
                    }
                    
                except docker.errors.ContainerError as e:
                    execution_time = time.time() - start_time
                    return {
                        "success": False,
                        "return_code": e.exit_status,
                        "stdout": "",
                        "stderr": str(e),
                        "execution_time": execution_time,
                        "cv_score": None,
                        "evaluator_type": "docker"
                    }
                
        except Exception as e:
            execution_time = time.time() - start_time
            if verbose:
                print(f"[Execution {self.evaluation_count}] Docker execution failed: {e}")
            
            return {
                "success": False,
                "return_code": -1,
                "stdout": "",
                "stderr": f"Docker execution error: {e}",
                "execution_time": execution_time,
                "cv_score": None,
                "evaluator_type": "docker"
            }
    
    def _extract_cv_score(self, stdout: str) -> Optional[float]:
        """Extract cross-validation score from stdout."""
        import re
        
        patterns = [
            r"Cross[- ]?validation.*?(?:score|accuracy).*?(\d+\.?\d*)",
            r"CV.*?(?:score|accuracy).*?(\d+\.?\d*)",
            r"(?:score|accuracy).*?(\d+\.?\d*)",
            r"(\d+\.?\d+)"  # Fallback: any decimal number
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, stdout, re.IGNORECASE)
            if matches:
                try:
                    score = float(matches[-1])  # Take last match
                    if 0 <= score <= 1:
                        return score
                    elif score > 1:  # Might be percentage
                        return score / 100
                except ValueError:
                    continue
        
        return None
    
    def _journal_claude_interaction(self, prompt: str, response: str, metadata: Dict[str, Any]):
        """Journal Claude interaction - same as ImprovedCodeEvaluator."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        journal_file = self.journals_dir / f"claude_session_{timestamp}.md"
        
        journal_content = f"""# Claude Interaction - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Metadata
- **Call Number**: {metadata.get('call_number', 'N/A')}
- **Model**: {metadata.get('model', 'unknown')}
- **Latency**: {metadata.get('latency', 0):.2f}s
- **Prompt Length**: {metadata.get('prompt_length', len(prompt))} characters
- **Response Length**: {metadata.get('response_length', len(response))} characters
- **Success**: {metadata.get('return_code', 0) == 0}

## Prompt Sent to Claude
```
{prompt}
```

## Claude's Response
```
{response}
```

## Execution Results
- **Evaluator**: Docker
- **Timestamp**: {datetime.now().isoformat()}
"""
        
        journal_file.write_text(journal_content)
    
    def _save_artifacts(self, code: str, execution_result: Dict[str, Any], verbose: bool = False):
        """Save execution artifacts - same as ImprovedCodeEvaluator."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifacts_dir = self.work_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        # Save solution code
        solution_file = artifacts_dir / f"solution_{self.evaluation_count}_{timestamp}.py"
        solution_file.write_text(code)
        
        # Save stdout
        if execution_result.get("stdout"):
            stdout_file = artifacts_dir / f"stdout_{self.evaluation_count}_{timestamp}.log"
            stdout_file.write_text(execution_result["stdout"])
        
        # Save stderr
        if execution_result.get("stderr"):
            stderr_file = artifacts_dir / f"stderr_{self.evaluation_count}_{timestamp}.log"
            stderr_file.write_text(execution_result["stderr"])
        
        # Save submission if available
        if execution_result.get("submission_content"):
            submission_file = artifacts_dir / f"submission_{self.evaluation_count}_{timestamp}.csv"
            submission_file.write_text(execution_result["submission_content"])
        
        if verbose:
            print(f"[Artifacts] Saved iteration {self.evaluation_count} artifacts with timestamp {timestamp}")
    
    def rebuild_image_with_requirements(self, requirements: List[str], verbose: bool = False) -> bool:
        """
        Rebuild Docker image with additional requirements.
        This allows Claude to modify dependencies during a session.
        
        Args:
            requirements: List of additional packages to install
            verbose: Whether to print build details
            
        Returns:
            True if successful, False otherwise
        """
        if not self.docker_available:
            if verbose:
                print("Docker not available, cannot rebuild image")
            return False
        
        try:
            if verbose:
                print(f"🐳 Rebuilding Docker image with additional packages: {', '.join(requirements)}")
            
            # Create new requirements file
            base_requirements = """
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.2.0
xgboost>=1.7.0
lightgbm>=3.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
"""
            
            additional_requirements = "\n".join(requirements)
            full_requirements = base_requirements + "\n" + additional_requirements
            
            dockerfile_content = """
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    software-properties-common \\
    && rm -rf /var/lib/apt/lists/*

# Install ML packages
COPY requirements-full.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-full.txt

# Create working directory
WORKDIR /workspace

# Set Python path
ENV PYTHONPATH=/workspace

# Default command
CMD ["python"]
"""
            
            # Create temporary build context
            with tempfile.TemporaryDirectory() as build_dir:
                build_path = Path(build_dir)
                
                # Write files
                (build_path / "Dockerfile").write_text(dockerfile_content)
                (build_path / "requirements-full.txt").write_text(full_requirements)
                
                # Build new image
                self.docker_client.images.build(
                    path=str(build_path),
                    tag="ml-solver-base:latest",
                    rm=True,
                    pull=True,
                    forcerm=True
                )
                
                if verbose:
                    print("✅ Docker image rebuilt successfully")
                
                return True
                
        except Exception as e:
            if verbose:
                print(f"❌ Failed to rebuild Docker image: {e}")
            return False
    
    def _handle_requirements_update(self, claude_response: str, verbose: bool = False):
        """Handle Claude's request to update requirements and rebuild image."""
        import re
        
        # Extract requirements from Claude's response
        requirements_match = re.search(r'```requirements\.txt\n(.*?)\n```', claude_response, re.DOTALL)
        if not requirements_match:
            if verbose:
                print("⚠️  Could not extract requirements.txt from Claude's response")
            return
        
        requirements_content = requirements_match.group(1).strip()
        new_packages = [line.strip() for line in requirements_content.split('\n') if line.strip()]
        
        if verbose:
            print(f"🔄 Claude requested Docker rebuild with packages: {', '.join(new_packages)}")
        
        # Rebuild image with new requirements
        self.rebuild_image_with_requirements(new_packages, verbose)


def test_docker_evaluator():
    """Test the Docker evaluator."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir)
        
        # Create test data
        data_dir = work_dir / "data"
        data_dir.mkdir()
        (data_dir / "test.csv").write_text("col1,col2\n1,2\n3,4\n")
        
        # Create evaluator
        evaluator = DockerEvaluator(work_dir)
        
        # Test code
        test_code = """
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv('./data/test.csv')
print(f"Data shape: {df.shape}")

# Mock ML task
X = np.random.random((10, 5))
y = np.random.randint(0, 3, 10)

# Cross-validation
clf = RandomForestClassifier(n_estimators=10, random_state=42)
scores = cross_val_score(clf, X, y, cv=3)
cv_score = scores.mean()

print(f"Cross-validation score: {cv_score:.4f}")

# Save submission
submission = pd.DataFrame({'id': range(5), 'prediction': [0, 1, 2, 0, 1]})
submission.to_csv('submission.csv', index=False)
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