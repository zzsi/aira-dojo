#!/usr/bin/env python3
"""
Docker image management and container operations.
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import docker


class DockerManager:
    """Manages Docker images and container operations."""
    
    def __init__(self, work_dir: Path, timeout_secs: int = 600):
        self.work_dir = Path(work_dir)
        self.timeout_secs = timeout_secs
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
            self.docker_available = True
            
            # Ensure base image exists
            if not self._check_base_image():
                self._build_base_image()
                
        except Exception as e:
            print(f"Docker not available: {e}")
            self.docker_available = False
            self.docker_client = None
    
    def _check_base_image(self) -> bool:
        """Check if base ML image exists."""
        try:
            self.docker_client.images.get("ml-solver-base:latest")
            return True
        except docker.errors.ImageNotFound:
            return False
    
    def _build_base_image(self):
        """Build base ML Docker image with common dependencies."""
        print("🔨 Building base ML Docker image...")
        
        dockerfile_content = '''
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Install common ML packages
RUN pip install --no-cache-dir \\
    pandas>=1.5.0 \\
    numpy>=1.21.0 \\
    scikit-learn>=1.1.0 \\
    lightgbm>=3.3.0 \\
    matplotlib>=3.5.0 \\
    seaborn>=0.11.0

# Set working directory
WORKDIR /workspace

# Default command
CMD ["python"]
'''
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            dockerfile_path = Path(tmp_dir) / "Dockerfile"
            dockerfile_path.write_text(dockerfile_content)
            
            try:
                print("Building Docker image (this may take a few minutes)...")
                self.docker_client.images.build(
                    path=str(tmp_dir),
                    tag="ml-solver-base:latest",
                    rm=True
                )
                print("✅ Base ML Docker image built successfully")
            except Exception as e:
                print(f"❌ Failed to build Docker image: {e}")
                raise
    
    def execute_in_container(self, code: str, verbose: bool = False) -> Dict[str, Any]:
        """Execute code in Docker container."""
        if not self.docker_available:
            return {
                "success": False,
                "error": "Docker not available",
                "return_code": -1,
                "stdout": "",
                "stderr": "",
                "execution_time": 0
            }
        
        import time
        start_time = time.time()
        
        try:
            # Create temporary directory within work_dir (Docker-accessible on macOS)
            temp_name = f"docker_exec_{int(time.time())}"
            temp_path = self.work_dir / temp_name
            temp_path.mkdir(exist_ok=True)
            
            try:
                # Write code to temporary file
                solution_file = temp_path / "solution.py"
                solution_file.write_text(code)
                
                if verbose:
                    print(f"Code written to {solution_file}")
                
                # Setup data access via bind mount
                data_dir = self.work_dir / "data"
                if not data_dir.exists() and verbose:
                    print("No data directory found")
                
                # Create container volumes (Docker requires absolute paths)
                volumes = {
                    str(temp_path.resolve()): {'bind': '/workspace', 'mode': 'rw'}
                }
                
                if data_dir.exists():
                    abs_data_path = str(data_dir.resolve())
                    volumes[abs_data_path] = {'bind': '/workspace/data', 'mode': 'ro'}
                    if verbose:
                        print(f"Data access via bind mount: {abs_data_path}")
                else:
                    # Create empty data symlink inside temp_path for compatibility
                    data_link = temp_path / "data"
                    data_link.mkdir(exist_ok=True)
                
                # Run container
                if verbose:
                    print("Running in Docker container...")
                
                try:
                    container = self.docker_client.containers.run(
                        image="ml-solver-base:latest",
                        command=["python", "solution.py"],
                        volumes=volumes,
                        working_dir="/workspace",
                        detach=True,
                        remove=False,  # Don't auto-remove so we can get logs
                        network_mode="none",  # No network access for security
                        mem_limit="2g",  # Memory limit
                        cpu_count=2  # CPU limit
                    )
                    
                    # Wait for completion with timeout
                    result = container.wait(timeout=self.timeout_secs)
                    execution_time = time.time() - start_time
                    
                    # Get output before removing container
                    stdout = container.logs(stdout=True, stderr=False).decode('utf-8')
                    stderr = container.logs(stdout=False, stderr=True).decode('utf-8')
                    return_code = result['StatusCode']
                    
                    # Now remove the container
                    try:
                        container.remove()
                    except Exception as remove_error:
                        if verbose:
                            print(f"Warning: Could not remove container: {remove_error}")
                    
                    # Check for submission file
                    submission_file = temp_path / "submission.csv"
                    submission_content = None
                    if submission_file.exists():
                        submission_content = submission_file.read_text()
                    
                    return {
                        "success": return_code == 0,
                        "return_code": return_code,
                        "stdout": stdout,
                        "stderr": stderr,
                        "execution_time": execution_time,
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
                        "evaluator_type": "docker"
                    }
                    
            finally:
                # Clean up temporary directory
                try:
                    # Fix permissions before cleanup to handle Docker-created files
                    if temp_path.exists():
                        subprocess.run(['chmod', '-R', '755', str(temp_path)], check=False)
                        shutil.rmtree(temp_path)
                except Exception as cleanup_error:
                    if verbose:
                        print(f"Warning: Could not clean up {temp_path}: {cleanup_error}")
                
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "success": False,
                "error": str(e),
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
                "execution_time": execution_time,
                "evaluator_type": "docker"
            }
    
    def rebuild_image_with_requirements(self, requirements: List[str], verbose: bool = False) -> bool:
        """Rebuild Docker image with additional requirements."""
        if not self.docker_available:
            return False
        
        print(f"🔨 Rebuilding Docker image with additional packages: {', '.join(requirements)}")
        
        # Create Dockerfile with additional requirements
        base_packages = [
            "pandas>=1.5.0",
            "numpy>=1.21.0", 
            "scikit-learn>=1.1.0",
            "lightgbm>=3.3.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0"
        ]
        
        all_packages = base_packages + requirements
        packages_str = " \\\n    ".join(all_packages)
        
        dockerfile_content = f'''
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Install ML packages
RUN pip install --no-cache-dir \\
    {packages_str}

# Set working directory
WORKDIR /workspace

# Default command
CMD ["python"]
'''
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            dockerfile_path = Path(tmp_dir) / "Dockerfile"
            dockerfile_path.write_text(dockerfile_content)
            
            try:
                if verbose:
                    print("Rebuilding Docker image...")
                
                # Remove old image first
                try:
                    old_image = self.docker_client.images.get("ml-solver-base:latest")
                    self.docker_client.images.remove(old_image.id, force=True)
                except docker.errors.ImageNotFound:
                    pass
                
                # Build new image
                self.docker_client.images.build(
                    path=str(tmp_dir),
                    tag="ml-solver-base:latest",
                    rm=True
                )
                
                if verbose:
                    print("✅ Docker image rebuilt successfully")
                return True
                
            except Exception as e:
                print(f"❌ Failed to rebuild Docker image: {e}")
                return False