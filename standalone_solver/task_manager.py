#!/usr/bin/env python3
"""
MLEBench task management for validation, setup, and data handling.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional


class TaskManager:
    """Manages MLEBench task validation, setup, and data operations."""
    
    def __init__(self, task_name: str, mlebench_data_dir: str, work_dir: str):
        self.task_name = task_name
        self.mlebench_data_dir = Path(mlebench_data_dir)
        self.work_dir = Path(work_dir)
        self.task_data_dir = self.mlebench_data_dir / task_name / "prepared" / "public"
    
    def validate_task_data(self, verbose: bool = False) -> None:
        """Validate that task data exists in MLEBench directory."""
        if not self.mlebench_data_dir.exists():
            raise ValueError(f"MLEBench data directory does not exist: {self.mlebench_data_dir}")
        
        if not self.task_data_dir.exists():
            available_tasks = self.list_available_tasks()
            if available_tasks:
                available_str = ", ".join(available_tasks[:10])  # Show first 10
                if len(available_tasks) > 10:
                    available_str += f" (and {len(available_tasks) - 10} more)"
            else:
                available_str = "None found"
            
            raise ValueError(
                f"Task '{self.task_name}' not found in MLEBench data.\n"
                f"Expected path: {self.task_data_dir}\n"
                f"Available tasks: {available_str}"
            )
        
        if verbose:
            print(f"✓ Task data validated: {self.task_data_dir}")
    
    def setup_task_data(self, verbose: bool = False) -> None:
        """Setup task data in working directory."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy or link data files
        target_data_dir = self.work_dir / "data"
        
        if target_data_dir.exists() or target_data_dir.is_symlink():
            if verbose:
                print(f"Data directory already exists: {target_data_dir}")
            return  # Data is already set up, nothing more to do
        else:
            try:
                # Try to create symlink first (faster)
                target_data_dir.symlink_to(self.task_data_dir.resolve(), target_is_directory=True)
                if verbose:
                    print(f"✓ Linked task data: {self.task_data_dir} -> {target_data_dir}")
            except OSError:
                # Fallback to copying (Windows or permissions issues)
                try:
                    shutil.copytree(self.task_data_dir, target_data_dir)
                    if verbose:
                        print(f"✓ Copied task data: {self.task_data_dir} -> {target_data_dir}")
                except FileExistsError:
                    # Directory was created between check and copy, populate it
                    target_data_dir.mkdir(parents=True, exist_ok=True)
                    for item in self.task_data_dir.iterdir():
                        if item.is_file():
                            shutil.copy2(item, target_data_dir)
                        elif item.is_dir():
                            shutil.copytree(item, target_data_dir / item.name, dirs_exist_ok=True)
                    if verbose:
                        print(f"✓ Populated data directory: {target_data_dir}")
        
        # Verify essential files exist
        essential_files = ["train.csv", "test.csv"]
        missing_files = []
        
        for file_name in essential_files:
            file_path = target_data_dir / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            print(f"⚠️ Warning: Missing expected files: {missing_files}")
        
        # Create description file if available
        desc_source = self.mlebench_data_dir / self.task_name / "description.md"
        desc_target = target_data_dir / "description.md"
        
        if desc_source.exists() and not desc_target.exists():
            shutil.copy2(desc_source, desc_target)
            if verbose:
                print(f"✓ Added task description: {desc_target}")
    
    def list_available_tasks(self) -> List[str]:
        """List all available MLEBench tasks."""
        if not self.mlebench_data_dir.exists():
            return []
        
        tasks = []
        for item in self.mlebench_data_dir.iterdir():
            if item.is_dir():
                # Check if it has the expected structure
                data_path = item / "prepared" / "public"
                if data_path.exists():
                    tasks.append(item.name)
        
        return sorted(tasks)
    
    def get_task_info(self) -> dict:
        """Get information about the current task."""
        info = {
            "task_name": self.task_name,
            "data_dir": str(self.task_data_dir),
            "work_dir": str(self.work_dir),
            "exists": self.task_data_dir.exists()
        }
        
        if self.task_data_dir.exists():
            # Count data files
            data_files = list(self.task_data_dir.glob("*.csv"))
            info["csv_files"] = [f.name for f in data_files]
            info["file_count"] = len(data_files)
            
            # Check for description
            desc_file = self.mlebench_data_dir / self.task_name / "description.md"
            info["has_description"] = desc_file.exists()
            
            # Get directory size
            total_size = sum(f.stat().st_size for f in self.task_data_dir.rglob('*') if f.is_file())
            info["total_size_mb"] = round(total_size / (1024 * 1024), 2)
        
        return info


def get_mlebench_data_dir() -> str:
    """Get MLEBench data directory from environment or default location."""
    # Check environment variable first
    env_dir = os.getenv('MLE_BENCH_DATA_DIR')
    if env_dir:
        return env_dir
    
    # Check common locations
    possible_locations = [
        Path.cwd() / ".mlebench",
        Path.home() / ".mlebench", 
        Path("/tmp/mlebench_data"),
        Path.cwd().parent / ".mlebench"  # Check parent directory
    ]
    
    for location in possible_locations:
        if location.exists() and location.is_dir():
            # Verify it contains MLEBench data by checking for task directories
            task_dirs = [d for d in location.iterdir() if d.is_dir() and (d / "prepared").exists()]
            if task_dirs:
                return str(location)
    
    # Default fallback
    return str(Path.cwd() / ".mlebench")


def list_available_tasks(mlebench_data_dir: str) -> List[str]:
    """List all available MLEBench tasks in the given directory."""
    data_dir = Path(mlebench_data_dir)
    if not data_dir.exists():
        return []
    
    tasks = []
    for item in data_dir.iterdir():
        if item.is_dir():
            # Check if it has the expected MLEBench structure
            data_path = item / "prepared" / "public"
            if data_path.exists():
                tasks.append(item.name)
    
    return sorted(tasks)