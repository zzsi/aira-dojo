#!/usr/bin/env python3
"""
Dynamic dependency management for ML solver.
Automatically detects, installs, and manages Python packages needed by generated code.
"""

import ast
import os
import sys
import subprocess
import hashlib
import venv
from pathlib import Path
from typing import Set, List, Dict, Optional, Tuple
import tempfile
import json

class DependencyManager:
    """Manages dynamic dependency installation and virtual environments."""
    
    def __init__(self, cache_dir: Path, verbose: bool = True):
        """
        Initialize dependency manager.
        
        Args:
            cache_dir: Directory to store virtual environments
            verbose: Whether to print detailed progress
        """
        self.cache_dir = Path(cache_dir)
        self.verbose = verbose
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Mapping of import names to package names
        self.import_to_package = {
            'cv2': 'opencv-python',
            'sklearn': 'scikit-learn',
            'skimage': 'scikit-image',
            'PIL': 'Pillow',
            'pil': 'Pillow',
            'lgb': 'lightgbm',
            'lightgbm': 'lightgbm',
            'xgb': 'xgboost',
            'xgboost': 'xgboost',
            'catboost': 'catboost',
            'torch': 'torch',
            'pytorch': 'torch',
            'tensorflow': 'tensorflow',
            'tf': 'tensorflow',
            'keras': 'keras',
            'transformers': 'transformers',
            'datasets': 'datasets',
            'huggingface_hub': 'huggingface_hub',
            'sentence_transformers': 'sentence-transformers',
            'spacy': 'spacy',
            'nltk': 'nltk',
            'gensim': 'gensim',
            'plotly': 'plotly',
            'seaborn': 'seaborn',
            'matplotlib': 'matplotlib',
            'bokeh': 'bokeh',
            'altair': 'altair',
            'streamlit': 'streamlit',
            'dash': 'dash',
            'requests': 'requests',
            'urllib3': 'urllib3',
            'beautifulsoup4': 'beautifulsoup4',
            'bs4': 'beautifulsoup4',
            'lxml': 'lxml',
            'openpyxl': 'openpyxl',
            'xlrd': 'xlrd',
            'tqdm': 'tqdm',
            'joblib': 'joblib',
            'dask': 'dask',
            'ray': 'ray',
        }
        
    
    def is_standard_library(self, module_name: str) -> bool:
        """
        Check if a module is part of the standard library.
        
        Args:
            module_name: Name of the module to check
            
        Returns:
            True if module is part of standard library, False otherwise
        """
        try:
            import importlib.util
            import sys
            import sysconfig
            
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                return False
            
            # If no origin (built-in module like 'sys', 'math')
            if spec.origin is None:
                return True
            
            # Standard library modules are NOT in site-packages
            if 'site-packages' in spec.origin:
                return False
            
            # Get the standard library path
            stdlib_dir = sysconfig.get_path('stdlib')
            
            # Check if the module is in the standard library directory
            return spec.origin.startswith(stdlib_dir)
            
        except (ImportError, ValueError, ModuleNotFoundError, AttributeError):
            # If we can't determine, assume it's not standard library
            return False
    
    def extract_imports(self, code: str) -> Set[str]:
        """
        Extract all import statements from Python code.
        
        Args:
            code: Python source code
            
        Returns:
            Set of imported module names
        """
        imports = set()
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]  # Get top-level module
                        imports.add(module_name)
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split('.')[0]  # Get top-level module
                        imports.add(module_name)
                        
        except SyntaxError as e:
            if self.verbose:
                print(f"[Dependency] Warning: Could not parse code for imports: {e}")
        
        return imports
    
    def resolve_packages(self, imports: Set[str]) -> List[str]:
        """
        Resolve import names to package names.
        
        Args:
            imports: Set of imported module names
            
        Returns:
            List of package names to install
        """
        packages = []
        
        for import_name in imports:
            # Skip standard library modules (check the import name, not package name)
            if self.is_standard_library(import_name):
                continue
                
            # Map to package name first
            package_name = self.import_to_package.get(import_name, import_name)
            
                
            packages.append(package_name)
        
        return sorted(list(set(packages)))  # Remove duplicates and sort
    
    def create_dependency_fingerprint(self, packages: List[str]) -> str:
        """
        Create a unique fingerprint for a set of dependencies.
        
        Args:
            packages: List of package names
            
        Returns:
            Hash string representing this dependency set
        """
        # Sort packages for consistent hashing
        sorted_packages = sorted(packages)
        content = json.dumps(sorted_packages, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def get_venv_path(self, fingerprint: str) -> Path:
        """Get the path for a virtual environment with given fingerprint."""
        return self.cache_dir / f"venv_{fingerprint}"
    
    def create_virtual_environment(self, venv_path: Path) -> bool:
        """
        Create a new virtual environment.
        
        Args:
            venv_path: Path where to create the virtual environment
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.verbose:
                print(f"[Dependency] Creating virtual environment: {venv_path}")
            
            # Create virtual environment
            venv.create(venv_path, with_pip=True, clear=True)
            
            # Upgrade pip in the new environment
            pip_path = self.get_pip_path(venv_path)
            subprocess.run([
                str(pip_path), 'install', '--upgrade', 'pip'
            ], check=True, capture_output=True, text=True)
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"[Dependency] Failed to create virtual environment: {e}")
            return False
    
    def get_python_path(self, venv_path: Path) -> Path:
        """Get the Python executable path for a virtual environment."""
        if os.name == 'nt':  # Windows
            return venv_path.resolve() / "Scripts" / "python.exe"
        else:  # Unix-like
            return venv_path.resolve() / "bin" / "python"
    
    def get_pip_path(self, venv_path: Path) -> Path:
        """Get the pip executable path for a virtual environment."""
        if os.name == 'nt':  # Windows
            return venv_path.resolve() / "Scripts" / "pip.exe"
        else:  # Unix-like
            return venv_path.resolve() / "bin" / "pip"
    
    def install_packages(self, venv_path: Path, packages: List[str]) -> bool:
        """
        Install packages in the virtual environment.
        
        Args:
            venv_path: Path to virtual environment
            packages: List of package names to install
            
        Returns:
            True if all packages installed successfully, False otherwise
        """
        if not packages:
            return True
            
        pip_path = self.get_pip_path(venv_path)
        
        if self.verbose:
            print(f"[Dependency] Installing packages: {', '.join(packages)}")
        
        try:
            # Install packages
            cmd = [str(pip_path), 'install'] + packages
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if self.verbose:
                print(f"[Dependency] Successfully installed: {', '.join(packages)}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            if self.verbose:
                print(f"[Dependency] Failed to install packages: {e}")
                print(f"[Dependency] Stderr: {e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            if self.verbose:
                print(f"[Dependency] Package installation timed out")
            return False
    
    def prepare_environment(self, code: str) -> Tuple[Optional[Path], List[str]]:
        """
        Prepare a virtual environment with all dependencies needed for the code.
        
        Args:
            code: Python source code to analyze
            
        Returns:
            Tuple of (venv_path, packages) where venv_path is None if preparation failed
        """
        # Extract imports and resolve to packages
        imports = self.extract_imports(code)
        packages = self.resolve_packages(imports)
        
        if self.verbose and packages:
            print(f"[Dependency] Detected packages needed: {', '.join(packages)}")
        elif self.verbose:
            print(f"[Dependency] No external packages needed")
        
        # Create fingerprint and check if environment exists
        fingerprint = self.create_dependency_fingerprint(packages)
        venv_path = self.get_venv_path(fingerprint)
        
        if venv_path.exists():
            if self.verbose:
                print(f"[Dependency] Reusing cached environment: {fingerprint}")
            return venv_path, packages
        
        # Create new environment
        if not self.create_virtual_environment(venv_path):
            return None, packages
        
        # Install packages
        if not self.install_packages(venv_path, packages):
            # Clean up failed environment
            import shutil
            shutil.rmtree(venv_path, ignore_errors=True)
            return None, packages
        
        # Save metadata
        metadata = {
            'packages': packages,
            'fingerprint': fingerprint,
            'created': str(Path().cwd()),  # Current timestamp would be better
        }
        
        metadata_file = venv_path / 'dependency_metadata.json'
        metadata_file.write_text(json.dumps(metadata, indent=2))
        
        if self.verbose:
            print(f"[Dependency] Environment ready: {fingerprint}")
        
        return venv_path, packages
    
    def cleanup_old_environments(self, max_count: int = 10):
        """
        Clean up old virtual environments to save disk space.
        
        Args:
            max_count: Maximum number of environments to keep
        """
        if not self.cache_dir.exists():
            return
        
        # Get all virtual environments sorted by modification time
        venv_dirs = [d for d in self.cache_dir.iterdir() if d.is_dir() and d.name.startswith('venv_')]
        venv_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove oldest environments
        for old_venv in venv_dirs[max_count:]:
            if self.verbose:
                print(f"[Dependency] Cleaning up old environment: {old_venv.name}")
            import shutil
            shutil.rmtree(old_venv, ignore_errors=True)


def main():
    """Test the dependency manager."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir) / "dep_cache"
        dm = DependencyManager(cache_dir, verbose=True)
        
        # Test code with various imports
        test_code = """
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from transformers import AutoModel
import torch
import matplotlib.pyplot as plt
"""
        
        print("Testing dependency management...")
        venv_path, packages = dm.prepare_environment(test_code)
        
        if venv_path:
            print(f"✅ Environment created successfully: {venv_path}")
            print(f"📦 Packages: {packages}")
        else:
            print("❌ Environment creation failed")


if __name__ == "__main__":
    main()