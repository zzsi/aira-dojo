#!/usr/bin/env python3
"""
File state manager for tracking working directory state in multifile solver.
Provides context about existing files and their relationships to Claude Code.
"""

import os
import ast
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
import hashlib


@dataclass
class FileInfo:
    """Information about a file in the working directory."""
    path: Path
    size: int
    last_modified: float
    content_hash: str
    is_python: bool
    imports: Set[str]
    functions: List[str]
    classes: List[str]


class FileStateManager:
    """Manages and tracks state of files in the working directory."""
    
    def __init__(self, work_dir: Path, verbose: bool = True):
        """
        Initialize file state manager.
        
        Args:
            work_dir: Working directory to track
            verbose: Whether to print detailed information
        """
        self.work_dir = Path(work_dir)
        self.verbose = verbose
        self.file_cache: Dict[str, FileInfo] = {}
        
        # File patterns to track
        self.tracked_extensions = {'.py', '.txt', '.md', '.json', '.yaml', '.yml', '.csv'}
        self.ignore_patterns = {'__pycache__', '.git', '.pytest_cache', 'venv', '.venv'}
    
    def scan_directory(self) -> Dict[str, FileInfo]:
        """
        Scan working directory and update file cache.
        
        Returns:
            Dictionary mapping relative paths to FileInfo objects
        """
        if not self.work_dir.exists():
            return {}
        
        current_files = {}
        
        for file_path in self.work_dir.rglob('*'):
            if not file_path.is_file():
                continue
                
            # Skip ignored patterns
            if any(pattern in str(file_path) for pattern in self.ignore_patterns):
                continue
                
            # Only track certain file types
            if file_path.suffix not in self.tracked_extensions:
                continue
            
            rel_path = str(file_path.relative_to(self.work_dir))
            
            try:
                file_info = self._analyze_file(file_path)
                current_files[rel_path] = file_info
                
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Failed to analyze {rel_path}: {e}")
        
        self.file_cache = current_files
        return current_files
    
    def _analyze_file(self, file_path: Path) -> FileInfo:
        """
        Analyze a single file and extract metadata.
        
        Args:
            file_path: Path to the file
            
        Returns:
            FileInfo object with file metadata
        """
        stat = file_path.stat()
        
        # Read content for analysis
        try:
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Binary file, just get basic info
            content = ""
        
        # Calculate content hash
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
        
        # Initialize metadata
        is_python = file_path.suffix == '.py'
        imports = set()
        functions = []
        classes = []
        
        # Analyze Python files
        if is_python and content:
            try:
                imports, functions, classes = self._analyze_python_file(content)
            except Exception:
                pass  # Ignore syntax errors in Python files
        
        return FileInfo(
            path=file_path,
            size=stat.st_size,
            last_modified=stat.st_mtime,
            content_hash=content_hash,
            is_python=is_python,
            imports=imports,
            functions=functions,
            classes=classes
        )
    
    def _analyze_python_file(self, content: str) -> Tuple[Set[str], List[str], List[str]]:
        """
        Analyze Python file content to extract imports, functions, and classes.
        
        Args:
            content: Python file content
            
        Returns:
            Tuple of (imports, functions, classes)
        """
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return set(), [], []
        
        imports = set()
        functions = []
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
            elif isinstance(node, ast.FunctionDef):
                if not node.name.startswith('_'):  # Skip private functions
                    functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        return imports, functions, classes
    
    def get_file_summary(self) -> str:
        """
        Generate a summary of all tracked files.
        
        Returns:
            Formatted string with file overview
        """
        if not self.file_cache:
            self.scan_directory()
        
        if not self.file_cache:
            return "No tracked files in working directory."
        
        summary_parts = ["# Working Directory Files\n"]
        
        # Group files by type
        python_files = []
        config_files = []
        data_files = []
        other_files = []
        
        for rel_path, file_info in sorted(self.file_cache.items()):
            if file_info.is_python:
                python_files.append((rel_path, file_info))
            elif file_info.path.suffix in {'.json', '.yaml', '.yml', '.txt'}:
                config_files.append((rel_path, file_info))
            elif file_info.path.suffix == '.csv':
                data_files.append((rel_path, file_info))
            else:
                other_files.append((rel_path, file_info))
        
        # Python files section
        if python_files:
            summary_parts.append("## Python Files\n")
            for rel_path, file_info in python_files:
                size_kb = file_info.size / 1024
                
                details = []
                if file_info.functions:
                    details.append(f"functions: {', '.join(file_info.functions[:3])}")
                    if len(file_info.functions) > 3:
                        details.append(f"+ {len(file_info.functions) - 3} more")
                if file_info.classes:
                    details.append(f"classes: {', '.join(file_info.classes)}")
                if file_info.imports:
                    key_imports = [imp for imp in file_info.imports if imp not in {'os', 'sys', 'time', 'datetime'}]
                    if key_imports:
                        details.append(f"imports: {', '.join(sorted(key_imports)[:3])}")
                
                detail_str = f" ({'; '.join(details)})" if details else ""
                summary_parts.append(f"- **{rel_path}** ({size_kb:.1f}KB){detail_str}\n")
        
        # Config files section
        if config_files:
            summary_parts.append("\n## Configuration Files\n")
            for rel_path, file_info in config_files:
                size_kb = file_info.size / 1024
                summary_parts.append(f"- **{rel_path}** ({size_kb:.1f}KB)\n")
        
        # Data files section
        if data_files:
            summary_parts.append("\n## Data Files\n")
            for rel_path, file_info in data_files:
                size_kb = file_info.size / 1024
                summary_parts.append(f"- **{rel_path}** ({size_kb:.1f}KB)\n")
        
        # Other files section
        if other_files:
            summary_parts.append("\n## Other Files\n")
            for rel_path, file_info in other_files:
                size_kb = file_info.size / 1024
                summary_parts.append(f"- **{rel_path}** ({size_kb:.1f}KB)\n")
        
        return "".join(summary_parts)
    
    def get_file_content(self, rel_path: str, max_lines: Optional[int] = None) -> Optional[str]:
        """
        Get content of a specific file.
        
        Args:
            rel_path: Relative path to the file
            max_lines: Maximum number of lines to return
            
        Returns:
            File content or None if file doesn't exist
        """
        file_path = self.work_dir / rel_path
        if not file_path.exists():
            return None
        
        try:
            content = file_path.read_text(encoding='utf-8')
            
            if max_lines:
                lines = content.split('\n')
                if len(lines) > max_lines:
                    truncated_lines = lines[:max_lines]
                    truncated_lines.append(f"... (truncated, {len(lines) - max_lines} more lines)")
                    content = '\n'.join(truncated_lines)
            
            return content
            
        except UnicodeDecodeError:
            return f"[Binary file: {file_path.suffix}]"
    
    def get_python_dependencies(self) -> Set[str]:
        """
        Get all Python dependencies from tracked Python files.
        
        Returns:
            Set of all imported modules
        """
        if not self.file_cache:
            self.scan_directory()
        
        all_imports = set()
        for file_info in self.file_cache.values():
            if file_info.is_python:
                all_imports.update(file_info.imports)
        
        return all_imports
    
    def detect_changes(self) -> Dict[str, str]:
        """
        Detect changes since last scan.
        
        Returns:
            Dictionary mapping file paths to change types ('added', 'modified', 'deleted')
        """
        old_cache = self.file_cache.copy()
        current_files = self.scan_directory()
        
        changes = {}
        
        # Find added and modified files
        for rel_path, file_info in current_files.items():
            if rel_path not in old_cache:
                changes[rel_path] = 'added'
            elif old_cache[rel_path].content_hash != file_info.content_hash:
                changes[rel_path] = 'modified'
        
        # Find deleted files
        for rel_path in old_cache:
            if rel_path not in current_files:
                changes[rel_path] = 'deleted'
        
        return changes
    
    def get_requirements_content(self) -> Optional[str]:
        """
        Get content of requirements.txt if it exists.
        
        Returns:
            Content of requirements.txt or None
        """
        req_file = self.work_dir / "requirements.txt"
        if req_file.exists():
            return req_file.read_text().strip()
        return None
    
    def suggest_next_files(self) -> List[str]:
        """
        Suggest which files might need attention next.
        
        Returns:
            List of file paths that might need work
        """
        if not self.file_cache:
            self.scan_directory()
        
        suggestions = []
        
        # Check if we have a main solution file
        has_solution = any('solution' in path.lower() for path in self.file_cache.keys())
        has_main = any(path in self.file_cache for path in ['main.py', 'run.py', 'solve.py'])
        
        if not has_solution and not has_main:
            suggestions.append("Create a main solution file (e.g., solution.py)")
        
        # Check for requirements.txt
        if not any(path == 'requirements.txt' for path in self.file_cache.keys()):
            dependencies = self.get_python_dependencies()
            external_deps = dependencies - {'os', 'sys', 'time', 'datetime', 'json', 're', 'pathlib', 'typing'}
            if external_deps:
                suggestions.append("Create requirements.txt for dependencies")
        
        # Check for utility modules
        python_files = [path for path, info in self.file_cache.items() if info.is_python]
        if len(python_files) == 1 and len(self.file_cache) > 3:
            suggestions.append("Consider splitting code into utility modules")
        
        return suggestions