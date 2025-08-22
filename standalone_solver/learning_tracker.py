#!/usr/bin/env python3
"""
Learning and insights tracking for iterative ML experimentation.
"""

import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


class LearningTracker:
    """Tracks learnings and insights from ML experimentation iterations."""
    
    def __init__(self, work_dir: str):
        self.work_dir = Path(work_dir)
        self.learnings_file = self.work_dir / "learnings.md"
        self._setup_learnings_tracking()
    
    def _setup_learnings_tracking(self):
        """Initialize learnings tracking file."""
        if not self.learnings_file.exists():
            initial_content = f"""# ML Learning Journal - {datetime.now().strftime("%Y-%m-%d")}

This file tracks insights, patterns, and learnings from iterative ML experimentation.

## What Works ✅

## What Doesn't Work ❌

## Key Insights 💡

## Error Patterns 🐛

## Next Steps 📋

---
*Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
            self.learnings_file.write_text(initial_content)
    
    def record_learning(self, iteration: int, solution: dict, category: str, insight: str):
        """
        Record a learning from an iteration.
        
        Args:
            iteration: The iteration number
            solution: The solution dictionary containing results
            category: Category of learning (success, error, insight, pattern)
            insight: The insight text to record
        """
        timestamp = self._get_timestamp()
        
        # Extract core insight without redundant details
        core_insight = self._extract_core_insight(insight)
        
        # Determine which section to update based on category and solution success
        if category == "success" or (solution.get('success') and category in ["insight", "pattern"]):
            section = "## What Works ✅"
        elif category == "error" or (not solution.get('success') and category in ["insight", "pattern"]):  
            section = "## What Doesn't Work ❌"
        elif category == "insight":
            section = "## Key Insights 💡"
        elif category == "error":
            section = "## Error Patterns 🐛"
        else:
            section = "## Key Insights 💡"  # Default fallback
        
        # Format the learning entry
        cv_score = solution.get('execution_result', {}).get('cv_score', 'N/A')
        entry = f"- **Iter {iteration}** (CV: {cv_score}): {core_insight} *({timestamp})*"
        
        # Update the file
        self._append_to_section(section, entry)
        
        # Update timestamp
        self._update_timestamp()
    
    def _get_timestamp(self) -> str:
        """Get formatted timestamp."""
        return datetime.now().strftime("%H:%M")
    
    def _extract_core_insight(self, insight: str) -> str:
        """Extract the core insight, removing redundant error details."""
        # Remove common prefixes
        prefixes_to_remove = [
            "STDERR: ",
            "STDOUT: ",
            "Exit code: ",
            "Error: ",
            "Failed: ",
            "Success: "
        ]
        
        cleaned = insight
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
        
        # Extract just the first meaningful error line for errors
        if any(error_indicator in cleaned.lower() for error_indicator in ['error', 'failed', 'exception']):
            lines = cleaned.split('\n')
            meaningful_line = None
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('Traceback') and 'File "' not in line:
                    meaningful_line = line
                    break
            
            if meaningful_line:
                cleaned = meaningful_line
        
        # Truncate if too long
        if len(cleaned) > 150:
            cleaned = cleaned[:147] + "..."
        
        return cleaned
    
    def _append_to_section(self, section_header: str, entry: str):
        """Append an entry to a specific section in the learnings file."""
        if not self.learnings_file.exists():
            self._setup_learnings_tracking()
        
        content = self.learnings_file.read_text()
        lines = content.split('\n')
        
        # Find the section
        section_start = -1
        for i, line in enumerate(lines):
            if line.strip() == section_header:
                section_start = i
                break
        
        if section_start == -1:
            # Section not found, append it
            lines.append(f"\n{section_header}\n")
            lines.append(entry)
        else:
            # Find where to insert (after section header, before next section or end)
            insert_pos = section_start + 1
            
            # Skip any existing entries in this section
            for i in range(section_start + 1, len(lines)):
                if lines[i].strip().startswith('##'):
                    insert_pos = i
                    break
                elif lines[i].strip().startswith('- '):
                    insert_pos = i + 1
                elif lines[i].strip() == '':
                    continue
                else:
                    break
            
            # Insert the new entry
            lines.insert(insert_pos, entry)
        
        # Write back to file
        self.learnings_file.write_text('\n'.join(lines))
    
    def _update_timestamp(self):
        """Update the 'last updated' timestamp at the bottom of the file."""
        if not self.learnings_file.exists():
            return
        
        content = self.learnings_file.read_text()
        
        # Update timestamp line
        timestamp_pattern = r'\*Last updated: [^*]+\*'
        new_timestamp = f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        if re.search(timestamp_pattern, content):
            content = re.sub(timestamp_pattern, new_timestamp, content)
        else:
            # Add timestamp if not present
            content = content.rstrip() + f"\n\n---\n{new_timestamp}\n"
        
        self.learnings_file.write_text(content)
    
    def analyze_and_learn_from_iteration(self, iteration: int, solution: dict) -> None:
        """
        Analyze an iteration and automatically extract learnings.
        
        Args:
            iteration: The iteration number
            solution: The solution dictionary with execution results
        """
        execution_result = solution.get('execution_result', {})
        success = solution.get('success', False)
        cv_score = execution_result.get('cv_score')
        
        if success and cv_score is not None:
            # Record successful approach
            self.record_learning(iteration, solution, "success", 
                               f"Achieved CV score {cv_score:.4f} with successful execution")
            
            # Extract successful patterns from code if available
            code = solution.get('code', '')
            if code:
                self._extract_successful_patterns(iteration, solution, code)
        
        else:
            # Analyze failure and extract insights
            error_info = solution.get('error', 'Unknown error')
            stderr = execution_result.get('stderr', '')
            
            # Record the failure
            if stderr:
                self.record_learning(iteration, solution, "error", 
                                   f"Execution failed: {self._extract_key_error(stderr)}")
            else:
                self.record_learning(iteration, solution, "error", f"Failed: {error_info}")
            
            # Extract specific error patterns
            self._extract_error_patterns(iteration, solution, error_info, stderr)
    
    def _extract_successful_patterns(self, iteration: int, solution: dict, code: str):
        """Extract patterns from successful code."""
        patterns = []
        
        # Common successful ML patterns
        if 'TfidfVectorizer' in code:
            patterns.append("TF-IDF vectorization works well for text classification")
        if 'LogisticRegression' in code:
            patterns.append("Logistic Regression is effective for this task")
        if 'cross_val_score' in code:
            patterns.append("cross_val_score provides reliable validation")
        if 'LeaveOneOut' in code:
            patterns.append("LeaveOneOut CV handles small datasets well")
        if 'StratifiedKFold' in code and 'n_splits=3' in code:
            patterns.append("3-fold StratifiedKFold works for small datasets")
        
        # Record patterns
        for pattern in patterns:
            self.record_learning(iteration, solution, "insight", pattern)
    
    def _extract_error_patterns(self, iteration: int, solution: dict, error_info: str, stderr: str):
        """Extract patterns from errors."""
        error_text = (error_info + " " + stderr).lower()
        
        # Common error patterns
        if "n_splits=5 cannot be greater" in error_text:
            self.record_learning(iteration, solution, "error", 
                               "n_splits=5 too high for small datasets - use LeaveOneOut or n_splits=3")
        
        elif "modulenotfounderror" in error_text or "importerror" in error_text:
            # Extract module name
            import re
            module_match = re.search(r"no module named ['\"]([^'\"]+)['\"]", error_text)
            if module_match:
                module = module_match.group(1)
                self.record_learning(iteration, solution, "error", 
                                   f"Missing module '{module}' - check dependencies")
            else:
                self.record_learning(iteration, solution, "error", "Import error - check dependencies")
        
        elif "memory" in error_text:
            self.record_learning(iteration, solution, "error", 
                               "Memory issue - try simpler models or reduce features")
        
        elif "timeout" in error_text:
            self.record_learning(iteration, solution, "error", 
                               "Execution timeout - optimize code or reduce complexity")
    
    def _extract_key_error(self, stderr: str) -> str:
        """Extract the key error message from stderr."""
        lines = stderr.split('\n')
        
        # Look for the actual error (usually last meaningful line)
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('  ') and ':' in line:
                return line
        
        # Fallback to first few non-empty lines
        meaningful_lines = [line.strip() for line in lines[:3] if line.strip()]
        return ' | '.join(meaningful_lines) if meaningful_lines else "Unknown error"
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of all learnings."""
        if not self.learnings_file.exists():
            return {"error": "No learnings file found"}
        
        content = self.learnings_file.read_text()
        
        # Count entries in each section
        what_works = len(re.findall(r'- \*\*Iter \d+\*\*.*', self._extract_section(content, "## What Works ✅")))
        what_fails = len(re.findall(r'- \*\*Iter \d+\*\*.*', self._extract_section(content, "## What Doesn't Work ❌")))
        insights = len(re.findall(r'- \*\*Iter \d+\*\*.*', self._extract_section(content, "## Key Insights 💡")))
        errors = len(re.findall(r'- \*\*Iter \d+\*\*.*', self._extract_section(content, "## Error Patterns 🐛")))
        
        return {
            "learnings_file": str(self.learnings_file),
            "total_entries": what_works + what_fails + insights + errors,
            "successful_patterns": what_works,
            "failed_patterns": what_fails,
            "insights": insights,
            "error_patterns": errors
        }
    
    def _extract_section(self, content: str, section_header: str) -> str:
        """Extract content from a specific section."""
        lines = content.split('\n')
        section_start = -1
        
        for i, line in enumerate(lines):
            if line.strip() == section_header:
                section_start = i
                break
        
        if section_start == -1:
            return ""
        
        section_lines = []
        for i in range(section_start + 1, len(lines)):
            line = lines[i]
            if line.strip().startswith('##'):
                break
            section_lines.append(line)
        
        return '\n'.join(section_lines)