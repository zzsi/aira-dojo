#!/usr/bin/env python3
"""
MLEBench-aware solver that can work with any MLEBench task.
Automatically sources data from MLEBench directory structure and organizes outputs by task.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    pass  # python-dotenv not installed, skip loading

from git_versioned_solver import GitVersionedSolver


class MLEBenchSolver(GitVersionedSolver):
    """Solver that can work with any MLEBench task."""
    
    def __init__(self, task_name: str, mlebench_data_dir: str, 
                 max_iterations: int = 5, verbose: bool = True, auto_commit: bool = True, use_docker: bool = True):
        """
        Initialize MLEBench solver for a specific task.
        
        Args:
            task_name: Name of the MLEBench task (e.g., 'spooky-author-identification')
            mlebench_data_dir: Directory containing MLEBench data
            max_iterations: Maximum number of iterations
            verbose: Whether to print detailed progress
            auto_commit: Whether to automatically commit after each iteration
        """
        self.task_name = task_name
        self.mlebench_data_dir = Path(mlebench_data_dir)
        self.verbose = verbose  # Set verbose early for validation
        
        # Task-specific working directory
        work_dir = f"output/{task_name}"
        
        # Validate task data exists
        self._validate_task_data()
        
        # Initialize parent with task-specific work directory
        super().__init__(work_dir, max_iterations, verbose, auto_commit, use_docker)
        
        # Setup task data in working directory
        self._setup_task_data()
        
        # Initialize learnings tracking
        self._setup_learnings_tracking()
    
    def _validate_task_data(self):
        """Validate that the task data exists in MLEBench directory."""
        task_data_dir = self.mlebench_data_dir / self.task_name / "prepared" / "public"
        
        if not self.mlebench_data_dir.exists():
            raise FileNotFoundError(f"MLEBench data directory not found: {self.mlebench_data_dir}")
        
        if not task_data_dir.exists():
            available_tasks = self.list_available_tasks()
            available_str = ", ".join(available_tasks) if available_tasks else "none"
            raise FileNotFoundError(
                f"Task '{self.task_name}' not found in {self.mlebench_data_dir}\n"
                f"Available tasks: {available_str}"
            )
        
        if self.verbose:
            print(f"✅ Found task data: {task_data_dir}")
    
    def _setup_task_data(self):
        """Setup task data in the working directory."""
        # Create working directory and data subdirectory
        work_path = Path(self.work_dir)
        data_dir = work_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Source and destination paths
        task_data_dir = self.mlebench_data_dir / self.task_name / "prepared" / "public"
        
        try:
            # Use symlinks for efficient data access (with fallback to copying)
            if os.name != 'nt':  # Unix-like systems
                # Remove existing symlink/directory if it exists
                if data_dir.exists() and data_dir.is_symlink():
                    data_dir.unlink()
                elif data_dir.exists():
                    import shutil
                    shutil.rmtree(data_dir)
                
                # Create symlink to task data
                data_dir.symlink_to(task_data_dir.resolve())
                if self.verbose:
                    print(f"📁 Created data symlink: {data_dir} -> {task_data_dir}")
            else:
                # Windows: fall back to copying
                import shutil
                if data_dir.exists():
                    shutil.rmtree(data_dir)
                shutil.copytree(task_data_dir, data_dir)
                if self.verbose:
                    print(f"📁 Copied task data: {task_data_dir} -> {data_dir}")
        
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Failed to setup data link, trying copy: {e}")
            
            # Fallback to copying
            import shutil
            if data_dir.exists():
                shutil.rmtree(data_dir)
            shutil.copytree(task_data_dir, data_dir)
            if self.verbose:
                print(f"📁 Copied task data (fallback): {task_data_dir} -> {data_dir}")
        
        # Copy task instructions if available
        instructions_src = self.mlebench_data_dir / self.task_name / "prepared" / "instructions.txt"
        if instructions_src.exists():
            instructions_dst = work_path / "instructions.txt" 
            import shutil
            shutil.copy2(instructions_src, instructions_dst)
            if self.verbose:
                print(f"📄 Copied instructions: {instructions_src} -> {instructions_dst}")
    
    def list_available_tasks(self) -> List[str]:
        """List all available tasks in the MLEBench data directory."""
        if not self.mlebench_data_dir.exists():
            return []
        
        tasks = []
        for task_dir in self.mlebench_data_dir.iterdir():
            if task_dir.is_dir():
                # Check if it has the expected structure
                public_dir = task_dir / "prepared" / "public"
                if public_dir.exists():
                    tasks.append(task_dir.name)
        
        return sorted(tasks)
    
    def _setup_learnings_tracking(self):
        """Setup learnings.md file for tracking what works/doesn't work."""
        work_path = Path(self.work_dir)
        self.learnings_file = work_path / "learnings.md"
        
        # Initialize learnings file if it doesn't exist
        if not self.learnings_file.exists():
            initial_content = f"""# Learnings for {self.task_name}

This document tracks what approaches work and don't work for this ML task.

## Task Overview
- **Task**: {self.task_name}
- **Started**: {self._get_timestamp()}
- **Type**: MLEBench task

## What Works ✅

## What Doesn't Work ❌

## Key Insights 💡

## Iteration Log
"""
            self.learnings_file.write_text(initial_content)
            if self.verbose:
                print(f"📝 Created learnings file: {self.learnings_file}")
    
    def _get_timestamp(self):
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def record_learning(self, iteration: int, solution: dict, category: str, insight: str):
        """
        Record a concise, unique learning insight.
        
        Args:
            iteration: Current iteration number
            solution: Solution dictionary with results
            category: 'works', 'doesnt_work', or 'insight'
            insight: The learning insight to record
        """
        # Extract the core learning without iteration-specific details
        core_insight = self._extract_core_insight(insight)
        
        # Read current content
        current_content = self.learnings_file.read_text()
        
        # Check if this insight already exists (avoid duplicates)
        if core_insight.lower() in current_content.lower():
            if self.verbose:
                print(f"📚 Learning already recorded: {core_insight[:50]}...")
            return
        
        cv_score = solution.get('execution_result', {}).get('cv_score')
        success = solution.get('success', False)
        
        # Create concise entry based on category
        if category == 'works' and success and cv_score:
            entry = f"- {core_insight} (CV: {cv_score:.3f})\n"
            section_marker = "## What Works ✅"
        elif category == 'doesnt_work':
            entry = f"- {core_insight}\n"
            section_marker = "## What Doesn't Work ❌"
        else:
            entry = f"- {core_insight}\n"
            section_marker = "## Key Insights 💡"
        
        # Insert after the section header
        section_start = current_content.find(section_marker)
        if section_start != -1:
            # Find the end of section header line
            line_end = current_content.find('\n', section_start) + 1
            next_section = current_content.find('\n## ', line_end)
            
            if next_section == -1:
                # Last section, append at end
                new_content = current_content[:line_end] + entry + current_content[line_end:]
            else:
                # Insert before next section
                new_content = current_content[:line_end] + entry + current_content[line_end:]
        else:
            # Section not found, append at end
            new_content = current_content + f"\n{section_marker}\n{entry}"
        
        self.learnings_file.write_text(new_content)
        
        if self.verbose:
            # Show more context for errors, less for simple insights
            display_length = 150 if any(word in core_insight.lower() for word in ['error', 'failed', 'exception']) else 80
            if len(core_insight) > display_length:
                print(f"📚 Recorded learning: {core_insight[:display_length]}...")
            else:
                print(f"📚 Recorded learning: {core_insight}")
    
    def _extract_core_insight(self, insight: str) -> str:
        """Extract the core learning without iteration-specific noise."""
        # Remove common prefixes and iteration-specific details
        insight = insight.replace("High-performing approach: ", "")
        insight = insight.replace("Decent approach: ", "")
        insight = insight.replace("Missing dependency: ", "")
        insight = insight.replace("Data access issue: ", "")
        insight = insight.replace("Code generation issue: ", "")
        insight = insight.replace("Solution runtime error ", "")
        insight = insight.replace("Execution failure: ", "")
        insight = insight.replace("Framework issue: ", "")
        insight = insight.replace("General failure: ", "")
        
        # Remove iteration numbers and timestamps
        import re
        insight = re.sub(r'Iteration \d+[:\-\s]*', '', insight)
        insight = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '', insight)
        insight = re.sub(r'\(CV[=:]?[\d\.]+\)', '', insight)  # Remove (CV=0.123) or (CV:0.123)
        insight = re.sub(r'CV[=:]?[\d\.]+', '', insight)      # Remove CV=0.123 or CV:0.123
        
        # Clean up and limit length
        insight = insight.strip().rstrip('...')
        if len(insight) > 100:
            insight = insight[:97] + "..."
        
        return insight
    
    def analyze_and_learn_from_iteration(self, iteration: int, solution: dict):
        """
        Analyze the iteration results and automatically extract learnings.
        
        Args:
            iteration: Current iteration number  
            solution: Solution dictionary with results
        """
        if not solution:
            return
        
        success = solution.get('success', False)
        cv_score = solution.get('cv_score')
        error = solution.get('error', '')
        explanation = solution.get('explanation', '')
        
        # Auto-extract insights based on results
        if success and cv_score:
            if cv_score > 0.8:
                insight = f"High-performing approach: {explanation[:100]}... (CV={cv_score:.4f})"
                self.record_learning(iteration, solution, 'works', insight)
            elif cv_score > 0.7:
                insight = f"Decent approach: {explanation[:100]}... (CV={cv_score:.4f})"
                self.record_learning(iteration, solution, 'insight', insight)
        
        elif not success:
            # Distinguish between framework errors and solution errors
            if solution.get('framework_error'):
                insight = f"Framework issue: {error[:200]}..."
                self.record_learning(iteration, solution, 'doesnt_work', insight)
            elif 'import' in error.lower() or 'modulenotfounderror' in error.lower():
                insight = f"Missing dependency: {error[:200]}..."
                self.record_learning(iteration, solution, 'doesnt_work', insight)
            elif 'file not found' in error.lower() or 'no such file' in error.lower():
                insight = f"Data access issue: {error[:200]}..."
                self.record_learning(iteration, solution, 'doesnt_work', insight)
            elif 'syntax' in error.lower() or 'syntaxerror' in error.lower():
                insight = f"Code generation issue: {error[:200]}..."
                self.record_learning(iteration, solution, 'doesnt_work', insight)
            elif 'traceback' in error.lower() and 'error:' in error.lower():
                # Extract the actual error type and message
                import re
                match = re.search(r'(\w+Error):\s*([^|]+)', error)
                if match:
                    error_type, error_msg = match.groups()
                    insight = f"Solution runtime error ({error_type}): {error_msg.strip()[:150]}..."
                else:
                    insight = f"Solution execution failed: {error[:200]}..."
                self.record_learning(iteration, solution, 'doesnt_work', insight)
            elif error.strip():
                insight = f"Execution failure: {error[:200]}..."
                self.record_learning(iteration, solution, 'doesnt_work', insight)
            else:
                insight = f"Unknown failure (no error details captured)"
                self.record_learning(iteration, solution, 'doesnt_work', insight)
    
    def run_intelligent_iteration(self, iteration: int) -> dict:
        """Run iteration with learnings tracking."""
        
        # Run the parent's intelligent iteration
        solution = super().run_intelligent_iteration(iteration)
        
        # Analyze and learn from this iteration
        self.analyze_and_learn_from_iteration(iteration, solution)
        
        return solution


def get_mlebench_data_dir() -> str:
    """Get MLEBench data directory from environment or default."""
    # Try environment variable first
    data_dir = os.getenv("MLE_BENCH_DATA_DIR")
    if data_dir:
        return data_dir
    
    # Fall back to .mlebench in current directory
    fallback_dir = ".mlebench"
    if not os.path.exists(fallback_dir):
        print(f"⚠️  MLEBench data directory not found. Set MLE_BENCH_DATA_DIR or create {fallback_dir}/")
    
    return fallback_dir


def list_available_tasks(mlebench_data_dir: str) -> List[str]:
    """List available tasks without creating a solver instance."""
    data_path = Path(mlebench_data_dir)
    if not data_path.exists():
        return []
    
    tasks = []
    for task_dir in data_path.iterdir():
        if task_dir.is_dir():
            public_dir = task_dir / "prepared" / "public"
            if public_dir.exists():
                tasks.append(task_dir.name)
    
    return sorted(tasks)


def main():
    """Main entry point for MLEBench solver."""
    parser = argparse.ArgumentParser(description="MLEBench Task Solver")
    
    # Task selection
    parser.add_argument("--task", "-t", help="MLEBench task name (e.g., 'spooky-author-identification')")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks and exit")
    
    # Data directory
    parser.add_argument("--mlebench-data-dir", help="MLEBench data directory", 
                       default=get_mlebench_data_dir())
    
    # Solver options
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum iterations")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    parser.add_argument("--no-auto-commit", action="store_true", help="Disable automatic commits")
    parser.add_argument("--show-history", action="store_true", help="Show solution history and exit")
    parser.add_argument("--no-docker", action="store_true", help="Use virtual environment instead of Docker")
    
    args = parser.parse_args()
    
    # Handle list tasks
    if args.list_tasks:
        print("Available MLEBench tasks:")
        print("-" * 40)
        
        tasks = list_available_tasks(args.mlebench_data_dir)
        if tasks:
            for task in tasks:
                print(f"  {task}")
        else:
            print(f"  No tasks found in {args.mlebench_data_dir}")
            print(f"  Set MLE_BENCH_DATA_DIR environment variable or use --mlebench-data-dir")
        
        return
    
    # Validate task argument
    if not args.task:
        print("Error: --task argument is required")
        print("Use --list-tasks to see available tasks")
        sys.exit(1)
    
    try:
        # Create and run solver
        solver = MLEBenchSolver(
            task_name=args.task,
            mlebench_data_dir=args.mlebench_data_dir,
            max_iterations=args.max_iterations,
            verbose=not args.quiet,
            auto_commit=not args.no_auto_commit,
            use_docker=not args.no_docker
        )
        
        if args.show_history:
            solver.show_solution_history()
            return
        
        # Setup stdout logging
        log_file = solver.work_dir / "solver_session.log"
        original_stdout = sys.stdout
        
        class TeeOutput:
            """Tee stdout to both console and log file"""
            def __init__(self, file):
                self.file = file
                self.console = original_stdout
                
            def write(self, text):
                self.console.write(text)
                self.file.write(text)
                self.file.flush()
                
            def flush(self):
                self.console.flush()
                self.file.flush()
        
        print(f"🔧 MLEBench Solver - Task: {args.task}")
        print("=" * 60)
        print(f"📝 Logging session to: {log_file}")
        
        # Start logging to file
        with open(log_file, 'w') as f:
            # Write header to log
            from datetime import datetime
            f.write(f"MLEBench Solver Session Log\n")
            f.write(f"Task: {args.task}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Max iterations: {args.max_iterations}\n")
            f.write("=" * 60 + "\n\n")
            f.flush()
            
            # Redirect stdout to tee
            sys.stdout = TeeOutput(f)
            
            try:
                summary = solver.run()
            finally:
                # Restore original stdout
                sys.stdout = original_stdout
        
        # Show final results
        if solver.verbose and solver.repo:
            print(f"\n📊 Final Repository State for {args.task}:")
            solver.show_solution_history()
        
        # Exit codes based on performance
        best_score = summary.get("best_cv_score")
        if best_score and best_score > 0.8:
            print(f"🎉 Excellent performance: {best_score:.4f}")
            sys.exit(0)
        elif best_score:
            print(f"✅ Reasonable performance: {best_score:.4f}")
            sys.exit(1)
        else:
            print("❌ No successful solutions found")
            sys.exit(2)
            
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ MLEBench solver failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()