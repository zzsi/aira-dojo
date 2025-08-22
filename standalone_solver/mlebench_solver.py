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
from task_manager import TaskManager, get_mlebench_data_dir, list_available_tasks
from learning_tracker import LearningTracker


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
            use_docker: Whether to use Docker for execution
        """
        self.task_name = task_name
        self.mlebench_data_dir = Path(mlebench_data_dir)
        self.verbose = verbose  # Set verbose early for validation
        
        # Task-specific working directory
        work_dir = f"output/{task_name}"
        
        # Initialize task manager
        self.task_manager = TaskManager(task_name, mlebench_data_dir, work_dir)
        
        # Validate task data exists
        self.task_manager.validate_task_data(verbose)
        
        # Initialize parent with task-specific work directory
        super().__init__(work_dir, max_iterations, verbose, auto_commit, use_docker)
        
        # Setup task data in working directory
        self.task_manager.setup_task_data(verbose)
        
        # Initialize learning tracker
        self.learning_tracker = LearningTracker(work_dir)
        
        if self.verbose:
            task_info = self.task_manager.get_task_info()
            print(f"🎯 Initialized MLEBench solver for task: {task_name}")
            print(f"📂 Working directory: {work_dir}")
            print(f"📊 Data files: {task_info.get('csv_files', [])}")
            if task_info.get('total_size_mb'):
                print(f"💾 Data size: {task_info['total_size_mb']} MB")
    
    def list_available_tasks(self) -> List[str]:
        """List all available MLEBench tasks."""
        return self.task_manager.list_available_tasks()
    
    def record_learning(self, iteration: int, solution: dict, category: str, insight: str):
        """
        Record a learning from an iteration.
        
        Args:
            iteration: The iteration number
            solution: The solution dictionary containing results
            category: Category of learning (success, error, insight, pattern)
            insight: The insight text to record
        """
        self.learning_tracker.record_learning(iteration, solution, category, insight)
        
        if self.verbose:
            print(f"📚 Recorded learning: {insight[:60]}{'...' if len(insight) > 60 else ''}")
    
    def analyze_and_learn_from_iteration(self, iteration: int, solution: dict):
        """
        Analyze an iteration and automatically extract learnings.
        
        Args:
            iteration: The iteration number
            solution: The solution dictionary with execution results
        """
        self.learning_tracker.analyze_and_learn_from_iteration(iteration, solution)
    
    def run_intelligent_iteration(self, iteration: int) -> dict:
        """
        Run a single iteration with learning integration.
        
        Args:
            iteration: The iteration number
            
        Returns:
            Dictionary with iteration results
        """
        # Run the parent's iteration
        solution = super().run_intelligent_iteration(iteration)
        
        # Analyze and learn from this iteration
        self.analyze_and_learn_from_iteration(iteration, solution)
        
        return solution


class OutputTee:
    """Utility class to tee output to both console and file."""
    
    def __init__(self, file_path: Path):
        self.file = open(file_path, 'w')
        self.stdout = sys.stdout
    
    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()
    
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()


def main():
    """Main entry point for MLEBench solver."""
    parser = argparse.ArgumentParser(description="MLEBench Task Solver")
    
    # Task selection
    parser.add_argument("--task", type=str, help="MLEBench task name (e.g., 'spooky-author-identification')")
    parser.add_argument("--list-tasks", action="store_true", help="List available MLEBench tasks")
    
    # Data location
    parser.add_argument("--mlebench-data-dir", type=str, 
                       help="Directory containing MLEBench data (default: auto-detect)")
    
    # Execution parameters
    parser.add_argument("--max-iterations", type=int, default=5, 
                       help="Maximum number of iterations (default: 5)")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Verbose output (default: True)")
    parser.add_argument("--quiet", action="store_true", 
                       help="Reduce output verbosity")
    parser.add_argument("--no-auto-commit", action="store_true",
                       help="Disable automatic git commits")
    parser.add_argument("--no-docker", action="store_true", 
                       help="Use virtual environment instead of Docker")
    
    # Output options
    parser.add_argument("--save-output", type=str,
                       help="Save all output to specified file")
    
    args = parser.parse_args()
    
    # Handle verbosity
    verbose = args.verbose and not args.quiet
    
    # Get MLEBench data directory
    mlebench_data_dir = args.mlebench_data_dir
    if not mlebench_data_dir:
        mlebench_data_dir = get_mlebench_data_dir()
        if verbose:
            print(f"🔍 Auto-detected MLEBench data directory: {mlebench_data_dir}")
    
    # Handle task listing
    if args.list_tasks:
        tasks = list_available_tasks(mlebench_data_dir)
        if tasks:
            print(f"📋 Available MLEBench tasks ({len(tasks)} found):")
            for i, task in enumerate(tasks, 1):
                print(f"  {i:2d}. {task}")
        else:
            print(f"❌ No MLEBench tasks found in: {mlebench_data_dir}")
            print("💡 Make sure the directory contains properly formatted MLEBench data")
        return
    
    # Validate task argument
    if not args.task:
        print("❌ Error: Task name is required. Use --task <task-name> or --list-tasks to see available tasks.")
        sys.exit(1)
    
    # Setup output redirection if requested
    output_tee = None
    if args.save_output:
        output_file = Path(args.save_output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_tee = OutputTee(output_file)
        sys.stdout = output_tee
        if verbose:
            print(f"📝 Saving output to: {output_file}")
    
    try:
        # Create and run solver
        solver = MLEBenchSolver(
            task_name=args.task,
            mlebench_data_dir=mlebench_data_dir,
            max_iterations=args.max_iterations,
            verbose=verbose,
            auto_commit=not args.no_auto_commit,
            use_docker=not args.no_docker
        )
        
        # Run the solver
        summary = solver.run()
        
        # Print summary
        if verbose:
            print(f"\n{'='*60}")
            print("🎊 FINAL SUMMARY")
            print(f"{'='*60}")
            print(f"Task: {args.task}")
            print(f"Total iterations: {summary.get('total_iterations', 0)}")
            print(f"Successful solutions: {summary.get('successful_solutions', 0)}")
            
            best_score = summary.get('best_cv_score')
            if best_score:
                print(f"Best CV score: {best_score:.4f} (iteration {summary.get('best_iteration', 'N/A')})")
            
            # Show learning summary
            learning_summary = solver.learning_tracker.get_learning_summary()
            if learning_summary.get('total_entries', 0) > 0:
                print(f"\n📚 Learning Summary:")
                print(f"   Total insights: {learning_summary['total_entries']}")
                print(f"   Successful patterns: {learning_summary['successful_patterns']}")
                print(f"   Error patterns: {learning_summary['error_patterns']}")
        
        # Exit with appropriate code
        if (summary.get('best_cv_score') or 0) > 0.8:
            sys.exit(0)  # Excellent performance
        elif (summary.get('successful_solutions') or 0) > 0:
            sys.exit(1)  # Some success
        else:
            sys.exit(2)  # No successful solutions
            
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up output redirection
        if output_tee:
            sys.stdout = output_tee.stdout
            output_tee.close()


if __name__ == "__main__":
    main()