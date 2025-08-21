#!/usr/bin/env python3
"""
Main outer loop orchestrator for the standalone solver.
Coordinates between Claude Code and Python evaluation.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import our modules
from data_setup import setup_working_directory, prepare_spooky_author_data, create_instructions_file
from claude_interface import ClaudeInterface, extract_code_from_response, validate_python_code
from prompt_templates import (
    prepare_draft_prompt, prepare_improve_prompt, 
    load_task_description, generate_data_overview
)
from code_evaluator import CodeEvaluator

class Solution:
    """Represents a solution attempt."""
    
    def __init__(self, iteration: int, code: str, explanation: str, 
                 execution_result: Dict[str, Any], metadata: Dict[str, Any]):
        self.iteration = iteration
        self.code = code
        self.explanation = explanation
        self.execution_result = execution_result
        self.metadata = metadata
        self.timestamp = time.time()
    
    @property
    def cv_score(self) -> Optional[float]:
        """Get cross-validation score."""
        return self.execution_result.get("cv_score")
    
    @property
    def success(self) -> bool:
        """Check if solution executed successfully."""
        return self.execution_result.get("success", False)
    
    @property
    def has_submission(self) -> bool:
        """Check if solution created submission file."""
        return self.execution_result.get("submission_exists", False)
    
    def summary(self) -> str:
        """Get a summary of this solution."""
        status = "✓" if self.success else "✗"
        score_str = f"{self.cv_score:.4f}" if self.cv_score else "N/A"
        time_str = f"{self.execution_result.get('execution_time', 0):.1f}s"
        return f"Iter {self.iteration}: {status} Score={score_str} Time={time_str}"

class StandaloneSolver:
    """Main solver orchestrator."""
    
    def __init__(self, work_dir: str = "output/working", max_iterations: int = 5, 
                 verbose: bool = True):
        """
        Initialize the standalone solver.
        
        Args:
            work_dir: Working directory for the solver
            max_iterations: Maximum number of iterations to run
            verbose: Whether to print detailed progress
        """
        self.work_dir = Path(work_dir)
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Initialize components
        self.claude = ClaudeInterface(timeout_secs=300)
        self.evaluator = CodeEvaluator(self.work_dir, timeout_secs=600)
        
        # Solution tracking
        self.solutions: List[Solution] = []
        self.memory = ""
        
        # Session metadata
        self.session_start = time.time()
        self.session_id = f"session_{int(self.session_start)}"
    
    def setup_data(self):
        """Set up the working directory and data."""
        if self.verbose:
            print("Setting up working directory and data...")
        
        # Create working directory structure
        setup_working_directory(self.work_dir.parent)
        
        # Prepare competition data
        prepare_spooky_author_data(self.work_dir)
        create_instructions_file(self.work_dir)
        
        if self.verbose:
            print(f"✓ Data setup complete in {self.work_dir}")
    
    def load_task_info(self) -> tuple[str, str]:
        """Load task description and data overview."""
        task_description = load_task_description(self.work_dir)
        data_overview = generate_data_overview(self.work_dir)
        return task_description, data_overview
    
    def update_memory(self, solution: Solution):
        """Update memory with the latest solution."""
        memory_entry = f"""
## Iteration {solution.iteration}
**Idea**: {solution.explanation[:200]}...
**Result**: {'Success' if solution.success else 'Failed'} 
**CV Score**: {solution.cv_score if solution.cv_score else 'N/A'}
**Execution Time**: {solution.execution_result.get('execution_time', 0):.1f}s
"""
        if solution.execution_result.get("stderr"):
            memory_entry += f"**Errors**: {solution.execution_result['stderr'][:100]}...\n"
        
        self.memory += memory_entry
    
    def run_iteration(self, iteration: int) -> Solution:
        """
        Run a single iteration of the solver.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Solution object representing this iteration
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration}")
            print(f"{'='*60}")
        
        # Load task information
        task_description, data_overview = self.load_task_info()
        
        # Prepare prompt
        if self.verbose:
            print("Preparing prompt for Claude...")
        
        prompt = prepare_draft_prompt(
            iteration=iteration,
            task_description=task_description,
            data_overview=data_overview,
            memory=self.memory
        )
        
        # Query Claude
        if self.verbose:
            print(f"Querying Claude ({len(prompt)} chars)...")
        
        response, success, claude_metadata = self.claude.query(prompt, verbose=self.verbose)
        
        if not success:
            if self.verbose:
                print("✗ Claude query failed")
            
            # Create a failed solution
            return Solution(
                iteration=iteration,
                code="",
                explanation="Claude query failed",
                execution_result={"success": False, "stderr": "Claude query failed"},
                metadata=claude_metadata
            )
        
        # Extract code and explanation
        code, explanation = extract_code_from_response(response)
        
        if not code:
            if self.verbose:
                print("✗ No code extracted from Claude response")
            
            return Solution(
                iteration=iteration,
                code="",
                explanation=explanation or "No explanation provided",
                execution_result={"success": False, "stderr": "No code extracted"},
                metadata=claude_metadata
            )
        
        if self.verbose:
            print(f"✓ Extracted code ({len(code)} chars)")
            print(f"✓ Extracted explanation ({len(explanation)} chars)")
        
        # Validate code syntax
        is_valid, error = validate_python_code(code)
        if not is_valid:
            if self.verbose:
                print(f"✗ Code validation failed: {error}")
            
            return Solution(
                iteration=iteration,
                code=code,
                explanation=explanation,
                execution_result={"success": False, "stderr": f"Syntax error: {error}"},
                metadata=claude_metadata
            )
        
        # Execute code
        if self.verbose:
            print("Executing code...")
        
        execution_result = self.evaluator.execute_code(code, verbose=self.verbose)
        
        # Create solution object
        solution = Solution(
            iteration=iteration,
            code=code,
            explanation=explanation,
            execution_result=execution_result,
            metadata=claude_metadata
        )
        
        # Print iteration summary
        if self.verbose:
            print(f"\n{solution.summary()}")
            
            if solution.success and solution.cv_score:
                print(f"✓ Successfully generated solution with CV score: {solution.cv_score:.4f}")
            elif solution.success:
                print("✓ Code executed successfully but no CV score detected")
            else:
                print("✗ Execution failed")
                if execution_result.get("stderr"):
                    print(f"  Error: {execution_result['stderr'][:100]}...")
        
        return solution
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete solving process.
        
        Returns:
            Dictionary with session results
        """
        if self.verbose:
            print("🚀 Starting Standalone Solver")
            print(f"Session ID: {self.session_id}")
            print(f"Max iterations: {self.max_iterations}")
        
        # Setup data
        self.setup_data()
        
        # Run iterations
        for iteration in range(1, self.max_iterations + 1):
            try:
                solution = self.run_iteration(iteration)
                self.solutions.append(solution)
                self.update_memory(solution)
                
                # Early stopping if we get a really good solution
                if solution.success and solution.cv_score and solution.cv_score > 0.95:
                    if self.verbose:
                        print(f"\n🎉 Excellent solution found (CV={solution.cv_score:.4f}), stopping early!")
                    break
                    
            except KeyboardInterrupt:
                if self.verbose:
                    print("\n⚠️  Interrupted by user")
                break
            except Exception as e:
                if self.verbose:
                    print(f"\n❌ Error in iteration {iteration}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return self.summarize_session()
    
    def summarize_session(self) -> Dict[str, Any]:
        """Summarize the complete session."""
        session_time = time.time() - self.session_start
        
        # Find best solution
        successful_solutions = [s for s in self.solutions if s.success and s.cv_score]
        best_solution = None
        
        if successful_solutions:
            best_solution = max(successful_solutions, key=lambda s: s.cv_score)
        
        # Create summary
        summary = {
            "session_id": self.session_id,
            "total_time": session_time,
            "total_iterations": len(self.solutions),
            "successful_solutions": len([s for s in self.solutions if s.success]),
            "solutions_with_scores": len([s for s in self.solutions if s.cv_score]),
            "best_cv_score": best_solution.cv_score if best_solution else None,
            "best_iteration": best_solution.iteration if best_solution else None,
            "claude_calls": self.claude.call_count,
            "evaluations": self.evaluator.evaluation_count
        }
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("SESSION SUMMARY")
            print(f"{'='*60}")
            print(f"Total time: {session_time:.1f}s")
            print(f"Iterations: {summary['total_iterations']}")
            print(f"Successful executions: {summary['successful_solutions']}")
            print(f"Solutions with CV scores: {summary['solutions_with_scores']}")
            
            if best_solution:
                print(f"Best solution: Iteration {best_solution.iteration} (CV={best_solution.cv_score:.4f})")
            else:
                print("No successful solutions with CV scores")
            
            print(f"\nAll iterations:")
            for solution in self.solutions:
                print(f"  {solution.summary()}")
        
        return summary
    
    def save_results(self, output_file: str = None):
        """Save session results to JSON file."""
        if output_file is None:
            output_file = f"results_{self.session_id}.json"
        
        # Convert solutions to serializable format
        solutions_data = []
        for solution in self.solutions:
            solutions_data.append({
                "iteration": solution.iteration,
                "explanation": solution.explanation,
                "code": solution.code,
                "execution_result": solution.execution_result,
                "metadata": solution.metadata,
                "timestamp": solution.timestamp
            })
        
        results = {
            "session_summary": self.summarize_session(),
            "solutions": solutions_data,
            "memory": self.memory
        }
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        if self.verbose:
            print(f"\n💾 Results saved to {output_file}")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Standalone ML Solver using Claude Code")
    parser.add_argument("--work-dir", default="output/working", 
                       help="Working directory for solver")
    parser.add_argument("--max-iterations", type=int, default=5,
                       help="Maximum number of iterations")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce verbosity")
    parser.add_argument("--save-results", 
                       help="Save results to specified JSON file")
    
    args = parser.parse_args()
    
    # Create solver
    solver = StandaloneSolver(
        work_dir=args.work_dir,
        max_iterations=args.max_iterations,
        verbose=not args.quiet
    )
    
    try:
        # Run solver
        summary = solver.run()
        
        # Save results if requested
        if args.save_results:
            solver.save_results(args.save_results)
        
        # Exit with appropriate code
        best_score = summary.get("best_cv_score")
        if best_score and best_score > 0.8:
            sys.exit(0)  # Success
        elif best_score:
            sys.exit(1)  # Partial success
        else:
            sys.exit(2)  # No successful solutions
            
    except KeyboardInterrupt:
        print("\n⚠️  Solver interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Solver failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()