#!/usr/bin/env python3
"""
Improved outer loop with data aliasing and Claude journaling.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import our modules
from data_setup import setup_working_directory, prepare_spooky_author_data, create_instructions_file
from claude_interface import ClaudeInterface, extract_code_from_response, validate_python_code
from prompt_templates import (
    prepare_draft_prompt, load_task_description, generate_data_overview
)
from improved_evaluator import ImprovedCodeEvaluator

class ImprovedStandaloneSolver:
    """Enhanced solver with data aliasing and journaling."""
    
    def __init__(self, work_dir: str = "output/working", max_iterations: int = 5, 
                 verbose: bool = True, use_data_aliases: bool = True):
        """
        Initialize the improved standalone solver.
        
        Args:
            work_dir: Working directory for the solver
            max_iterations: Maximum number of iterations to run
            verbose: Whether to print detailed progress
            use_data_aliases: Whether to use symlinks instead of copying data
        """
        self.work_dir = Path(work_dir)
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.use_data_aliases = use_data_aliases
        
        # Initialize components
        self.claude = ClaudeInterface(timeout_secs=300)
        self.evaluator = ImprovedCodeEvaluator(
            self.work_dir, 
            timeout_secs=600, 
            use_data_aliases=use_data_aliases
        )
        
        # Solution tracking
        self.solutions = []
        self.memory = ""
        
        # Session metadata
        self.session_start = time.time()
        self.session_id = f"session_{int(self.session_start)}"
        
        if self.verbose:
            print(f"🔧 Data access mode: {'Symlinks (fast)' if use_data_aliases else 'Copy (safe)'}")
    
    def setup_data(self):
        """Set up the working directory and data."""
        if self.verbose:
            print("Setting up working directory and data...")
        
        # Create working directory structure
        setup_working_directory(self.work_dir.parent)
        
        # Prepare competition data
        prepare_spooky_author_data(self.work_dir)
        create_instructions_file(self.work_dir)
        
        # Check data size for optimization recommendations
        self._analyze_data_size()
        
        if self.verbose:
            print(f"✓ Data setup complete in {self.work_dir}")
    
    def _analyze_data_size(self):
        """Analyze data size and provide optimization recommendations."""
        data_dir = self.work_dir / "data"
        if not data_dir.exists():
            return
        
        total_size = sum(f.stat().st_size for f in data_dir.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        
        if self.verbose:
            print(f"📊 Dataset size: {size_mb:.2f} MB")
            
            if size_mb > 100:  # > 100MB
                if self.use_data_aliases:
                    print("✓ Using symlinks - good choice for large dataset")
                else:
                    print("⚠️  Consider enabling data aliases for large dataset (use_data_aliases=True)")
            elif size_mb < 10:  # < 10MB
                print("ℹ️  Small dataset - data copying is fine")
    
    def run_iteration_with_journaling(self, iteration: int) -> Dict[str, Any]:
        """
        Run iteration with enhanced journaling.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Solution dictionary with enhanced metadata
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
            
            return {
                "iteration": iteration,
                "success": False,
                "error": "Claude query failed",
                "claude_metadata": claude_metadata
            }
        
        # Extract code and explanation
        code, explanation = extract_code_from_response(response)
        
        if not code:
            if self.verbose:
                print("✗ No code extracted from Claude response")
            
            # Still log the interaction for debugging
            self.evaluator.log_claude_interaction(prompt, response, claude_metadata)
            
            return {
                "iteration": iteration,
                "success": False,
                "error": "No code extracted",
                "explanation": explanation,
                "claude_metadata": claude_metadata
            }
        
        if self.verbose:
            print(f"✓ Extracted code ({len(code)} chars)")
            print(f"✓ Extracted explanation ({len(explanation)} chars)")
        
        # Validate code syntax
        is_valid, error = validate_python_code(code)
        if not is_valid:
            if self.verbose:
                print(f"✗ Code validation failed: {error}")
            
            # Log the interaction even for invalid code
            self.evaluator.log_claude_interaction(prompt, response, claude_metadata)
            
            return {
                "iteration": iteration,
                "success": False,
                "error": f"Syntax error: {error}",
                "code": code,
                "explanation": explanation,
                "claude_metadata": claude_metadata
            }
        
        # Execute code with full journaling
        if self.verbose:
            print("Executing code with full journaling...")
        
        execution_result = self.evaluator.execute_code_with_journaling(
            code=code,
            claude_prompt=prompt,
            claude_response=response,
            claude_metadata=claude_metadata,
            verbose=self.verbose
        )
        
        # Create enhanced solution object
        solution = {
            "iteration": iteration,
            "success": execution_result["success"],
            "code": code,
            "explanation": explanation,
            "execution_result": execution_result,
            "claude_metadata": claude_metadata,
            "timestamp": time.time(),
            "data_access_method": execution_result.get("data_access_method", "unknown")
        }
        
        # Print iteration summary
        if self.verbose:
            cv_score = execution_result.get("cv_score")
            exec_time = execution_result.get("execution_time", 0)
            status = "✓" if solution["success"] else "✗"
            score_str = f"{cv_score:.4f}" if cv_score else "N/A"
            
            print(f"\nIter {iteration}: {status} Score={score_str} Time={exec_time:.1f}s")
            
            if solution["success"] and cv_score:
                print(f"✓ Successfully generated solution with CV score: {cv_score:.4f}")
            elif solution["success"]:
                print("✓ Code executed successfully but no CV score detected")
            else:
                print("✗ Execution failed")
                if execution_result.get("stderr"):
                    print(f"  Error: {execution_result['stderr'][:100]}...")
        
        return solution
    
    def load_task_info(self) -> tuple[str, str]:
        """Load task description and data overview."""
        task_description = load_task_description(self.work_dir)
        data_overview = generate_data_overview(self.work_dir)
        return task_description, data_overview
    
    def update_memory(self, solution: Dict[str, Any]):
        """Update memory with the latest solution."""
        memory_entry = f"""
## Iteration {solution['iteration']}
**Idea**: {solution.get('explanation', 'No explanation')[:200]}...
**Result**: {'Success' if solution['success'] else 'Failed'} 
**CV Score**: {solution.get('execution_result', {}).get('cv_score', 'N/A')}
**Execution Time**: {solution.get('execution_result', {}).get('execution_time', 0):.1f}s
**Data Access**: {solution.get('data_access_method', 'unknown')}
"""
        if solution.get('execution_result', {}).get('stderr'):
            stderr = solution['execution_result']['stderr']
            memory_entry += f"**Errors**: {stderr[:100]}...\n"
        
        self.memory += memory_entry
    
    def run(self) -> Dict[str, Any]:
        """Run the complete solving process with enhancements."""
        if self.verbose:
            print("🚀 Starting Enhanced Standalone Solver")
            print(f"Session ID: {self.session_id}")
            print(f"Max iterations: {self.max_iterations}")
            print(f"Journaling: Enabled (timestamps + full responses)")
        
        # Setup data
        self.setup_data()
        
        # Run iterations
        for iteration in range(1, self.max_iterations + 1):
            try:
                solution = self.run_iteration_with_journaling(iteration)
                self.solutions.append(solution)
                self.update_memory(solution)
                
                # Early stopping for excellent solutions
                cv_score = solution.get('execution_result', {}).get('cv_score')
                if solution['success'] and cv_score and cv_score > 0.95:
                    if self.verbose:
                        print(f"\n🎉 Excellent solution found (CV={cv_score:.4f}), stopping early!")
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
        
        return self.summarize_enhanced_session()
    
    def summarize_enhanced_session(self) -> Dict[str, Any]:
        """Summarize session with enhanced metrics."""
        session_time = time.time() - self.session_start
        
        # Find best solution
        successful_solutions = []
        for s in self.solutions:
            if s['success'] and s.get('execution_result', {}).get('cv_score'):
                successful_solutions.append(s)
        
        best_solution = None
        if successful_solutions:
            best_solution = max(successful_solutions, 
                              key=lambda s: s['execution_result']['cv_score'])
        
        # Enhanced summary
        summary = {
            "session_id": self.session_id,
            "total_time": session_time,
            "total_iterations": len(self.solutions),
            "successful_solutions": len([s for s in self.solutions if s['success']]),
            "solutions_with_scores": len(successful_solutions),
            "best_cv_score": best_solution['execution_result']['cv_score'] if best_solution else None,
            "best_iteration": best_solution['iteration'] if best_solution else None,
            "claude_calls": self.claude.call_count,
            "evaluations": self.evaluator.evaluation_count,
            "data_access_method": "symlinks" if self.use_data_aliases else "copy",
            "journals_created": len(list(self.evaluator.journals_dir.glob("*.md")))
        }
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("ENHANCED SESSION SUMMARY")
            print(f"{'='*60}")
            print(f"Total time: {session_time:.1f}s")
            print(f"Iterations: {summary['total_iterations']}")
            print(f"Successful executions: {summary['successful_solutions']}")
            print(f"Solutions with CV scores: {summary['solutions_with_scores']}")
            print(f"Data access: {summary['data_access_method']}")
            print(f"Claude journals: {summary['journals_created']}")
            
            if best_solution:
                best_cv = best_solution['execution_result']['cv_score']
                print(f"Best solution: Iteration {best_solution['iteration']} (CV={best_cv:.4f})")
                
                # Show where to find detailed logs
                print(f"\n📁 Detailed logs available in:")
                print(f"   - {self.work_dir}/journals/ (Claude interactions)")
                print(f"   - {self.work_dir}/artifacts/ (Code & results)")
            else:
                print("No successful solutions with CV scores")
        
        return summary

def main():
    """Main entry point for improved solver."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced ML Solver with Data Aliasing & Journaling")
    parser.add_argument("--work-dir", default="output/working", 
                       help="Working directory for solver")
    parser.add_argument("--max-iterations", type=int, default=5,
                       help="Maximum number of iterations")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce verbosity")
    parser.add_argument("--no-aliases", action="store_true",
                       help="Disable data aliasing (use copying)")
    parser.add_argument("--save-results", 
                       help="Save results to specified JSON file")
    
    args = parser.parse_args()
    
    # Create enhanced solver
    solver = ImprovedStandaloneSolver(
        work_dir=args.work_dir,
        max_iterations=args.max_iterations,
        verbose=not args.quiet,
        use_data_aliases=not args.no_aliases
    )
    
    try:
        # Run solver
        summary = solver.run()
        
        # Save results if requested
        if args.save_results:
            with open(args.save_results, 'w') as f:
                json.dump({
                    "session_summary": summary,
                    "solutions": solver.solutions,
                    "memory": solver.memory
                }, f, indent=2)
            print(f"💾 Enhanced results saved to {args.save_results}")
        
        # Exit with appropriate code
        best_score = summary.get("best_cv_score")
        if best_score and best_score > 0.8:
            sys.exit(0)  # Success
        elif best_score:
            sys.exit(1)  # Partial success
        else:
            sys.exit(2)  # No successful solutions
            
    except KeyboardInterrupt:
        print("\n⚠️  Enhanced solver interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Enhanced solver failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()