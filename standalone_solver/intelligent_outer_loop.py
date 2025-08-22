#!/usr/bin/env python3
"""
Intelligent outer loop that uses journal analysis to inform prompting strategy.
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

# Import our modules
from data_setup import setup_working_directory, prepare_spooky_author_data, create_instructions_file
from claude_interface import ClaudeInterface, extract_code_from_response, validate_python_code
from prompt_templates import (
    prepare_draft_prompt, load_task_description, generate_data_overview
)
from improved_evaluator import ImprovedCodeEvaluator
from docker_evaluator import DockerEvaluator
from journal_manager import JournalManager
from journal_analyzer import JournalAnalyzer
from intelligent_policy import IntelligentPolicy
from claude_verifier import ClaudeVerifier


class IntelligentSolver:
    """Solver with journal-informed intelligence."""
    
    def __init__(self, work_dir: str = "output/working", max_iterations: int = 5, 
                 verbose: bool = True, use_docker: bool = True):
        self.work_dir = Path(work_dir)
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.use_docker = use_docker
        
        # Initialize components
        self.claude = ClaudeInterface(timeout_secs=900)  # 15 minutes for complex ML solutions
        
        # Choose evaluator based on preference
        if use_docker:
            self.evaluator = DockerEvaluator(self.work_dir, timeout_secs=600)
            if self.verbose:
                print("🐳 Using Docker evaluator for dependency management")
        else:
            self.evaluator = ImprovedCodeEvaluator(self.work_dir, timeout_secs=600)
            if self.verbose:
                print("📦 Using virtual environment evaluator")
        
        self.journal_manager = JournalManager(self.work_dir)
        self.analyzer = JournalAnalyzer(self.journal_manager)
        self.policy = IntelligentPolicy(self.analyzer)
        self.verifier = ClaudeVerifier(timeout_secs=300)
        
        # Solution tracking
        self.solutions = []
        self.memory = ""
        
        # Session metadata
        self.session_start = time.time()
        self.session_id = f"session_{int(self.session_start)}"
    
    def setup_data(self):
        """Set up the working directory and data."""
        if self.verbose:
            print("Setting up working directory and data...")
        
        setup_working_directory(self.work_dir.parent)
        prepare_spooky_author_data(self.work_dir)
        create_instructions_file(self.work_dir)
        
        if self.verbose:
            print(f"✓ Data setup complete in {self.work_dir}")
    
    def run_intelligent_iteration(self, iteration: int) -> Dict[str, Any]:
        """Run iteration with intelligent adaptations."""
        
        if self.verbose:
            print(f"\\n{'='*60}")
            print(f"INTELLIGENT ITERATION {iteration}")
            print(f"{'='*60}")
        
        # Update policy analysis
        self.policy.update_analysis()
        
        # Show intelligence insights
        if self.verbose and self.policy.analysis and "error" not in self.policy.analysis:
            analysis = self.policy.analysis
            print(f"🧠 Intelligence: {analysis['success_rate']:.1%} success rate, {len(analysis['recommendations'])} recommendations")
        
        # Load task information
        task_description = load_task_description(self.work_dir)
        data_overview = generate_data_overview(self.work_dir)
        
        # Prepare base prompt
        base_prompt = prepare_draft_prompt(
            iteration=iteration,
            task_description=task_description,
            data_overview=data_overview,
            memory=self.memory
        )
        
        # Apply intelligent adaptations
        adapted_prompt = self.policy.adapt_prompt_strategy(iteration, base_prompt)
        
        if self.verbose and adapted_prompt != base_prompt:
            print("🎯 Applied intelligent prompt adaptations")
        
        # Query Claude with adapted prompt
        if self.verbose:
            print(f"Querying Claude ({len(adapted_prompt)} chars)...")
        
        response, success, claude_metadata = self.claude.query(adapted_prompt, verbose=self.verbose)
        
        # Save all Claude responses for debugging (both successful and failed)
        self._save_claude_response_artifacts(iteration, adapted_prompt, response, success, claude_metadata)
        
        if not success:
            return {"iteration": iteration, "success": False, "error": "Claude query failed"}
        
        # Extract and validate code
        code, explanation = extract_code_from_response(response)
        if not code:
            # Check if this is a problematic short response
            if len(response.strip()) < 100:
                error_msg = f"Claude gave very short response ({len(response)} chars): '{response.strip()[:50]}...'"
            else:
                error_msg = f"No code block found in {len(response)} char response"
            
            return {"iteration": iteration, "success": False, "error": error_msg, "response_length": len(response)}
        
        is_valid, error = validate_python_code(code)
        if not is_valid:
            return {"iteration": iteration, "success": False, "error": f"Syntax error: {error}"}
        
        # Execute with intelligent evaluation focus
        evaluation_focus = self.policy.adapt_evaluation_focus()
        
        if self.verbose and evaluation_focus:
            critical_checks = [k for k, v in evaluation_focus.items() if k.endswith('_critical')]
            if critical_checks:
                print(f"⚠️  Critical checks: {', '.join(critical_checks)}")
        
        # Execute code
        execution_result = self.evaluator.execute_code_with_journaling(
            code=code,
            claude_prompt=adapted_prompt,
            claude_response=response,
            claude_metadata=claude_metadata,
            verbose=self.verbose
        )
        
        # Use Claude to verify the execution result
        task_description = load_task_description(self.work_dir)
        verification = self.verifier.verify_execution_result(
            execution_result, 
            task_description, 
            verbose=self.verbose
        )
        
        if self.verbose and verification.get("verification_success"):
            confidence = verification.get("confidence", "UNKNOWN")
            issues = verification.get("issues", [])
            print(f"🔍 Claude verification: {confidence} confidence")
            if issues:
                print(f"⚠️  Issues identified: {', '.join(issues)}")
        
        # Add verification to execution result
        execution_result["claude_verification"] = verification
        
        # Create solution with intelligence metadata
        solution = {
            "iteration": iteration,
            "success": execution_result["success"],
            "code": code,
            "explanation": explanation,
            "execution_result": execution_result,
            "claude_metadata": claude_metadata,
            "prompt_adapted": adapted_prompt != base_prompt,
            "evaluation_focus": evaluation_focus,
            "timestamp": time.time()
        }
        
        # Add error details for failed executions
        if not execution_result["success"]:
            error_parts = []
            if execution_result.get("stderr"):
                error_parts.append(f"STDERR: {execution_result['stderr']}")
            if execution_result.get("stdout"):
                error_parts.append(f"STDOUT: {execution_result['stdout']}")
            if execution_result.get("return_code") != 0:
                error_parts.append(f"Exit code: {execution_result['return_code']}")
            
            solution["error"] = " | ".join(error_parts) if error_parts else "Unknown execution failure"
        
        return solution
    
    def run(self) -> Dict[str, Any]:
        """Run the intelligent solving process."""
        
        if self.verbose:
            print("🧠 Starting Intelligent Solver with Journal Learning")
            print(f"Session ID: {self.session_id}")
        
        # Setup data
        self.setup_data()
        
        # Run iterations with intelligence
        for iteration in range(1, self.max_iterations + 1):
            try:
                solution = self.run_intelligent_iteration(iteration)
                self.solutions.append(solution)
                self.update_memory(solution)
                
                # Claude-based early stopping decision
                task_description = load_task_description(self.work_dir) 
                should_stop, reason = self.verifier.should_stop_early(
                    self.solutions, 
                    task_description, 
                    verbose=self.verbose
                )
                if should_stop:
                    if self.verbose:
                        print(f"\\n🛑 Claude-based early stopping: {reason}")
                    break
                    
            except KeyboardInterrupt:
                if self.verbose:
                    print("\\n⚠️  Interrupted by user")
                break
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                if self.verbose:
                    print(f"\\n❌ Framework error in iteration {iteration}: {type(e).__name__}: {e}")
                    print(f"Full traceback:\\n{error_details}")
                
                # Create a failed solution record for proper learning
                failed_solution = {
                    "iteration": iteration,
                    "success": False,
                    "error": f"Framework error: {type(e).__name__}: {e}",
                    "framework_error": True,
                    "traceback": error_details,
                    "timestamp": time.time()
                }
                self.solutions.append(failed_solution)
                self.update_memory(failed_solution)
                continue
        
        return self.summarize_intelligent_session()
    
    def update_memory(self, solution: Dict[str, Any]):
        """Update memory with solution info."""
        cv_score = solution.get('execution_result', {}).get('cv_score', 'N/A')
        exec_time = solution.get('execution_result', {}).get('execution_time', 0)
        
        memory_entry = f"""
## Iteration {solution['iteration']}
**Adapted Prompt**: {solution.get('prompt_adapted', False)}
**Result**: {'Success' if solution['success'] else 'Failed'}
**CV Score**: {cv_score}
**Time**: {exec_time:.1f}s
"""
        
        if not solution['success']:
            error = solution.get('error', 'Unknown error')
            
            # Extract key error details for Claude
            if 'execution_result' in solution:
                exec_result = solution['execution_result']
                stderr = exec_result.get('stderr', '')
                
                # Extract meaningful error patterns
                if 'ModuleNotFoundError' in stderr:
                    key_error = self._extract_import_error(stderr)
                elif 'XGBoostError' in stderr:
                    key_error = "XGBoost failed - missing system dependencies (try sklearn alternatives)"
                elif 'Library not loaded' in stderr:
                    key_error = "System library missing - use basic sklearn packages instead"
                elif 'SyntaxError' in stderr:
                    key_error = "Python syntax error in generated code"
                elif 'No code extracted' in error:
                    key_error = "Claude failed to generate valid code - try simpler approach"
                else:
                    # Take first few lines of stderr for context
                    error_lines = stderr.split('\\n')[:3]
                    key_error = ' | '.join(line.strip() for line in error_lines if line.strip())
                
                memory_entry += f"**Error**: {key_error}\\n"
            else:
                memory_entry += f"**Error**: {error}\\n"
        
        self.memory += memory_entry
    
    def _extract_import_error(self, stderr: str) -> str:
        """Extract key information from import errors."""
        import re
        
        # Look for "No module named 'X'"
        match = re.search(r"No module named '([^']+)'", stderr)
        if match:
            missing_module = match.group(1)
            return f"Missing module '{missing_module}' - check dependencies or use alternatives"
        
        # Fallback
        return "Import error - check package dependencies"
    
    def summarize_intelligent_session(self) -> Dict[str, Any]:
        """Summarize the intelligent session."""
        session_time = time.time() - self.session_start
        
        # Find best solution
        successful_solutions = [s for s in self.solutions if s['success'] and s.get('execution_result', {}).get('cv_score')]
        best_solution = max(successful_solutions, key=lambda s: s['execution_result']['cv_score']) if successful_solutions else None
        
        # Intelligence metrics
        adapted_prompts = sum(1 for s in self.solutions if s.get('prompt_adapted', False))
        
        summary = {
            "session_id": self.session_id,
            "total_time": session_time,
            "total_iterations": len(self.solutions),
            "successful_solutions": len([s for s in self.solutions if s['success']]),
            "best_cv_score": best_solution['execution_result']['cv_score'] if best_solution else None,
            "best_iteration": best_solution['iteration'] if best_solution else None,
            "intelligence_metrics": {
                "adapted_prompts": adapted_prompts,
                "learning_enabled": True,
                "policy_recommendations": len(self.policy.analysis.get('recommendations', [])) if self.policy.analysis else 0
            }
        }
        
        if self.verbose:
            print(f"\\n{'='*60}")
            print("INTELLIGENT SESSION SUMMARY")
            print(f"{'='*60}")
            print(f"Total time: {session_time:.1f}s")
            print(f"Iterations: {summary['total_iterations']}")
            print(f"Adapted prompts: {adapted_prompts}/{len(self.solutions)}")
            print(f"Policy recommendations: {summary['intelligence_metrics']['policy_recommendations']}")
            
            if best_solution:
                print(f"Best solution: Iteration {best_solution['iteration']} (CV={best_solution['execution_result']['cv_score']:.4f})")
        
        return summary
    
    def _save_claude_response_artifacts(self, iteration: int, prompt: str, response: str, 
                                      success: bool, metadata: dict):
        """Save Claude response artifacts for debugging all calls (success and failures)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create artifacts directory
        artifacts_dir = self.work_dir.parent / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        # Save prompt
        prompt_file = artifacts_dir / f"prompt_{iteration}_{timestamp}.txt"
        prompt_file.write_text(prompt)
        
        # Save response 
        response_file = artifacts_dir / f"response_{iteration}_{timestamp}.txt"
        response_file.write_text(response)
        
        # Save metadata
        metadata_file = artifacts_dir / f"metadata_{iteration}_{timestamp}.json"
        metadata_with_context = {
            **metadata,
            "iteration": iteration,
            "success": success,
            "response_length": len(response),
            "prompt_length": len(prompt),
            "timestamp": timestamp
        }
        
        import json
        metadata_file.write_text(json.dumps(metadata_with_context, indent=2))
        
        if self.verbose:
            status = "SUCCESS" if success else "FAILED"
            print(f"[Claude Artifacts] Saved {status} call artifacts for iteration {iteration}: {timestamp}")


def main():
    """Main entry point for intelligent solver."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Intelligent ML Solver with Journal Learning")
    parser.add_argument("--work-dir", default="output/working", help="Working directory")
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum iterations")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    parser.add_argument("--no-docker", action="store_true", help="Use virtual environment instead of Docker (default: Docker)")
    
    args = parser.parse_args()
    
    solver = IntelligentSolver(
        work_dir=args.work_dir,
        max_iterations=args.max_iterations,
        verbose=not args.quiet,
        use_docker=not args.no_docker
    )
    
    try:
        summary = solver.run()
        
        # Exit codes based on performance
        best_score = summary.get("best_cv_score")
        if best_score and best_score > 0.8:
            sys.exit(0)
        elif best_score:
            sys.exit(1)
        else:
            sys.exit(2)
            
    except Exception as e:
        print(f"❌ Intelligent solver failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()