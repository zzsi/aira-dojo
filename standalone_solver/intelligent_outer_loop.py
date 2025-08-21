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
from journal_manager import JournalManager

class JournalAnalyzer:
    """Analyzes journals to extract learning patterns."""
    
    def __init__(self, journal_manager: JournalManager):
        self.journal_manager = journal_manager
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """
        Analyze journal patterns to inform strategy.
        
        Returns:
            Dictionary with analysis results
        """
        journals = self.journal_manager.list_journals()
        
        if not journals:
            return {"error": "No journals to analyze"}
        
        analysis = {
            "total_interactions": len(journals),
            "success_rate": 0,
            "common_errors": [],
            "successful_approaches": [],
            "failed_approaches": [],
            "response_time_trend": [],
            "code_extraction_rate": 0,
            "recommendations": []
        }
        
        successful_journals = []
        failed_journals = []
        
        for journal in journals:
            if journal.get('success'):
                successful_journals.append(journal)
            else:
                failed_journals.append(journal)
        
        analysis["success_rate"] = len(successful_journals) / len(journals) if journals else 0
        analysis["code_extraction_rate"] = sum(1 for j in journals if j.get('code_extracted', False)) / len(journals)
        
        # Analyze successful approaches
        analysis["successful_approaches"] = self._extract_successful_patterns(successful_journals)
        
        # Analyze failure patterns
        analysis["common_errors"] = self._extract_error_patterns(failed_journals)
        analysis["failed_approaches"] = self._extract_failed_patterns(failed_journals)
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _extract_successful_patterns(self, successful_journals: List[Dict]) -> List[str]:
        """Extract patterns from successful interactions."""
        patterns = []
        
        for journal in successful_journals:
            try:
                content = Path(journal['path']).read_text()
                
                # Look for successful algorithmic approaches
                if "tfidf" in content.lower():
                    patterns.append("TF-IDF vectorization works well")
                if "logistic regression" in content.lower():
                    patterns.append("Logistic Regression is effective")
                if "cross_val_score" in content.lower():
                    patterns.append("sklearn.cross_val_score is reliable")
                if "leaveoneout" in content.lower():
                    patterns.append("LeaveOneOut CV handles small datasets")
                
            except Exception:
                continue
        
        return list(set(patterns))  # Remove duplicates
    
    def _extract_error_patterns(self, failed_journals: List[Dict]) -> List[str]:
        """Extract common error patterns."""
        error_patterns = defaultdict(int)
        
        for journal in failed_journals:
            try:
                content = Path(journal['path']).read_text()
                
                # Common error patterns
                if "n_splits=5 cannot be greater" in content:
                    error_patterns["Too many CV splits for small dataset"] += 1
                if "ImportError" in content:
                    error_patterns["Missing dependency"] += 1
                if "ModuleNotFoundError" in content:
                    error_patterns["Module not found"] += 1
                if "SyntaxError" in content:
                    error_patterns["Python syntax error"] += 1
                if "FileNotFoundError" in content:
                    error_patterns["File path issue"] += 1
                if "MemoryError" in content:
                    error_patterns["Memory issue"] += 1
                
            except Exception:
                continue
        
        # Return most common errors
        sorted_errors = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)
        return [f"{error} (occurred {count} times)" for error, count in sorted_errors[:5]]
    
    def _extract_failed_patterns(self, failed_journals: List[Dict]) -> List[str]:
        """Extract patterns from failed approaches."""
        patterns = []
        
        for journal in failed_journals:
            try:
                content = Path(journal['path']).read_text()
                
                # Look for problematic approaches
                if "stratifiedkfold" in content.lower() and "n_splits=5" in content.lower():
                    patterns.append("5-fold StratifiedKFold fails on small datasets")
                if "xgboost" in content.lower() and "ImportError" in content:
                    patterns.append("XGBoost not available in environment")
                if "deep learning" in content.lower() and any(word in content for word in ["memory", "timeout"]):
                    patterns.append("Deep learning approaches are too resource intensive")
                
            except Exception:
                continue
        
        return list(set(patterns))
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        success_rate = analysis["success_rate"]
        code_extraction_rate = analysis["code_extraction_rate"]
        common_errors = analysis["common_errors"]
        
        # Success rate recommendations
        if success_rate < 0.3:
            recommendations.append("Low success rate - consider simpler, more explicit prompts")
        elif success_rate > 0.8:
            recommendations.append("High success rate - can try more complex approaches")
        
        # Code extraction recommendations
        if code_extraction_rate < 0.7:
            recommendations.append("Poor code extraction - emphasize code block formatting in prompts")
        
        # Error-specific recommendations
        for error in common_errors:
            if "CV splits" in error:
                recommendations.append("Always specify LeaveOneOut or n_splits=3 for small datasets")
            elif "dependency" in error.lower():
                recommendations.append("Stick to standard libraries: sklearn, pandas, numpy")
            elif "syntax" in error.lower():
                recommendations.append("Add explicit syntax validation examples to prompts")
        
        return recommendations

class IntelligentPolicy:
    """Intelligent prompting policy based on journal analysis."""
    
    def __init__(self, analyzer: JournalAnalyzer):
        self.analyzer = analyzer
        self.analysis = None
        self.update_analysis()
    
    def update_analysis(self):
        """Update the analysis from journals."""
        self.analysis = self.analyzer.analyze_patterns()
    
    def adapt_prompt_strategy(self, iteration: int, base_prompt: str) -> str:
        """
        Adapt the prompt based on learned patterns.
        
        Args:
            iteration: Current iteration number
            base_prompt: Base prompt to adapt
            
        Returns:
            Enhanced prompt with learned insights
        """
        if not self.analysis or "error" in self.analysis:
            return base_prompt  # No analysis available
        
        # Build adaptive additions
        adaptations = []
        
        # Add success patterns
        if self.analysis["successful_approaches"]:
            successful_text = "\\n".join(f"- {approach}" for approach in self.analysis["successful_approaches"][:3])
            adaptations.append(f"""
**LEARNED SUCCESS PATTERNS (use these approaches):**
{successful_text}
""")
        
        # Add error avoidance
        if self.analysis["common_errors"]:
            error_text = "\\n".join(f"- AVOID: {error}" for error in self.analysis["common_errors"][:3])
            adaptations.append(f"""
**AVOID THESE COMMON ERRORS:**
{error_text}
""")
        
        # Add specific recommendations
        if self.analysis["recommendations"]:
            rec_text = "\\n".join(f"- {rec}" for rec in self.analysis["recommendations"][:3])
            adaptations.append(f"""
**SPECIFIC RECOMMENDATIONS:**
{rec_text}
""")
        
        # Insert adaptations before the response format section
        if adaptations and "**RESPONSE FORMAT FOR IMPLEMENTATION**:" in base_prompt:
            adaptation_block = "\\n".join(adaptations)
            enhanced_prompt = base_prompt.replace(
                "**RESPONSE FORMAT FOR IMPLEMENTATION**:",
                f"{adaptation_block}\\n\\n**RESPONSE FORMAT FOR IMPLEMENTATION**:"
            )
            return enhanced_prompt
        
        return base_prompt
    
    def should_stop_early(self, current_results: List[Dict]) -> Tuple[bool, str]:
        """
        Determine if we should stop iterating early.
        
        Args:
            current_results: List of solution results so far
            
        Returns:
            Tuple of (should_stop, reason)
        """
        if not current_results:
            return False, ""
        
        # Check for excellent results
        best_score = max((r.get('execution_result', {}).get('cv_score', 0) or 0) for r in current_results)
        if best_score >= 0.95:
            return True, f"Excellent score achieved: {best_score:.4f}"
        
        # Check for repeated failures (based on journal analysis)
        if self.analysis and self.analysis["success_rate"] < 0.2 and len(current_results) >= 3:
            return True, "Low success rate detected, stopping to avoid wasted compute"
        
        # Check for convergence (similar approaches not improving)
        recent_scores = [r.get('execution_result', {}).get('cv_score') for r in current_results[-3:]]
        recent_scores = [s for s in recent_scores if s is not None]
        
        if len(recent_scores) >= 3:
            score_variance = max(recent_scores) - min(recent_scores)
            if score_variance < 0.05:  # Very small improvement
                return True, f"Convergence detected (variance: {score_variance:.4f})"
        
        return False, ""
    
    def adapt_evaluation_focus(self) -> Dict[str, Any]:
        """
        Determine what to focus on during evaluation based on patterns.
        
        Returns:
            Dictionary with evaluation focus areas
        """
        focus = {
            "check_cv_method": True,
            "validate_imports": True,
            "monitor_execution_time": True,
            "check_data_loading": True
        }
        
        if not self.analysis or "error" in self.analysis:
            return focus
        
        # Adapt based on common errors
        for error in self.analysis.get("common_errors", []):
            if "CV splits" in error:
                focus["cv_method_critical"] = True
            if "dependency" in error.lower():
                focus["import_validation_critical"] = True
            if "memory" in error.lower():
                focus["memory_monitoring_critical"] = True
        
        return focus

class IntelligentSolver:
    """Solver with journal-informed intelligence."""
    
    def __init__(self, work_dir: str = "output/working", max_iterations: int = 5, 
                 verbose: bool = True):
        self.work_dir = Path(work_dir)
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Initialize components
        self.claude = ClaudeInterface(timeout_secs=300)
        self.evaluator = ImprovedCodeEvaluator(self.work_dir, timeout_secs=600)
        self.journal_manager = JournalManager(self.work_dir)
        self.analyzer = JournalAnalyzer(self.journal_manager)
        self.policy = IntelligentPolicy(self.analyzer)
        
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
        
        if not success:
            return {"iteration": iteration, "success": False, "error": "Claude query failed"}
        
        # Extract and validate code
        code, explanation = extract_code_from_response(response)
        if not code:
            return {"iteration": iteration, "success": False, "error": "No code extracted"}
        
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
                
                # Intelligent early stopping
                should_stop, reason = self.policy.should_stop_early(self.solutions)
                if should_stop:
                    if self.verbose:
                        print(f"\\n🛑 Intelligent early stopping: {reason}")
                    break
                    
            except KeyboardInterrupt:
                if self.verbose:
                    print("\\n⚠️  Interrupted by user")
                break
            except Exception as e:
                if self.verbose:
                    print(f"\\n❌ Error in iteration {iteration}: {e}")
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
            memory_entry += f"**Error**: {error}\\n"
        
        self.memory += memory_entry
    
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

def main():
    """Main entry point for intelligent solver."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Intelligent ML Solver with Journal Learning")
    parser.add_argument("--work-dir", default="output/working", help="Working directory")
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum iterations")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    
    args = parser.parse_args()
    
    solver = IntelligentSolver(
        work_dir=args.work_dir,
        max_iterations=args.max_iterations,
        verbose=not args.quiet
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