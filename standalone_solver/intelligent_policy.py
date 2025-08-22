#!/usr/bin/env python3
"""
Intelligent prompting policy that adapts based on journal analysis.
"""

from typing import List, Dict, Any, Tuple
from journal_analyzer import JournalAnalyzer


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
        
        # Check for excellent results (only from successful executions)
        successful_results = [r for r in current_results if r.get('success', False)]
        if successful_results:
            best_score = max((r.get('execution_result', {}).get('cv_score', 0) or 0) for r in successful_results)
            if best_score >= 0.95 and best_score <= 1.0:  # Valid score range
                return True, f"Excellent score achieved: {best_score:.4f}"
        
        # Check for Claude breakdown (multiple consecutive failures with no code extraction)
        recent_results = current_results[-4:]  # Look at more recent results
        if len(recent_results) >= 4:  # Need 4 attempts before stopping
            no_code_failures = [r for r in recent_results if not r.get('success') and 'No code' in r.get('error', '')]
            if len(no_code_failures) >= 3:  # Need 3 consecutive failures
                return True, "Claude stopped generating code - likely in error state"
        
        # Check for repeated failures (based on journal analysis)
        if self.analysis and self.analysis.get("success_rate", 1.0) < 0.1 and len(current_results) >= 4:
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