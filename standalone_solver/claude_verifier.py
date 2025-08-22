#!/usr/bin/env python3
"""
Claude-based execution result verifier.
Offloads analysis that requires ML expertise to Claude.
"""

from typing import Dict, Any, Optional, Tuple
from claude_interface import ClaudeInterface


class ClaudeVerifier:
    """Uses Claude to verify and analyze execution results."""
    
    def __init__(self, timeout_secs: int = 300):
        self.claude = ClaudeInterface(timeout_secs=timeout_secs)
    
    def verify_execution_result(self, execution_result: Dict[str, Any], 
                              task_description: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Ask Claude to verify if an execution result makes sense.
        
        Args:
            execution_result: The result from code execution
            task_description: Description of the ML task
            verbose: Whether to print details
            
        Returns:
            Dictionary with verification results
        """
        prompt = self._create_verification_prompt(execution_result, task_description)
        
        if verbose:
            print("🔍 Asking Claude to verify execution results...")
        
        response, success, metadata = self.claude.query(prompt, verbose=False)
        
        if not success:
            return {
                "verification_success": False,
                "error": "Claude verification failed",
                "is_valid_result": True,  # Default to valid if can't verify
                "should_continue": True
            }
        
        # Parse Claude's analysis
        analysis = self._parse_verification_response(response)
        analysis["verification_success"] = True
        analysis["claude_response"] = response
        
        return analysis
    
    def should_stop_early(self, results_history: list, task_description: str,
                         verbose: bool = False) -> Tuple[bool, str]:
        """
        Ask Claude whether to stop early based on results history.
        
        Args:
            results_history: List of execution results
            task_description: Description of the ML task
            verbose: Whether to print details
            
        Returns:
            Tuple of (should_stop, reason)
        """
        if not results_history:
            return False, ""
        
        prompt = self._create_early_stopping_prompt(results_history, task_description)
        
        if verbose:
            print("🤔 Asking Claude about early stopping...")
        
        response, success, metadata = self.claude.query(prompt, verbose=False)
        
        if not success:
            return False, "Claude early stopping analysis failed"
        
        # Parse Claude's decision
        should_stop = "YES_STOP" in response.upper()
        
        # Extract reason
        reason_start = response.find("REASON:")
        if reason_start != -1:
            reason = response[reason_start + 7:].strip().split('\n')[0]
        else:
            reason = "Claude recommended stopping" if should_stop else ""
        
        return should_stop, reason
    
    def _create_verification_prompt(self, execution_result: Dict[str, Any], 
                                   task_description: str) -> str:
        """Create prompt for verifying execution results."""
        
        success = execution_result.get('success', False)
        cv_score = execution_result.get('cv_score')
        return_code = execution_result.get('return_code', 0)
        stdout = execution_result.get('stdout', '')
        stderr = execution_result.get('stderr', '')
        
        return f"""You are a Kaggle Grandmaster analyzing execution results for validation.

TASK: {task_description}

EXECUTION RESULT:
- Success: {success}
- Return Code: {return_code}
- CV Score: {cv_score}
- Stdout ({len(stdout)} chars): {stdout[:1000]}{'...' if len(stdout) > 1000 else ''}
- Stderr ({len(stderr)} chars): {stderr[:500]}{'...' if len(stderr) > 500 else ''}

ANALYSIS NEEDED:
1. Is the CV score reasonable for this type of task?
2. Is the validation methodology appropriate?
3. Are there any red flags in the execution?
4. Should we trust this result for comparison?

Respond in this format:
VALID_RESULT: [YES/NO]
CV_APPROPRIATE: [YES/NO]
SCORE_REASONABLE: [YES/NO]
CONTINUE_ITERATING: [YES/NO]
ISSUES: [List any problems you see]
CONFIDENCE: [HIGH/MEDIUM/LOW]

Brief analysis in 1-2 sentences."""
    
    def _create_early_stopping_prompt(self, results_history: list, 
                                     task_description: str) -> str:
        """Create prompt for early stopping decision."""
        
        # Summarize recent results
        summary_lines = []
        for i, result in enumerate(results_history[-5:], 1):  # Last 5 results
            success = result.get('success', False)
            cv_score = result.get('execution_result', {}).get('cv_score', 'N/A')
            summary_lines.append(f"Iteration {i}: {'Success' if success else 'Failed'}, CV={cv_score}")
        
        results_summary = '\n'.join(summary_lines)
        
        return f"""You are a Kaggle Grandmaster deciding whether to stop iterating.

TASK: {task_description}

RECENT RESULTS:
{results_summary}

STOPPING CRITERIA TO CONSIDER:
1. Are we getting diminishing returns?
2. Have we achieved a good enough score?
3. Are we stuck in a failure loop?
4. Is the approach fundamentally flawed?

Respond in this format:
DECISION: [YES_STOP/NO_CONTINUE]
REASON: [Brief explanation]
CONFIDENCE: [HIGH/MEDIUM/LOW]

Consider that compute is expensive, but we want good results."""
    
    def _parse_verification_response(self, response: str) -> Dict[str, Any]:
        """Parse Claude's verification response."""
        
        result = {
            "is_valid_result": True,
            "cv_appropriate": True, 
            "score_reasonable": True,
            "should_continue": True,
            "issues": [],
            "confidence": "MEDIUM"
        }
        
        # Parse structured fields
        for line in response.split('\n'):
            line = line.strip().upper()
            if line.startswith('VALID_RESULT:'):
                result["is_valid_result"] = 'YES' in line
            elif line.startswith('CV_APPROPRIATE:'):
                result["cv_appropriate"] = 'YES' in line
            elif line.startswith('SCORE_REASONABLE:'):
                result["score_reasonable"] = 'YES' in line
            elif line.startswith('CONTINUE_ITERATING:'):
                result["should_continue"] = 'YES' in line
            elif line.startswith('CONFIDENCE:'):
                if 'HIGH' in line:
                    result["confidence"] = "HIGH"
                elif 'LOW' in line:
                    result["confidence"] = "LOW"
                else:
                    result["confidence"] = "MEDIUM"
            elif line.startswith('ISSUES:'):
                issues_text = line[7:].strip()
                if issues_text and issues_text != 'NONE':
                    result["issues"] = [issues_text]
        
        return result


def main():
    """Test the Claude verifier."""
    verifier = ClaudeVerifier()
    
    # Test verification
    test_result = {
        "success": False,
        "return_code": 1,
        "cv_score": 0.85,
        "stdout": "Training data shape: (1000, 2)\nMean CV Accuracy: 0.85",
        "stderr": "ValueError: n_splits=5 too many splits"
    }
    
    verification = verifier.verify_execution_result(
        test_result, 
        "Spooky Author Identification - text classification",
        verbose=True
    )
    
    print("Verification result:", verification)
    
    # Test early stopping
    test_history = [
        {"success": True, "execution_result": {"cv_score": 0.75}},
        {"success": True, "execution_result": {"cv_score": 0.76}},
        {"success": False, "execution_result": {"cv_score": None}},
    ]
    
    should_stop, reason = verifier.should_stop_early(
        test_history,
        "Spooky Author Identification",
        verbose=True
    )
    
    print(f"Early stopping: {should_stop}, reason: {reason}")


if __name__ == "__main__":
    main()