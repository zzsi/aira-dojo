#!/usr/bin/env python3
"""
Claude CLI interface for the standalone solver.
Handles communication with Claude Code.
"""

import subprocess
import time
import json
import re
import os
from typing import Tuple, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    pass  # python-dotenv not installed, skip loading

class ClaudeInterface:
    """Interface to Claude CLI."""
    
    def __init__(self, timeout_secs: int = None):
        """
        Initialize Claude interface.
        
        Args:
            timeout_secs: Timeout for Claude CLI calls (defaults to CLAUDE_TIMEOUT_SECS env var or 900s)
        """
        if timeout_secs is None:
            timeout_secs = int(os.getenv('CLAUDE_TIMEOUT_SECS', '900'))
        
        self.timeout_secs = timeout_secs
        self.call_count = 0
    
    def query(self, prompt: str, verbose: bool = False) -> Tuple[str, bool, dict]:
        """
        Send a query to Claude CLI.
        
        Args:
            prompt: The prompt to send to Claude
            verbose: Whether to print debug information
            
        Returns:
            Tuple of (response, success, metadata)
        """
        self.call_count += 1
        
        # Build Claude CLI command
        cmd = [
            "claude",
            "--dangerously-skip-permissions",
            "--permission-mode", "bypassPermissions",
            "--model", "sonnet",
            "--output-format", "text",
            "-p", prompt
        ]
        
        if verbose:
            print(f"[Claude Call {self.call_count}] Sending prompt ({len(prompt)} chars)")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_secs
            )
            
            latency = time.time() - start_time
            success = result.returncode == 0
            
            metadata = {
                "call_number": self.call_count,
                "latency": latency,
                "return_code": result.returncode,
                "prompt_length": len(prompt),
                "response_length": len(result.stdout) if result.stdout else 0
            }
            
            if verbose:
                print(f"[Claude Call {self.call_count}] Response received ({latency:.2f}s, {len(result.stdout)} chars)")
            
            if not success:
                print(f"[Claude Error] Return code: {result.returncode}")
                if result.stderr:
                    print(f"[Claude Error] Stderr: {result.stderr}")
                print(f"[Claude Error] Stdout: {result.stdout}")
            
            return result.stdout, success, metadata
            
        except subprocess.TimeoutExpired:
            latency = time.time() - start_time
            metadata = {
                "call_number": self.call_count,
                "latency": latency,
                "return_code": 124,
                "prompt_length": len(prompt),
                "response_length": 0,
                "error": "timeout"
            }
            
            print(f"[Claude Error] Timeout after {self.timeout_secs}s")
            return "", False, metadata
            
        except Exception as e:
            latency = time.time() - start_time
            metadata = {
                "call_number": self.call_count,
                "latency": latency,
                "return_code": -1,
                "prompt_length": len(prompt),
                "response_length": 0,
                "error": str(e)
            }
            
            print(f"[Claude Error] Exception: {e}")
            return "", False, metadata

def extract_code_from_response(response: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract code and explanation from Claude's response.
    
    Args:
        response: Claude's response text
        
    Returns:
        Tuple of (code, explanation)
    """
    # Look for code blocks (```python ... ``` or ``` ... ```)
    code_pattern = r'```(?:python)?\s*(.*?)\s*```'
    code_matches = re.findall(code_pattern, response, re.DOTALL)
    
    if code_matches:
        # Take the longest code block (most likely to be the main solution)
        code = max(code_matches, key=len).strip()
        
        # Extract explanation (text before first code block)
        explanation_match = re.search(r'^(.*?)```', response, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else ""
        
        return code, explanation
    
    return None, response.strip()

def validate_python_code(code: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that code is syntactically correct Python.
    
    Args:
        code: Python code to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not code:
        return False, "No code provided"
    
    try:
        compile(code, "<string>", "exec")
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Compilation error: {e}"

def test_claude_interface():
    """Test the Claude interface with a simple query."""
    claude = ClaudeInterface()
    
    test_prompt = """Write a simple Python function that adds two numbers. 
    
Format your response as:

# Explanation
Brief explanation here

```python
your code here
```"""
    
    print("Testing Claude interface...")
    response, success, metadata = claude.query(test_prompt, verbose=True)
    
    if success:
        print("\nResponse:")
        print(response)
        
        code, explanation = extract_code_from_response(response)
        if code:
            print(f"\nExtracted code ({len(code)} chars):")
            print(code)
            
            is_valid, error = validate_python_code(code)
            print(f"\nCode validation: {'PASS' if is_valid else 'FAIL'}")
            if error:
                print(f"Error: {error}")
        else:
            print("\nNo code extracted from response")
    else:
        print(f"Claude query failed: {metadata}")

if __name__ == "__main__":
    test_claude_interface()