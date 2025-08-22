#!/usr/bin/env python3
"""
Artifact management for saving execution results, code, and outputs.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class ArtifactManager:
    """Manages saving and loading of execution artifacts."""
    
    def __init__(self, work_dir: Path):
        self.work_dir = Path(work_dir)
        self.artifacts_dir = self.work_dir / "artifacts"
        self.journals_dir = self.work_dir / "journals"
        
        # Create directories (with parents)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.journals_dir.mkdir(parents=True, exist_ok=True)
        
        self.artifact_count = 0
    
    def save_execution_artifacts(self, code: str, execution_result: Dict[str, Any], 
                               verbose: bool = False) -> str:
        """Save execution artifacts and return timestamp."""
        self.artifact_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save solution code
        solution_file = self.artifacts_dir / f"solution_{self.artifact_count}_{timestamp}.py"
        solution_file.write_text(code)
        
        # Save stdout
        if execution_result.get("stdout"):
            stdout_file = self.artifacts_dir / f"stdout_{self.artifact_count}_{timestamp}.log"
            stdout_file.write_text(execution_result["stdout"])
        
        # Save stderr
        if execution_result.get("stderr"):
            stderr_file = self.artifacts_dir / f"stderr_{self.artifact_count}_{timestamp}.log"
            stderr_file.write_text(execution_result["stderr"])
        
        # Save submission file if it exists
        if execution_result.get("submission_content"):
            submission_file = self.artifacts_dir / f"submission_{self.artifact_count}_{timestamp}.csv"
            submission_file.write_text(execution_result["submission_content"])
            
            # Also save to root level for convenience
            root_submission = self.work_dir / "submission.csv"
            root_submission.write_text(execution_result["submission_content"])
        
        # Save execution metadata
        metadata = {
            "artifact_count": self.artifact_count,
            "timestamp": timestamp,
            "success": execution_result.get("success", False),
            "return_code": execution_result.get("return_code", -1),
            "execution_time": execution_result.get("execution_time", 0),
            "cv_score": execution_result.get("cv_score"),
            "evaluator_type": execution_result.get("evaluator_type", "unknown"),
            "stdout_length": len(execution_result.get("stdout", "")),
            "stderr_length": len(execution_result.get("stderr", "")),
            "has_submission": bool(execution_result.get("submission_content"))
        }
        
        metadata_file = self.artifacts_dir / f"metadata_{self.artifact_count}_{timestamp}.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))
        
        if verbose:
            status = "SUCCESS" if execution_result.get("success") else "FAILED"
            print(f"[Artifacts] Saved {status} execution artifacts with timestamp {timestamp}")
        
        return timestamp
    
    def save_claude_artifacts(self, iteration: int, prompt: str, response: str, 
                            success: bool, metadata: Dict[str, Any]) -> None:
        """Save Claude interaction artifacts."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save prompt
        prompt_file = self.artifacts_dir / f"prompt_{iteration}_{timestamp}.txt"
        prompt_file.write_text(prompt)
        
        # Save response 
        response_file = self.artifacts_dir / f"response_{iteration}_{timestamp}.txt"
        response_file.write_text(response)
        
        # Save metadata
        metadata_file = self.artifacts_dir / f"claude_metadata_{iteration}_{timestamp}.json"
        metadata_with_context = {
            **metadata,
            "iteration": iteration,
            "success": success,
            "response_length": len(response),
            "prompt_length": len(prompt),
            "timestamp": timestamp
        }
        
        metadata_file.write_text(json.dumps(metadata_with_context, indent=2))
    
    def journal_claude_interaction(self, prompt: str, response: str, 
                                 metadata: Dict[str, Any]) -> Path:
        """Create journal entry for Claude interaction."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        journal_file = self.journals_dir / f"claude_session_{timestamp}.md"
        
        # Extract code block for metadata
        code_extracted = "```python" in response or "```" in response
        
        journal_content = f"""# Claude Interaction - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Metadata
- **Call Number**: {metadata.get('call_number', 'N/A')}
- **Model**: {metadata.get('model', 'unknown')}
- **Latency**: {metadata.get('latency', 0):.2f}s
- **Prompt Length**: {metadata.get('prompt_length', len(prompt))} characters
- **Response Length**: {metadata.get('response_length', len(response))} characters
- **Success**: True
- **Code Extracted**: {'Yes' if code_extracted else 'No'}

## Prompt Sent to Claude
```
{prompt}
```

## Claude's Response
```
{response}
```

## Execution Results
- **Evaluator**: Docker
- **Timestamp**: {datetime.now().isoformat()}
"""
        
        journal_file.write_text(journal_content)
        return journal_file
    
    def update_journal_with_execution(self, journal_file: Path, 
                                    execution_result: Dict[str, Any]) -> None:
        """Update journal file with execution results."""
        success = execution_result.get('success', False)
        cv_score = execution_result.get('cv_score', 'N/A')
        execution_time = execution_result.get('execution_time', 0)
        return_code = execution_result.get('return_code', -1)
        
        status_text = "EXECUTION SUCCESS" if success else "EXECUTION FAILED"
        
        execution_update = f"""
- **Execution Status**: {status_text}
- **Return Code**: {return_code}
- **Execution Time**: {execution_time:.2f}s
- **CV Score**: {cv_score}
- **Stdout Length**: {len(execution_result.get('stdout', ''))} chars
- **Stderr Length**: {len(execution_result.get('stderr', ''))} chars
"""
        
        # Read current content and append execution details
        current_content = journal_file.read_text()
        updated_content = current_content + execution_update
        journal_file.write_text(updated_content)
    
    def get_artifact_summary(self) -> Dict[str, Any]:
        """Get summary of saved artifacts."""
        return {
            "total_artifacts": self.artifact_count,
            "artifacts_dir": str(self.artifacts_dir),
            "journals_dir": str(self.journals_dir),
            "solution_files": len(list(self.artifacts_dir.glob("solution_*.py"))),
            "stdout_files": len(list(self.artifacts_dir.glob("stdout_*.log"))),
            "stderr_files": len(list(self.artifacts_dir.glob("stderr_*.log"))),
            "submission_files": len(list(self.artifacts_dir.glob("submission_*.csv"))),
            "journal_files": len(list(self.journals_dir.glob("claude_session_*.md")))
        }