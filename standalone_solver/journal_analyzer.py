#!/usr/bin/env python3
"""
Journal analyzer that extracts learning patterns from execution history.
"""

from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
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
        """Extract patterns from successful interactions and learnings.md."""
        patterns = []
        
        # Extract from journal files
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
        
        # Extract from learnings.md for task-specific insights
        try:
            learnings_file = self.journal_manager.work_dir / "learnings.md"
            if learnings_file.exists():
                learnings_content = learnings_file.read_text()
                
                # Parse "What Works" section
                works_section = self._extract_section_content(learnings_content, "## What Works ✅")
                if works_section:
                    for line in works_section.split('\n'):
                        if line.strip().startswith('- '):
                            insight = line.strip()[2:].strip()
                            if insight and len(insight) > 10:  # Skip trivial entries
                                patterns.append(insight)
                
        except Exception:
            pass
        
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
        """Extract patterns from failed approaches and learnings.md."""
        patterns = []
        
        # Extract from journal files
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
        
        # Extract from learnings.md for task-specific failures
        try:
            learnings_file = self.journal_manager.work_dir / "learnings.md"
            if learnings_file.exists():
                learnings_content = learnings_file.read_text()
                
                # Parse "What Doesn't Work" section
                fails_section = self._extract_section_content(learnings_content, "## What Doesn't Work ❌")
                if fails_section:
                    for line in fails_section.split('\n'):
                        if line.strip().startswith('- '):
                            insight = line.strip()[2:].strip()
                            if insight and len(insight) > 10:  # Skip trivial entries
                                patterns.append(f"AVOID: {insight}")
                
        except Exception:
            pass
        
        return list(set(patterns))
    
    def _extract_section_content(self, content: str, section_header: str) -> str:
        """Extract content of a specific markdown section."""
        start = content.find(section_header)
        if start == -1:
            return ""
        
        # Find start of content (after section header line)
        content_start = content.find('\n', start) + 1
        
        # Find next section or end of content
        next_section = content.find('\n## ', content_start)
        if next_section == -1:
            return content[content_start:].strip()
        else:
            return content[content_start:next_section].strip()
    
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
        
        # Add task-specific insights from learnings.md
        try:
            learnings_file = self.journal_manager.work_dir / "learnings.md"
            if learnings_file.exists():
                learnings_content = learnings_file.read_text()
                
                # Extract insights from "Key Insights" section
                insights_section = self._extract_section_content(learnings_content, "## Key Insights 💡")
                if insights_section:
                    insights_count = 0
                    for line in insights_section.split('\n'):
                        if line.strip().startswith('- ') and insights_count < 3:
                            insight = line.strip()[2:].strip()
                            if insight and len(insight) > 10:
                                recommendations.append(insight)
                                insights_count += 1
        except Exception:
            pass
        
        return recommendations