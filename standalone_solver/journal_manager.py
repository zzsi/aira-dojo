#!/usr/bin/env python3
"""
Simple, pragmatic journal management for Claude interactions.
"""

import os
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

class JournalManager:
    """Simple journal management utility."""
    
    def __init__(self, work_dir: str = "output/working"):
        """
        Initialize journal manager.
        
        Args:
            work_dir: Working directory containing journals
        """
        self.work_dir = Path(work_dir)
        self.journals_dir = self.work_dir / "journals"
        self.journals_dir.mkdir(exist_ok=True)
    
    def list_journals(self) -> List[Dict[str, Any]]:
        """
        List all journal files with metadata.
        
        Returns:
            List of journal info dictionaries
        """
        journals = []
        
        for journal_file in sorted(self.journals_dir.glob("*.md")):
            # Parse timestamp from filename
            timestamp_match = re.search(r'(\d{8}_\d{6})', journal_file.name)
            timestamp_str = timestamp_match.group(1) if timestamp_match else "unknown"
            
            # Get file stats
            stat = journal_file.stat()
            size_kb = stat.st_size / 1024
            
            # Try to extract metadata from content
            try:
                content = journal_file.read_text()
                lines = content.split('\n')
                
                # Extract basic info
                latency = self._extract_value(content, "Latency")
                claude_success = "Success**: True" in content
                code_extracted = "Code Extracted**: Yes" in content
                
                # Check for execution success (more important than Claude response success)
                execution_success = "EXECUTION SUCCESS" in content or "SUCCESS (" in content
                execution_failed = "EXECUTION FAILED" in content or "FAILED (" in content
                
                # Overall success means both Claude responded AND code executed successfully
                success = claude_success and execution_success and not execution_failed
                
                journals.append({
                    "file": journal_file.name,
                    "path": str(journal_file),
                    "timestamp": timestamp_str,
                    "size_kb": round(size_kb, 1),
                    "latency": latency,
                    "success": success,
                    "code_extracted": code_extracted,
                    "lines": len(lines)
                })
                
            except Exception as e:
                # Fallback for corrupted files
                journals.append({
                    "file": journal_file.name,
                    "path": str(journal_file),
                    "timestamp": timestamp_str,
                    "size_kb": round(size_kb, 1),
                    "error": str(e)
                })
        
        return journals
    
    def _extract_value(self, content: str, key: str) -> str:
        """Extract a value from journal content."""
        pattern = rf"- \*\*{key}\*\*:\s*(.+)"
        match = re.search(pattern, content)
        return match.group(1) if match else "N/A"
    
    def search_journals(self, query: str, case_sensitive: bool = False) -> List[str]:
        """
        Search for text across all journals.
        
        Args:
            query: Search query
            case_sensitive: Whether search is case sensitive
            
        Returns:
            List of matching journal file names
        """
        matches = []
        
        if not case_sensitive:
            query = query.lower()
        
        for journal_file in self.journals_dir.glob("*.md"):
            try:
                content = journal_file.read_text()
                if not case_sensitive:
                    content = content.lower()
                
                if query in content:
                    matches.append(journal_file.name)
                    
            except Exception:
                # Skip corrupted files
                continue
        
        return sorted(matches)
    
    def get_latest_journal(self) -> str:
        """Get the most recent journal file."""
        journals = list(self.journals_dir.glob("*.md"))
        if not journals:
            return "No journals found"
        
        latest = max(journals, key=lambda f: f.stat().st_mtime)
        return str(latest)
    
    def clean_old_journals(self, keep_recent: int = 10) -> int:
        """
        Clean old journal files, keeping only the most recent ones.
        
        Args:
            keep_recent: Number of recent journals to keep
            
        Returns:
            Number of files deleted
        """
        journals = sorted(self.journals_dir.glob("*.md"), 
                         key=lambda f: f.stat().st_mtime, reverse=True)
        
        deleted_count = 0
        for old_journal in journals[keep_recent:]:
            try:
                old_journal.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"Warning: Could not delete {old_journal}: {e}")
        
        return deleted_count
    
    def print_summary(self):
        """Print a simple summary of journals."""
        journals = self.list_journals()
        
        if not journals:
            print("📝 No journals found")
            return
        
        print(f"📝 Journal Summary ({len(journals)} files)")
        print("-" * 60)
        print(f"{'File':<25} {'Size':<8} {'Success':<8} {'Code':<6} {'Latency'}")
        print("-" * 60)
        
        for journal in journals[-10:]:  # Show last 10
            name = journal['file'][:24]  # Truncate long names
            size = f"{journal['size_kb']}KB"
            success = "✓" if journal.get('success') else "✗"
            code = "✓" if journal.get('code_extracted') else "✗"
            latency = journal.get('latency', 'N/A')
            
            print(f"{name:<25} {size:<8} {success:<8} {code:<6} {latency}")
        
        if len(journals) > 10:
            print(f"... and {len(journals) - 10} more files")
    
    def view_journal(self, filename: str = None):
        """
        View a journal file (latest if no filename provided).
        
        Args:
            filename: Specific journal filename, or None for latest
        """
        if filename is None:
            journal_path = Path(self.get_latest_journal())
        else:
            journal_path = self.journals_dir / filename
        
        if not journal_path.exists():
            print(f"Journal not found: {journal_path}")
            return
        
        try:
            content = journal_path.read_text()
            print(f"📖 Viewing: {journal_path.name}")
            print("=" * 60)
            print(content)
        except Exception as e:
            print(f"Error reading journal: {e}")

def main():
    """Simple command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Journal Manager")
    parser.add_argument("--work-dir", default="output/working",
                       help="Working directory containing journals")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # List command
    subparsers.add_parser("list", help="List all journals")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search journals")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--case-sensitive", action="store_true",
                              help="Case sensitive search")
    
    # View command
    view_parser = subparsers.add_parser("view", help="View journal")
    view_parser.add_argument("filename", nargs="?", help="Journal filename (latest if not specified)")
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean old journals")
    clean_parser.add_argument("--keep", type=int, default=10,
                             help="Number of recent journals to keep")
    
    args = parser.parse_args()
    
    manager = JournalManager(args.work_dir)
    
    if args.command == "list" or args.command is None:
        manager.print_summary()
    
    elif args.command == "search":
        matches = manager.search_journals(args.query, args.case_sensitive)
        if matches:
            print(f"Found {len(matches)} matching journals:")
            for match in matches:
                print(f"  - {match}")
        else:
            print("No matching journals found")
    
    elif args.command == "view":
        manager.view_journal(args.filename)
    
    elif args.command == "clean":
        deleted = manager.clean_old_journals(args.keep)
        print(f"Deleted {deleted} old journal files")

if __name__ == "__main__":
    main()