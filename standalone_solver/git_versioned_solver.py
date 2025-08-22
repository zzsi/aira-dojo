#!/usr/bin/env python3
"""
Git-versioned solver that tracks all solution attempts with proper version control.
"""

import os
import sys
import git
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from intelligent_outer_loop import IntelligentSolver

class GitVersionedSolver(IntelligentSolver):
    """Intelligent solver with git version control for solutions."""
    
    def __init__(self, work_dir: str = "output/working", max_iterations: int = 5, 
                 verbose: bool = True, auto_commit: bool = True, use_docker: bool = True):
        """
        Initialize git-versioned solver.
        
        Args:
            work_dir: Working directory for the solver
            max_iterations: Maximum number of iterations
            verbose: Whether to print detailed progress
            auto_commit: Whether to automatically commit after each iteration
            use_docker: Whether to use Docker for dependency management
        """
        super().__init__(work_dir, max_iterations, verbose, use_docker)
        self.auto_commit = auto_commit
        self.repo: Optional[git.Repo] = None
        
        # Initialize git repo
        self._setup_git_repo()
    
    def _setup_git_repo(self):
        """Initialize git repository in working directory."""
        try:
            # Try to open existing repo
            self.repo = git.Repo(self.work_dir)
            if self.verbose:
                print(f"📁 Using existing git repo in {self.work_dir}")
                
        except git.InvalidGitRepositoryError:
            # Initialize new repo
            if self.verbose:
                print(f"🔧 Initializing git repo in {self.work_dir}")
            
            # Ensure work_dir exists
            self.work_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize repo
            self.repo = git.Repo.init(self.work_dir)
            
            # Create .gitignore for appropriate files
            self._create_gitignore()
            
            # Initial commit with data setup
            self._initial_commit()
    
    def _create_gitignore(self):
        """Create appropriate .gitignore for ML solutions."""
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.pyc

# Jupyter Notebooks
.ipynb_checkpoints

# Large data files (keep small sample data)
*.csv.gz
*.parquet
*.h5
*.pkl

# Model files
*.joblib
*.model
*.weights

# Temporary files
tmp/
temp/
*.tmp

# OS files
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/

# Logs (keep execution logs but ignore temporary ones)
*.log
!execution_*.log
"""
        
        gitignore_path = self.work_dir / ".gitignore"
        gitignore_path.write_text(gitignore_content)
        
        if self.verbose:
            print("📝 Created .gitignore for ML project")
    
    def _initial_commit(self):
        """Create initial commit with data and setup."""
        try:
            # Add basic files
            self.repo.index.add([".gitignore"])
            
            # Add data files if they exist
            data_dir = self.work_dir / "data"
            if data_dir.exists():
                for data_file in data_dir.glob("*"):
                    if data_file.is_file() and data_file.stat().st_size < 1024 * 1024:  # < 1MB
                        self.repo.index.add([str(data_file.relative_to(self.work_dir))])
            
            # Add instructions if they exist
            instructions_file = self.work_dir / "instructions.txt"
            if instructions_file.exists():
                self.repo.index.add([str(instructions_file.relative_to(self.work_dir))])
            
            # Initial commit
            self.repo.index.commit(
                "Initial commit: ML solver workspace setup\\n\\n"
                f"- Initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n"
                "- Added data files and configuration\\n"
                "- Set up .gitignore for ML projects"
            )
            
            if self.verbose:
                print("✅ Initial commit created")
                
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Initial commit failed: {e}")
    
    def commit_iteration_results(self, iteration: int, solution: Dict[str, Any]):
        """
        Commit the results of an iteration.
        
        Args:
            iteration: Iteration number
            solution: Solution dictionary with results
        """
        if not self.repo:
            return
        
        try:
            # Stage files to commit
            files_to_add = []
            
            # Add artifacts from this iteration
            artifacts_dir = self.work_dir / "artifacts"
            if artifacts_dir.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Look for files from this iteration
                for artifact_file in artifacts_dir.glob(f"*{iteration}_*"):
                    if artifact_file.is_file():
                        rel_path = str(artifact_file.relative_to(self.work_dir))
                        files_to_add.append(rel_path)
                
                # Also add timestamped files from around this time (last 5 minutes)
                for artifact_file in artifacts_dir.glob(f"*{timestamp[:11]}*"):  # Same hour/minute
                    if artifact_file.is_file():
                        rel_path = str(artifact_file.relative_to(self.work_dir))
                        if rel_path not in files_to_add:
                            files_to_add.append(rel_path)
            
            # Add latest submission if it exists
            submission_file = self.work_dir / "submission.csv"
            if submission_file.exists():
                files_to_add.append("submission.csv")
            
            # Add latest journal entries
            journals_dir = self.work_dir / "journals"
            if journals_dir.exists():
                # Add recent journal files (from this session)
                session_start = datetime.now().strftime("%Y%m%d")
                for journal_file in journals_dir.glob(f"*{session_start}*.md"):
                    if journal_file.is_file():
                        rel_path = str(journal_file.relative_to(self.work_dir))
                        if rel_path not in files_to_add:
                            files_to_add.append(rel_path)
            
            if files_to_add:
                # Stage files
                self.repo.index.add(files_to_add)
                
                # Create commit message
                success = solution.get('success', False)
                cv_score = solution.get('execution_result', {}).get('cv_score')
                exec_time = solution.get('execution_result', {}).get('execution_time', 0)
                
                status_emoji = "✅" if success else "❌"
                score_text = f"CV={cv_score:.4f}" if cv_score else "No score"
                
                commit_message = f"{status_emoji} Iteration {iteration}: {score_text} ({exec_time:.1f}s)\\n\\n"
                
                # Add solution details
                if solution.get('explanation'):
                    explanation = solution['explanation'][:200]  # Truncate long explanations
                    commit_message += f"Approach: {explanation}...\\n\\n"
                
                if success and cv_score:
                    commit_message += f"✅ Successful solution with CV score: {cv_score:.4f}\\n"
                elif success:
                    commit_message += "✅ Code executed successfully\\n"
                else:
                    error = solution.get('error', 'Unknown error')
                    commit_message += f"❌ Failed: {error}\\n"
                
                # Add file information
                commit_message += f"\\nFiles added:\\n"
                for file_path in files_to_add:
                    commit_message += f"- {file_path}\\n"
                
                commit_message += f"\\nGenerated by: Intelligent ML Solver"
                
                # Commit
                commit = self.repo.index.commit(commit_message)
                
                if self.verbose:
                    print(f"📝 Git commit: {commit.hexsha[:8]} - Iteration {iteration}")
                    print(f"   Files: {', '.join(files_to_add)}")
                
            else:
                if self.verbose:
                    print(f"⚠️  No files to commit for iteration {iteration}")
                    
        except Exception as e:
            if self.verbose:
                print(f"❌ Git commit failed for iteration {iteration}: {e}")
    
    def create_solution_branch(self, iteration: int, description: str = None) -> str:
        """
        Create a new branch for experimenting with a solution approach.
        
        Args:
            iteration: Iteration number
            description: Optional description of the approach
            
        Returns:
            Branch name created
        """
        if not self.repo:
            return "main"
        
        try:
            # Generate branch name
            desc_part = description.lower().replace(' ', '-')[:20] if description else "approach"
            branch_name = f"iteration-{iteration}-{desc_part}"
            
            # Create and checkout branch
            branch = self.repo.create_head(branch_name)
            branch.checkout()
            
            if self.verbose:
                print(f"🌿 Created branch: {branch_name}")
            
            return branch_name
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Branch creation failed: {e}")
            return "main"
    
    def tag_best_solution(self, best_iteration: int, cv_score: float):
        """
        Tag the best solution for easy reference.
        
        Args:
            best_iteration: Iteration number of best solution
            cv_score: CV score of best solution
        """
        if not self.repo:
            return
        
        try:
            tag_name = f"best-solution-cv-{cv_score:.4f}".replace(".", "p")
            tag_message = f"Best solution: Iteration {best_iteration} with CV score {cv_score:.4f}"
            
            # Create annotated tag
            self.repo.create_tag(tag_name, message=tag_message)
            
            if self.verbose:
                print(f"🏷️  Tagged best solution: {tag_name}")
                
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Tagging failed: {e}")
    
    def run_intelligent_iteration(self, iteration: int) -> Dict[str, Any]:
        """Run iteration with git versioning."""
        
        # Run the parent's intelligent iteration
        solution = super().run_intelligent_iteration(iteration)
        
        # Commit results if auto-commit is enabled
        if self.auto_commit and solution.get('success'):
            self.commit_iteration_results(iteration, solution)
        
        return solution
    
    def summarize_intelligent_session(self) -> Dict[str, Any]:
        """Summarize session with git information."""
        
        summary = super().summarize_intelligent_session()
        
        # Add git information
        if self.repo:
            try:
                summary["git_info"] = {
                    "repository": str(self.work_dir),
                    "current_branch": self.repo.active_branch.name,
                    "total_commits": len(list(self.repo.iter_commits())),
                    "latest_commit": self.repo.head.commit.hexsha[:8] if self.repo.head.commit else None
                }
                
                # Tag best solution if we found one
                if summary.get("best_cv_score") and summary.get("best_iteration"):
                    self.tag_best_solution(summary["best_iteration"], summary["best_cv_score"])
                
                if self.verbose:
                    print(f"📊 Git summary:")
                    print(f"   Branch: {summary['git_info']['current_branch']}")
                    print(f"   Commits: {summary['git_info']['total_commits']}")
                    print(f"   Latest: {summary['git_info']['latest_commit']}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Git summary failed: {e}")
        
        return summary
    
    def show_solution_history(self):
        """Show git history of solutions."""
        if not self.repo:
            print("No git repository available")
            return
        
        try:
            print("\\n📚 Solution History:")
            print("-" * 60)
            
            commits = list(self.repo.iter_commits(max_count=10))  # Last 10 commits
            
            for commit in commits:
                short_sha = commit.hexsha[:8]
                date = commit.committed_datetime.strftime("%Y-%m-%d %H:%M")
                message = commit.message.split('\\n')[0]  # First line only
                
                print(f"{short_sha} | {date} | {message}")
            
            # Show tags
            tags = list(self.repo.tags)
            if tags:
                print("\\n🏷️  Tags:")
                for tag in tags:
                    print(f"   {tag.name}")
                    
        except Exception as e:
            print(f"Error showing history: {e}")

def main():
    """Main entry point for git-versioned solver."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Git-Versioned ML Solver")
    parser.add_argument("--work-dir", default="output/working", help="Working directory")
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum iterations")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    parser.add_argument("--no-auto-commit", action="store_true", help="Disable automatic commits")
    parser.add_argument("--show-history", action="store_true", help="Show solution history and exit")
    
    args = parser.parse_args()
    
    solver = GitVersionedSolver(
        work_dir=args.work_dir,
        max_iterations=args.max_iterations,
        verbose=not args.quiet,
        auto_commit=not args.no_auto_commit
    )
    
    if args.show_history:
        solver.show_solution_history()
        return
    
    try:
        print("🔧 Git-Versioned Intelligent Solver")
        print("=" * 50)
        
        summary = solver.run()
        
        # Show final git status
        if solver.verbose and solver.repo:
            print("\\n📊 Final Repository State:")
            solver.show_solution_history()
        
        # Exit codes
        best_score = summary.get("best_cv_score")
        if best_score and best_score > 0.8:
            sys.exit(0)
        elif best_score:
            sys.exit(1)
        else:
            sys.exit(2)
            
    except Exception as e:
        print(f"❌ Git-versioned solver failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()