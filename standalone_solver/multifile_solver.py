#!/usr/bin/env python3
"""
Multi-file tree search solver that uses git branches for exploration
and allows Claude Code to edit files in the working directory.
"""

import os
import sys
import git
import json
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass

from git_versioned_solver import GitVersionedSolver
from file_state_manager import FileStateManager
from multifile_prompts import MultifilePromptBuilder
from claude_interface import ClaudeInterface
from improved_evaluator import ImprovedCodeEvaluator


@dataclass
class TreeNode:
    """Represents a node in the search tree."""
    branch_name: str
    parent_branch: Optional[str]
    approach: str
    score: Optional[float]
    iteration: int
    success: bool
    timestamp: float
    files_modified: Set[str]


class MultifileSolver(GitVersionedSolver):
    """Solver that uses tree search with git branches and edits working directory files."""
    
    def __init__(self, work_dir: str = "output/working", max_iterations: int = 10,
                 max_branches: int = 5, verbose: bool = True, auto_commit: bool = True, use_docker: bool = True):
        """
        Initialize multifile tree search solver.
        
        Args:
            work_dir: Working directory for solutions
            max_iterations: Maximum iterations total
            max_branches: Maximum number of parallel branches to explore
            verbose: Whether to print detailed progress
            auto_commit: Whether to automatically commit changes
        """
        super().__init__(work_dir, max_iterations, verbose, auto_commit, use_docker)
        
        self.max_branches = max_branches
        self.file_manager = FileStateManager(self.work_dir, verbose)
        self.prompt_builder = MultifilePromptBuilder(self.file_manager, verbose)
        
        # Tree search state
        self.tree_nodes: Dict[str, TreeNode] = {}
        self.active_branches: List[str] = []
        self.completed_branches: List[str] = []
        self.current_depth = 0
        
        # Initialize main branch as root
        self.main_branch = "main"
        self._ensure_main_branch()
    
    def _ensure_main_branch(self):
        """Ensure main branch exists and is set up."""
        try:
            if self.repo:
                # Create main branch if it doesn't exist
                if self.main_branch not in [b.name for b in self.repo.branches]:
                    main_branch = self.repo.create_head(self.main_branch)
                    main_branch.checkout()
                else:
                    self.repo.heads[self.main_branch].checkout()
                
                # Add root node
                self.tree_nodes[self.main_branch] = TreeNode(
                    branch_name=self.main_branch,
                    parent_branch=None,
                    approach="initial_setup",
                    score=None,
                    iteration=0,
                    success=True,
                    timestamp=time.time(),
                    files_modified=set()
                )
                self.active_branches = [self.main_branch]
                
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Failed to ensure main branch: {e}")
    
    def create_exploration_branch(self, parent_branch: str, approach: str, iteration: int) -> str:
        """
        Create a new branch for exploring an approach.
        
        Args:
            parent_branch: Name of parent branch
            approach: Description of the approach to try
            iteration: Iteration number
            
        Returns:
            Name of created branch
        """
        # Clean approach name for branch naming
        clean_approach = re.sub(r'[^a-zA-Z0-9-]', '-', approach.lower())[:20]
        clean_approach = re.sub(r'-+', '-', clean_approach).strip('-')
        
        branch_name = f"iter-{iteration}-{clean_approach}"
        
        # Make branch name unique
        counter = 1
        original_name = branch_name
        while branch_name in self.tree_nodes:
            branch_name = f"{original_name}-{counter}"
            counter += 1
        
        try:
            if self.repo:
                # Checkout parent branch
                self.repo.heads[parent_branch].checkout()
                
                # Create new branch
                new_branch = self.repo.create_head(branch_name)
                new_branch.checkout()
                
                if self.verbose:
                    print(f"🌿 Created exploration branch: {branch_name}")
                
                return branch_name
                
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to create branch {branch_name}: {e}")
            return parent_branch
        
        return branch_name
    
    def switch_to_branch(self, branch_name: str) -> bool:
        """
        Switch to a specific branch.
        
        Args:
            branch_name: Name of branch to switch to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.repo and branch_name in [b.name for b in self.repo.branches]:
                self.repo.heads[branch_name].checkout()
                if self.verbose:
                    print(f"🔄 Switched to branch: {branch_name}")
                return True
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to switch to branch {branch_name}: {e}")
        
        return False
    
    def parse_file_operations(self, response: str) -> List[Tuple[str, str, str]]:
        """
        Parse Claude's response to extract file operations.
        
        Args:
            response: Claude's response containing file operations
            
        Returns:
            List of (operation, filename, content) tuples
        """
        operations = []
        
        # Pattern to match file operations
        patterns = [
            (r'## (Edit|Create|Fix): ([^\n]+)\n```(\w*)\n(.*?)\n```', lambda m: (m.group(1).lower(), m.group(2).strip(), m.group(4))),
            (r'## File: ([^\n]+)\n```(\w*)\n(.*?)\n```', lambda m: ('create', m.group(1).strip(), m.group(3))),
        ]
        
        for pattern, extractor in patterns:
            matches = re.finditer(pattern, response, re.DOTALL | re.MULTILINE)
            for match in matches:
                try:
                    op, filename, content = extractor(match)
                    operations.append((op, filename, content))
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Failed to parse operation: {e}")
        
        return operations
    
    def apply_file_operations(self, operations: List[Tuple[str, str, str]]) -> Set[str]:
        """
        Apply file operations to the working directory.
        
        Args:
            operations: List of (operation, filename, content) tuples
            
        Returns:
            Set of modified file paths
        """
        modified_files = set()
        
        for operation, filename, content in operations:
            try:
                file_path = self.work_dir / filename
                
                # Ensure parent directory exists
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                if operation in ['create', 'edit', 'fix']:
                    file_path.write_text(content, encoding='utf-8')
                    modified_files.add(filename)
                    
                    if self.verbose:
                        print(f"📝 {operation.title()}: {filename} ({len(content)} chars)")
                
            except Exception as e:
                if self.verbose:
                    print(f"❌ Failed to {operation} {filename}: {e}")
        
        return modified_files
    
    def evaluate_current_solution(self) -> Dict[str, Any]:
        """
        Evaluate the current solution in the working directory.
        
        Returns:
            Evaluation results dictionary
        """
        # Find main solution file
        solution_files = ['solution.py', 'main.py', 'run.py']
        solution_file = None
        
        for filename in solution_files:
            if (self.work_dir / filename).exists():
                solution_file = filename
                break
        
        if not solution_file:
            return {
                "success": False,
                "error": "No solution file found (looking for: solution.py, main.py, run.py)",
                "cv_score": None,
                "execution_time": 0
            }
        
        try:
            # Read solution code
            code = (self.work_dir / solution_file).read_text()
            
            # Execute using evaluator
            result = self.evaluator.execute_code_with_journaling(
                code=code,
                claude_prompt="Multifile solution evaluation",
                claude_response=f"Evaluating {solution_file}",
                claude_metadata={"solver": "multifile", "file": solution_file},
                verbose=self.verbose
            )
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to evaluate {solution_file}: {e}",
                "cv_score": None,
                "execution_time": 0
            }
    
    def run_multifile_iteration(self, branch_name: str, iteration: int, approach: str) -> TreeNode:
        """
        Run a single iteration on a specific branch.
        
        Args:
            branch_name: Name of branch to work on
            iteration: Iteration number
            approach: Approach description
            
        Returns:
            TreeNode with results
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"MULTIFILE ITERATION {iteration} - Branch: {branch_name}")
            print(f"Approach: {approach}")
            print(f"{'='*60}")
        
        # Switch to branch
        if not self.switch_to_branch(branch_name):
            return TreeNode(
                branch_name=branch_name,
                parent_branch=None,
                approach=approach,
                score=None,
                iteration=iteration,
                success=False,
                timestamp=time.time(),
                files_modified=set()
            )
        
        # Prepare prompt based on iteration type
        if iteration == 1:
            prompt_type = 'initial'
        else:
            # Determine prompt type based on previous results
            parent_node = self.tree_nodes.get(branch_name)
            if parent_node and not parent_node.success:
                prompt_type = 'debug'
            else:
                prompt_type = 'improve'
        
        # Build prompt
        task_description = self._load_task_description()
        data_overview = self._generate_data_overview()
        
        context = {}
        if prompt_type in ['improve', 'debug']:
            context['memory'] = self._build_memory()
            if prompt_type == 'debug':
                context['error_details'] = self._get_last_error()
        
        prompt = self.prompt_builder.build_prompt(
            prompt_type=prompt_type,
            task_description=task_description,
            data_overview=data_overview,
            **context
        )
        
        # Query Claude
        if self.verbose:
            print(f"🤖 Querying Claude ({len(prompt)} chars)...")
        
        response, success, claude_metadata = self.claude.query(prompt, verbose=self.verbose)
        
        if not success:
            return TreeNode(
                branch_name=branch_name,
                parent_branch=self.tree_nodes.get(branch_name, TreeNode("", None, "", None, 0, False, 0, set())).parent_branch,
                approach=approach,
                score=None,
                iteration=iteration,
                success=False,
                timestamp=time.time(),
                files_modified=set()
            )
        
        # Parse and apply file operations
        operations = self.parse_file_operations(response)
        if not operations:
            if self.verbose:
                print("⚠️  No file operations found in response")
        
        modified_files = self.apply_file_operations(operations)
        
        # Commit changes
        if self.auto_commit and modified_files:
            self._commit_changes(iteration, approach, modified_files)
        
        # Evaluate solution
        evaluation_result = self.evaluate_current_solution()
        score = evaluation_result.get('cv_score')
        
        # Create tree node
        node = TreeNode(
            branch_name=branch_name,
            parent_branch=self.tree_nodes.get(branch_name, TreeNode("", None, "", None, 0, False, 0, set())).parent_branch,
            approach=approach,
            score=score,
            iteration=iteration,
            success=evaluation_result.get('success', False),
            timestamp=time.time(),
            files_modified=modified_files
        )
        
        # Update tree
        self.tree_nodes[branch_name] = node
        
        if self.verbose:
            status = "✅" if node.success else "❌"
            score_text = f"CV={score:.4f}" if score else "No score"
            print(f"{status} Branch {branch_name}: {score_text}")
        
        return node
    
    def select_next_explorations(self) -> List[Tuple[str, str]]:
        """
        Select next branches and approaches to explore.
        
        Returns:
            List of (branch_name, approach) tuples
        """
        explorations = []
        
        # Get promising branches (successful or high-scoring)
        promising_branches = []
        for branch_name, node in self.tree_nodes.items():
            if node.success or (node.score and node.score > 0.5):
                promising_branches.append((branch_name, node))
        
        # Sort by score (descending)
        promising_branches.sort(key=lambda x: x[1].score or 0, reverse=True)
        
        # Select top branches to explore from
        for branch_name, node in promising_branches[:self.max_branches]:
            if len(explorations) >= self.max_branches:
                break
            
            # Generate exploration approaches
            approaches = self._generate_exploration_approaches(node)
            for approach in approaches[:2]:  # Max 2 approaches per branch
                if len(explorations) >= self.max_branches:
                    break
                explorations.append((branch_name, approach))
        
        # If no promising branches, explore from main
        if not explorations:
            approaches = self._generate_exploration_approaches(self.tree_nodes[self.main_branch])
            for approach in approaches[:self.max_branches]:
                explorations.append((self.main_branch, approach))
        
        return explorations
    
    def _generate_exploration_approaches(self, node: TreeNode) -> List[str]:
        """Generate possible exploration approaches from a node."""
        approaches = []
        
        if node.iteration == 0:
            # Initial setup approaches
            approaches = [
                "basic-sklearn-pipeline",
                "feature-engineering-focus",
                "ensemble-methods"
            ]
        else:
            # Improvement approaches
            base_approaches = [
                "hyperparameter-tuning",
                "feature-selection",
                "model-stacking",
                "data-preprocessing",
                "cross-validation-optimization"
            ]
            
            # Filter based on what hasn't been tried
            approaches = base_approaches[:3]  # Limit to top 3
        
        return approaches
    
    def _commit_changes(self, iteration: int, approach: str, modified_files: Set[str]):
        """Commit changes to current branch."""
        try:
            if self.repo and modified_files:
                # Stage modified files
                for filename in modified_files:
                    self.repo.index.add([filename])
                
                # Create commit message
                commit_msg = f"Iteration {iteration}: {approach}\n\nModified files:\n"
                for filename in sorted(modified_files):
                    commit_msg += f"- {filename}\n"
                
                # Commit
                commit = self.repo.index.commit(commit_msg)
                
                if self.verbose:
                    print(f"📝 Committed changes: {commit.hexsha[:8]}")
                
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to commit changes: {e}")
    
    def _load_task_description(self) -> str:
        """Load task description from instructions file."""
        instructions_file = self.work_dir / "instructions.txt"
        if instructions_file.exists():
            return instructions_file.read_text()
        return "Machine learning task (no detailed description available)"
    
    def _generate_data_overview(self) -> str:
        """Generate overview of available data."""
        data_dir = self.work_dir / "data"
        if not data_dir.exists():
            return "No data directory found"
        
        overview_parts = []
        for data_file in data_dir.glob("*.csv"):
            size_kb = data_file.stat().st_size / 1024
            overview_parts.append(f"- {data_file.name}: {size_kb:.1f}KB")
        
        return "\n".join(overview_parts) if overview_parts else "No CSV files found"
    
    def _build_memory(self) -> str:
        """Build memory of previous iterations."""
        memory_parts = []
        
        for branch_name, node in sorted(self.tree_nodes.items(), key=lambda x: x[1].iteration):
            if node.iteration > 0:
                status = "✅" if node.success else "❌"
                score_text = f"CV={node.score:.4f}" if node.score else "No score"
                memory_parts.append(f"{status} {node.approach}: {score_text}")
        
        return "\n".join(memory_parts) if memory_parts else "No previous iterations"
    
    def _get_last_error(self) -> str:
        """Get error details from last failed iteration."""
        # This would ideally get error details from evaluator logs
        return "Check execution logs for error details"
    
    def run(self) -> Dict[str, Any]:
        """Run the multifile tree search solver."""
        
        if self.verbose:
            print("🌳 Starting Multifile Tree Search Solver")
            print(f"Max iterations: {self.max_iterations}, Max branches: {self.max_branches}")
        
        # Setup data
        self.setup_data()
        
        iteration = 1
        
        while iteration <= self.max_iterations:
            try:
                # Select explorations for this iteration
                explorations = self.select_next_explorations()
                
                if not explorations:
                    if self.verbose:
                        print("🛑 No more explorations to try")
                    break
                
                # Run explorations
                for branch_name, approach in explorations:
                    if iteration > self.max_iterations:
                        break
                    
                    # Create new branch for exploration
                    new_branch = self.create_exploration_branch(branch_name, approach, iteration)
                    
                    # Run iteration
                    node = self.run_multifile_iteration(new_branch, iteration, approach)
                    
                    iteration += 1
                
            except KeyboardInterrupt:
                if self.verbose:
                    print("\n⚠️  Interrupted by user")
                break
            except Exception as e:
                if self.verbose:
                    print(f"\n❌ Error in iteration {iteration}: {e}")
                iteration += 1
        
        return self.summarize_tree_search()
    
    def summarize_tree_search(self) -> Dict[str, Any]:
        """Summarize the tree search results."""
        # Find best solution
        best_node = None
        best_score = -1
        
        for node in self.tree_nodes.values():
            if node.success and node.score and node.score > best_score:
                best_node = node
                best_score = node.score
        
        # Switch to best branch for final state
        if best_node:
            self.switch_to_branch(best_node.branch_name)
        
        summary = {
            "solver_type": "multifile_tree_search",
            "total_branches": len(self.tree_nodes),
            "successful_branches": len([n for n in self.tree_nodes.values() if n.success]),
            "best_score": best_score if best_score > -1 else None,
            "best_branch": best_node.branch_name if best_node else None,
            "tree_structure": self._build_tree_structure()
        }
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("TREE SEARCH SUMMARY")
            print(f"{'='*60}")
            print(f"Total branches explored: {summary['total_branches']}")
            print(f"Successful branches: {summary['successful_branches']}")
            if best_node:
                print(f"Best solution: {best_node.branch_name} (CV={best_score:.4f})")
            print(f"{'='*60}")
        
        return summary
    
    def _build_tree_structure(self) -> Dict[str, Any]:
        """Build a representation of the tree structure."""
        tree = {}
        for branch_name, node in self.tree_nodes.items():
            tree[branch_name] = {
                "parent": node.parent_branch,
                "approach": node.approach,
                "score": node.score,
                "success": node.success,
                "iteration": node.iteration
            }
        return tree


def main():
    """Main entry point for multifile solver."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multifile Tree Search Solver")
    parser.add_argument("--work-dir", default="output/working", help="Working directory")
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum iterations")
    parser.add_argument("--max-branches", type=int, default=5, help="Maximum parallel branches")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    parser.add_argument("--no-auto-commit", action="store_true", help="Disable automatic commits")
    
    args = parser.parse_args()
    
    try:
        solver = MultifileSolver(
            work_dir=args.work_dir,
            max_iterations=args.max_iterations,
            max_branches=args.max_branches,
            verbose=not args.quiet,
            auto_commit=not args.no_auto_commit
        )
        
        summary = solver.run()
        
        # Exit codes based on performance
        best_score = summary.get("best_score")
        if best_score and best_score > 0.8:
            sys.exit(0)
        elif best_score:
            sys.exit(1)
        else:
            sys.exit(2)
            
    except Exception as e:
        print(f"❌ Multifile solver failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()