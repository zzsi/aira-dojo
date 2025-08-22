#!/usr/bin/env python3
"""
Test script to verify Docker integration with the solver.
"""

import tempfile
from pathlib import Path


def test_solver_with_docker_flag():
    """Test that solver accepts Docker flag correctly."""
    from intelligent_outer_loop import IntelligentSolver
    
    print("🧪 Testing solver with Docker flag...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = temp_dir
        
        # Test with Docker disabled (default)
        solver_venv = IntelligentSolver(work_dir=work_dir, use_docker=False, verbose=True)
        print(f"✅ Virtual env solver created: {type(solver_venv.evaluator).__name__}")
        
        # Test with Docker enabled
        solver_docker = IntelligentSolver(work_dir=work_dir, use_docker=True, verbose=True)
        print(f"✅ Docker solver created: {type(solver_docker.evaluator).__name__}")
        
        # Both should have the same interface
        assert hasattr(solver_venv.evaluator, 'execute_code_with_journaling')
        assert hasattr(solver_docker.evaluator, 'execute_code_with_journaling')
        print("✅ Both evaluators have required interface")


def test_mlebench_with_docker_flag():
    """Test MLEBench solver with Docker flag."""
    try:
        from mlebench_solver import MLEBenchSolver
        
        print("🧪 Testing MLEBench solver with Docker flag...")
        
        # This will fail because we don't have real MLEBench data, but it should accept the parameter
        try:
            solver = MLEBenchSolver(
                task_name="test-task",
                mlebench_data_dir="/nonexistent",
                use_docker=True,
                verbose=False
            )
        except FileNotFoundError:
            print("✅ MLEBench solver accepts Docker parameter (expected FileNotFoundError)")
        
    except ImportError as e:
        print(f"⚠️  MLEBench solver test skipped: {e}")


def test_command_line_integration():
    """Test command line argument parsing."""
    import sys
    import subprocess
    
    print("🧪 Testing command line Docker flag...")
    
    # Test help shows Docker option
    result = subprocess.run([
        sys.executable, "intelligent_outer_loop.py", "--help"
    ], capture_output=True, text=True)
    
    if "--docker" in result.stdout:
        print("✅ Docker flag available in command line help")
    else:
        print("❌ Docker flag missing from command line help")
    
    # Test MLEBench solver help
    result = subprocess.run([
        sys.executable, "mlebench_solver.py", "--help"
    ], capture_output=True, text=True)
    
    if "--docker" in result.stdout:
        print("✅ Docker flag available in MLEBench solver help")
    else:
        print("❌ Docker flag missing from MLEBench solver help")


def main():
    """Run all tests."""
    print("🚀 Testing Docker integration...\n")
    
    test_solver_with_docker_flag()
    print()
    
    test_mlebench_with_docker_flag()
    print()
    
    test_command_line_integration()
    print()
    
    print("🎉 All Docker integration tests completed!")


if __name__ == "__main__":
    main()