#!/usr/bin/env python3
"""
Test runner script for the movie recommender system.

This script provides an easy way to run different test suites with proper
environment setup and reporting.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} - PASSED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED (exit code: {e.returncode})")
        return False
    except FileNotFoundError:
        print(f"❌ {description} - FAILED (command not found)")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tests for the movie recommender system")
    parser.add_argument("--type", choices=["unit", "integration", "all"], default="all",
                       help="Type of tests to run")
    parser.add_argument("--coverage", action="store_true",
                       help="Run tests with coverage reporting")
    parser.add_argument("--fast", action="store_true",
                       help="Skip slow tests")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose test output")
    
    args = parser.parse_args()
    
    # Add src to Python path
    src_path = Path(__file__).parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    success_count = 0
    total_tests = 0
    
    # Base pytest command
    pytest_cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        pytest_cmd.append("-v")
    
    if args.fast:
        pytest_cmd.extend(["-m", "not slow"])
    
    if args.coverage:
        pytest_cmd.extend([
            "--cov=src/movie_recommender",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Run specific test types
    if args.type == "unit" or args.type == "all":
        total_tests += 1
        cmd = pytest_cmd + ["tests/test_config.py", "tests/test_data_processor.py", "tests/test_vector_store.py"]
        if run_command(cmd, "Unit Tests"):
            success_count += 1
    
    if args.type == "integration" or args.type == "all":
        total_tests += 1
        cmd = pytest_cmd + ["tests/test_integration.py"]
        if run_command(cmd, "Integration Tests"):
            success_count += 1
    
    # Special test for agents (may need mocking)
    if args.type == "all":
        total_tests += 1
        cmd = pytest_cmd + ["tests/test_agents.py"]
        if run_command(cmd, "Agent Tests"):
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 All tests passed!")
        if args.coverage:
            print("📈 Coverage report generated in htmlcov/index.html")
        return 0
    else:
        print(f"💥 {total_tests - success_count} test suite(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())