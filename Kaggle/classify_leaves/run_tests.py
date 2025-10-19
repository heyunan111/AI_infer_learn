#!/usr/bin/env python3
"""
Test runner script for the ResNet-50 image classification system.

This script runs all unit tests, integration tests, and functionality preservation tests.
"""

import unittest
import sys
import os

def run_all_tests():
    """Run all tests in the tests directory."""
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1

def run_specific_test_module(module_name):
    """Run tests from a specific module."""
    try:
        suite = unittest.TestLoader().loadTestsFromName(f'tests.{module_name}')
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return 0 if result.wasSuccessful() else 1
    except Exception as e:
        print(f"Error loading test module '{module_name}': {e}")
        return 1

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Run specific test module
        module_name = sys.argv[1]
        exit_code = run_specific_test_module(module_name)
    else:
        # Run all tests
        exit_code = run_all_tests()
    
    sys.exit(exit_code)