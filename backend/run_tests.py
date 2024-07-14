import os
import sys
import unittest



def run_tests():
    # Add the 'app' subdir to the sys.path
    app_path = os.path.join(os.path.dirname(__file__), 'app')
    sys.path.insert(0, app_path)
    
    # Discover and execute all tests
    loader = unittest.TestLoader()
    tests = loader.discover(start_dir=os.path.join(app_path, 'tests'))
    test_runner = unittest.TextTestRunner()
    result = test_runner.run(tests)
    
    # Return exit code
    if result.wasSuccessful():
        return 0
    else:
        return 1



if __name__ == "__main__":
    exit_code = run_tests()
    print(f"Command exited with code: {exit_code}")
    

