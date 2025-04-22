import pytest
import importlib
import os

def test_imports_from_init():
    # Path to the __init__.py file
    init_path = os.path.join(os.path.dirname(__file__), 'preprocessing', '__init__.py')

    # Read __init__.py to get the `__all__` exports
    with open(init_path, 'r') as file:
        content = file.read()

    # Extract `__all__` from the file using simple string matching (you could use regex for more robustness)
    start_idx = content.find('__all__ = [') + len('__all__ = [')
    end_idx = content.find(']', start_idx)
    all_exports_str = content[start_idx:end_idx].strip()
    
    # Convert the string to a list of names in __all__
    exports = [e.strip().strip("'").strip('"') for e in all_exports_str.split(',')]

    # Test each import dynamically
    for export in exports:
        print(f"Testing import: {export}")  # Print what is being tested
        
        # Try to import each symbol from the module
        try:
            module = importlib.import_module('preprocessing')
            item = getattr(module, export)  # Get the item from the module
            
            # Perform the test: check that it is not None
            assert item is not None, f"{export} import failed"
        
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Import of {export} failed: {str(e)}")
