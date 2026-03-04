#!/usr/bin/env python3
"""
Root entry point for the Defense Line Analysis Pipeline.

This module serves as a shim that delegates to the preprocessing.main module.

Usage:
    python main.py --method girl
    python main.py --data_match barcelona_madrid --back_four back_four
"""

import sys
from pathlib import Path

# Add preprocessing to path so we can import it
sys.path.insert(0, str(Path(__file__).parent))

# Delegate to preprocessing.main
if __name__ == "__main__":
    from preprocessing.main import main
    main()
