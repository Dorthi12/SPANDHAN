"""
converters/image_to_mat/run.py
==============================
Quick launcher script to start the Image to .MAT Converter GUI.

Run directly via:
    python converters/image_to_mat/run.py
"""

import sys
from pathlib import Path

# Add project root to sys.path so imports work regardless of working directory
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from converters.image_to_mat.gui import main

if __name__ == "__main__":
    main()
