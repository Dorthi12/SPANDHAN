"""
SPANDHAN — Image ML UI Entrypoint (ml/image/ui)
==============================================
Convenience interface exposing the Image ML Analysis Page and runner
directly within the ml/image hierarchy.
"""

import sys
from pathlib import Path

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ui.ml.image.image_ml_page import ImageMLPage
from ui.app import SpandhanMainWindow
from ui.main import main

__all__ = ["ImageMLPage", "SpandhanMainWindow", "main"]

if __name__ == "__main__":
    main()
