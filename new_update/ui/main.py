"""
SPANDHAN — Desktop Application Launcher
========================================
Main entrypoint for the SPANDHAN Signal Intelligence Workstation.
Launches the PySide6 Qt GUI with the Image ML Analysis page active.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure root repository directory is in sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QApplication

from ui.app import SpandhanMainWindow


def main():
    """Launch SPANDHAN Workstation."""
    app = QApplication(sys.argv)
    app.setApplicationName("SPANDHAN Signal Intelligence")
    app.setOrganizationName("SPANDHAN")

    # Set application font
    app_font = QFont("Segoe UI", 10)
    app.setFont(app_font)

    # Instantiate and show main window
    window = SpandhanMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
