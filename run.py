"""
Main entry point for Spandhan Desktop PySide6 Application.
"""

import sys
from PySide6.QtWidgets import QApplication
from app.main_window import MainWindow
from core.config import APP_NAME, VERSION


def main():
    print("=" * 60)
    print(f"Starting {APP_NAME} v{VERSION} Desktop GUI...")
    print("Multi-Domain Digital Signal Analysis & Diagnostics Platform")
    print("=" * 60)

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()