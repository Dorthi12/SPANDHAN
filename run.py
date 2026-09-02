"""Main entry point for Spandhan application."""

from core.session import SignalSession
from core.config import APP_NAME, VERSION


def main():
    print("=" * 60)
    print(f"{APP_NAME} v{VERSION}")
    print("Multi-Domain Digital Signal Analysis Platform")
    print("=" * 60)

    session = SignalSession()

    print("Session initialized successfully.")
    print("Spandhan V1 foundation is ready.")


if __name__ == "__main__":
    main()