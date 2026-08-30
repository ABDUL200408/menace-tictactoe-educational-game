"""Application entry point for the MENACE educational Streamlit interface."""

from src.config import DEFAULT_CONFIG
from ui.app_config import AppConfig
from ui.controller import MENACEStreamlitApp
from ui.session import SessionKey

__all__ = ["AppConfig", "MENACEStreamlitApp", "SessionKey", "main"]


def main() -> None:
    """Create the configured MENACE Streamlit application and start it."""
    app = MENACEStreamlitApp(AppConfig(DEFAULT_CONFIG))
    app.run()


if __name__ == "__main__":
    main()
