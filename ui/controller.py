"""Main Streamlit controller for the MENACE learning game."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid

import streamlit as st

from src.visualisation import VisualisationConfig, Visualiser
from ui.app_config import AppConfig
from ui.common import CommonMixin
from ui.demos import DemoMixin
from ui.navigation import NavigationMixin
from ui.pages.comparison import ComparisonPageMixin
from ui.pages.learning import LearningPageMixin
from ui.pages.play import PlayPageMixin
from ui.pages.results import ResultsPageMixin
from ui.pages.training import TrainingPageMixin
from ui.persistence_controls import PersistenceControlsMixin
from ui.session import SessionKey, SessionMixin

_SERVER_RUN_ID = uuid.uuid4().hex
_PARTICIPANT_LOCK = threading.Lock()


class MENACEStreamlitApp(
    SessionMixin,
    NavigationMixin,
    PersistenceControlsMixin,
    DemoMixin,
    CommonMixin,
    PlayPageMixin,
    TrainingPageMixin,
    ComparisonPageMixin,
    ResultsPageMixin,
    LearningPageMixin,
):
    """Coordinate shared state, navigation, and the four interface pages."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.config.validate()
        self._build_visualiser()

    def _build_visualiser(self) -> None:
        """Create the visualiser using the active output directory."""
        visual_defaults = self.config.project_config.visualisation
        self.visualiser = Visualiser(
            VisualisationConfig(
                title_prefix=visual_defaults.title_prefix,
                moving_average_window=visual_defaults.moving_average_window,
                figure_output_dir=self.config.figure_output_dir,
                export_png_enabled=visual_defaults.export_png,
                max_visible_beads_per_move=(
                    visual_defaults.max_visible_beads_per_move
                ),
                square_labels=self.config.square_labels,
            )
        )

    @staticmethod
    def _decode_streamlit_xsrf_cookie(cookie_value: str) -> bytes | None:
        """Return the stable token bytes from Streamlit's XSRF cookie."""
        value = str(cookie_value or "").strip().strip("\"'")
        if not value:
            return None

        try:
            if value.startswith("2|"):
                _, mask_hex, masked_hex, _timestamp = value.split("|", 3)
                mask = bytes.fromhex(mask_hex)
                masked = bytes.fromhex(masked_hex)
                if not mask or not masked:
                    return None
                return bytes(
                    byte ^ mask[index % len(mask)]
                    for index, byte in enumerate(masked)
                )

            token = bytes.fromhex(value)
            return token or None
        except (TypeError, ValueError):
            return None

    def _browser_identity_key(self) -> str | None:
        """Return a privacy-preserving stable key for this browser profile.

        Streamlit's XSRF cookie is different for separate browser profiles but
        remains available when the same browser reconnects. Only a SHA-256 hash
        of the underlying token is used; the cookie itself is never stored.
        """
        try:
            cookie_value = st.context.cookies.get("_streamlit_xsrf")
        except Exception:
            return None

        token = self._decode_streamlit_xsrf_cookie(str(cookie_value or ""))
        if token is None:
            return None
        return hashlib.sha256(token).hexdigest()

    def _participant_registry_path(self):
        """Return the shared browser-to-participant registry path."""
        paths = self.config.project_config.paths
        return paths.results_dir / "sessions" / ".participant_registry.json"

    def _load_participant_registry(self) -> dict[str, dict[str, str]]:
        """Load valid browser-to-participant mappings from disk."""
        path = self._participant_registry_path()
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

        raw = payload.get("participants", {}) if isinstance(payload, dict) else {}
        if not isinstance(raw, dict):
            return {}

        registry: dict[str, dict[str, str]] = {}
        for browser_key, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            session_id = str(entry.get("session_id", "")).strip().lower()
            participant_label = str(
                entry.get("participant_label", "")
            ).strip().lower()
            if self._valid_persistence_identity(session_id, participant_label):
                registry[str(browser_key)] = {
                    "session_id": session_id,
                    "participant_label": participant_label,
                }
        return registry

    def _save_participant_registry(
        self,
        registry: dict[str, dict[str, str]],
    ) -> None:
        """Atomically save browser-to-participant mappings."""
        path = self._participant_registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {"version": 1, "participants": registry},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _valid_persistence_identity(
        session_id: str,
        participant_label: str,
    ) -> bool:
        """Return whether a browser identity has a valid matching label."""
        valid_id = re.fullmatch(r"[0-9a-f]{32}", session_id) is not None
        valid_label = (
            re.fullmatch(
                r"user_[0-9]{2,}_[0-9a-f]{4}",
                participant_label,
            )
            is not None
        )
        return bool(
            valid_id
            and valid_label
            and participant_label.endswith(session_id[:4])
        )

    def _next_participant_number(
        self,
        registry: dict[str, dict[str, str]],
    ) -> int:
        """Return the next readable participant number across restarts."""
        numbers: list[int] = []
        pattern = re.compile(r"user_([0-9]{2,})_[0-9a-f]{4}")

        for entry in registry.values():
            match = pattern.fullmatch(str(entry.get("participant_label", "")))
            if match:
                numbers.append(int(match.group(1)))

        paths = self.config.project_config.paths
        for root in (
            paths.saved_models_dir / "sessions",
            paths.results_dir / "sessions",
        ):
            if not root.exists():
                continue
            for directory in root.iterdir():
                if not directory.is_dir():
                    continue
                match = pattern.fullmatch(directory.name.lower())
                if match:
                    numbers.append(int(match.group(1)))

        return max(numbers, default=0) + 1

    def _new_persistence_identity(
        self,
        registry: dict[str, dict[str, str]],
    ) -> tuple[str, str]:
        """Create a readable participant label backed by a private UUID."""
        session_id = uuid.uuid4().hex
        participant_number = self._next_participant_number(registry)
        label = f"user_{participant_number:02d}_{session_id[:4]}"
        return session_id, label

    @staticmethod
    def _identity_owned_by_other_browser(
        registry: dict[str, dict[str, str]],
        browser_key: str,
        session_id: str,
        participant_label: str,
    ) -> bool:
        """Return whether the supplied identity belongs to another browser."""
        for known_browser, entry in registry.items():
            if known_browser == browser_key:
                continue
            if (
                entry.get("session_id") == session_id
                or entry.get("participant_label") == participant_label
            ):
                return True
        return False

    def _scope_persistence_to_browser(self) -> None:
        """Give each browser profile its own stable persistence identity.

        Separate browsers receive separate ``user_xx_xxxx`` labels. The same
        browser can reconnect after a Streamlit restart because a hashed browser
        cookie key is mapped to its participant identity in a small server-side
        registry. Continue/reset actions therefore remain scoped to that browser's
        own saved learning journey.
        """
        session_id = st.session_state.get(SessionKey.PERSISTENCE_ID)
        participant_label = st.session_state.get(SessionKey.PERSISTENCE_LABEL)

        if not session_id or not participant_label:
            candidate_id = str(
                st.query_params.get("menace_session", "")
            ).strip().lower()
            candidate_label = str(
                st.query_params.get("menace_user", "")
            ).strip().lower()
            browser_key = self._browser_identity_key()

            with _PARTICIPANT_LOCK:
                registry = self._load_participant_registry()
                registered = registry.get(browser_key) if browser_key else None

                if registered is not None:
                    # The browser already owns a participant identity. This also
                    # prevents a shared personalised URL from switching it to
                    # another participant's saved journey.
                    session_id = registered["session_id"]
                    participant_label = registered["participant_label"]
                elif (
                    browser_key
                    and self._valid_persistence_identity(
                        candidate_id,
                        candidate_label,
                    )
                    and not self._identity_owned_by_other_browser(
                        registry,
                        browser_key,
                        candidate_id,
                        candidate_label,
                    )
                ):
                    # Preserve an existing personalised URL for this browser and
                    # register it so future base-URL visits can recover it.
                    session_id, participant_label = candidate_id, candidate_label
                    registry[browser_key] = {
                        "session_id": session_id,
                        "participant_label": participant_label,
                    }
                    self._save_participant_registry(registry)
                elif browser_key:
                    # A genuinely new browser/profile always receives a new
                    # participant, even when other users already have saved data.
                    session_id, participant_label = self._new_persistence_identity(
                        registry
                    )
                    registry[browser_key] = {
                        "session_id": session_id,
                        "participant_label": participant_label,
                    }
                    self._save_participant_registry(registry)
                elif self._valid_persistence_identity(
                    candidate_id,
                    candidate_label,
                ):
                    # Fallback for environments where browser cookies are not
                    # exposed: a valid personalised URL still survives reruns.
                    session_id, participant_label = candidate_id, candidate_label
                else:
                    session_id, participant_label = self._new_persistence_identity(
                        registry
                    )

            st.session_state[SessionKey.PERSISTENCE_ID] = session_id
            st.session_state[SessionKey.PERSISTENCE_LABEL] = participant_label

        st.query_params["menace_session"] = str(session_id)
        st.query_params["menace_user"] = str(participant_label)
        st.query_params["menace_run"] = _SERVER_RUN_ID

        self.config = self.config.with_isolated_paths(str(participant_label))
        self._build_visualiser()

    def run(self) -> None:
        """Run the Streamlit application."""
        self._configure_page()
        self._scope_persistence_to_browser()
        self._prepare_directories()
        self._initialise_session_state()
        self._inject_styles()
        self._render_page_header()
        self._render_sidebar()

        page_labels = list(self.config.simple_page_labels)

        if st.session_state.get(SessionKey.ACTIVE_PAGE) not in page_labels:
            st.session_state[SessionKey.ACTIVE_PAGE] = page_labels[0]

        selected_page = st.session_state[SessionKey.ACTIVE_PAGE]
        self._render_current_page_banner(selected_page)

        # The visual demo belongs only to the Play page.
        if selected_page == page_labels[0]:
            self._render_welcome_guide()
        else:
            st.session_state[SessionKey.SHOW_DEMO] = False
            st.session_state[SessionKey.DEMO_STEP] = 0

        page_renderers = {
            page_labels[0]: self._render_play_page,
            page_labels[1]: self._render_training_page,
            page_labels[2]: self._render_comparison_page,
            page_labels[3]: self._render_results_and_learning_page,
        }
        page_renderers[selected_page]()

    def _render_results_and_learning_page(self) -> None:
        """Combine measured results and the visual learning explanation."""
        self._render_results_page()
        st.divider()
        self._render_learning_page()

    def _configure_page(self) -> None:
        """Configure Streamlit page settings."""
        st.set_page_config(page_title=self.config.page_title, page_icon=self.config.page_icon, layout=self.config.layout)

    def _prepare_directories(self) -> None:
        """Create required shared roots without eagerly recreating participant folders."""
        paths = self.config.project_config.paths
        saved_sessions_root = self.config.model_path.parent.parent
        results_sessions_root = self.config.training_log_path.parent.parent
        saved_sessions_root.mkdir(parents=True, exist_ok=True)
        results_sessions_root.mkdir(parents=True, exist_ok=True)

        stale_counter = results_sessions_root / ".next_user_number"
        if stale_counter.exists():
            stale_counter.unlink()

        paths.docs_dir.mkdir(parents=True, exist_ok=True)
        paths.screenshots_dir.mkdir(parents=True, exist_ok=True)
