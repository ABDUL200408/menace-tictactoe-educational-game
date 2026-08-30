"""Streamlit navigation and sidebar helpers for the MENACE interface."""

from __future__ import annotations


import streamlit as st

from ui.session import SessionKey

class NavigationMixin:
    """Provide page navigation, sidebar controls, and progress summaries."""

    def _button_help(self, label: str) -> str:
        """Return a one-line hover explanation for navigation and control buttons."""
        simple_help = {
            "play": "Play against MENACE and watch coloured beads explain each choice.",
            "train": "Let MENACE practise games so it can learn which moves help.",
            "compare": "See how MENACE performs against three different computer opponents.",
            "result and learning": "View results, coloured explanations, and how matchboxes and beads teach MENACE.",
            "Play": "Play noughts and crosses against MENACE and watch it learn from the result.",
            "Start here: beginner guide": "Open the beginner guide explaining matchboxes, beads, rewards and how to play.",
            "▶ Watch demo": "Watch a complete example showing how MENACE plays and changes its beads.",
            "New Game": "Clear the board and start again while keeping everything MENACE has learned.",
            "Continue previous learning": "Load the saved MENACE model together with its matching learning evidence.",
            "Start MENACE from the beginning": "Erase active and saved learning, then return MENACE to its initial state.",
        }
        return simple_help.get(label, f"Open {label}.")

    def _render_page_header(self) -> None:
        """Render a concise project header suited to the selected audience."""
        title = "MENACE: a machine that learns by playing"
        subtitle = (
            "Learn how a simple machine improves through rewards. Each board "
            "situation has a matchbox, and its coloured beads represent the "
            "legal squares MENACE can choose. Helpful choices gain beads, so "
            "they become more likely next time."
        )
        st.markdown(
            f"<div class='menace-hero'><h1>{title}</h1><p>{subtitle}</p></div>",
            unsafe_allow_html=True,
        )


    def _render_current_page_banner(self, page_name: str) -> None:
        """Show the active page and a two-line explanation at its top."""
        page_details = {
            "play": (
                "Play",
                "Play noughts and crosses against MENACE and inspect the matchbox used for each choice.",
                "At the end, see how rewards, draws, or penalties change its green beads.",
            ),
            "train": (
                "Train",
                "Give MENACE practice games against a random opponent so it can improve through experience.",
                "This page shows the latest batch, memory growth, and win, loss, and draw totals.",
            ),
            "compare": (
                "Compare",
                "Test a separately trained copy of MENACE against Random, Heuristic, and Minimax opponents.",
                "Use the results to see which playing style MENACE finds easiest or hardest.",
            ),
            "result and learning": (
                "Results & Learning",
                "Review the latest practice evidence and how MENACE's performance changed during training.",
                "Open the graphs and learning explanation to connect matchboxes, green beads, and outcomes.",
            ),
        }
        title, line_one, line_two = page_details.get(
            page_name,
            (page_name.title(), "You are viewing this page.", "Use the controls below to continue."),
        )
        st.markdown(
            "<div class='current-page-banner' role='status'>"
            f"<div class='current-page-kicker'>You are here</div>"
            f"<div class='current-page-title'>{title}</div>"
            f"<div class='current-page-description'>{line_one}<br>{line_two}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    def _render_sidebar(self) -> None:
        """Render page navigation, game controls, and learning-journey controls."""
        current_labels = self.config.simple_page_labels

        requested_page = st.session_state.get(SessionKey.REQUESTED_PAGE)
        if requested_page in current_labels:
            st.session_state[SessionKey.ACTIVE_PAGE] = requested_page
            st.session_state[SessionKey.REQUESTED_PAGE] = None

        active_page = st.session_state.get(
            SessionKey.ACTIVE_PAGE,
            current_labels[0],
        )
        if active_page not in current_labels:
            active_page = current_labels[0]

        st.sidebar.markdown("### Choose a page")
        st.sidebar.caption("Select a page. Hover over a button to see what it does.")
        selected_page = active_page
        for page_index, page_label in enumerate(current_labels):
            is_active = page_label == active_page
            button_label = f"✓ {page_label}" if is_active else page_label
            if st.sidebar.button(
                button_label,
                key=f"sidebar_page_{page_index}",
                width="stretch",
                type="primary" if is_active else "secondary",
                help=self._button_help(page_label),
            ):
                selected_page = page_label
                st.session_state[SessionKey.ACTIVE_PAGE] = page_label
                st.session_state[SessionKey.REQUESTED_PAGE] = None
                st.rerun()
        st.session_state[SessionKey.ACTIVE_PAGE] = selected_page

        play_page = current_labels[0]
        on_play_page = (
            st.session_state.get(SessionKey.ACTIVE_PAGE, play_page)
            == play_page
        )

        if on_play_page:
            instructions_label = "Start here: beginner guide"
            if st.sidebar.button(
                instructions_label,
                width="stretch",
                help=self._button_help(instructions_label),
            ):
                st.session_state[SessionKey.SHOW_WELCOME] = True

            if st.sidebar.button(
                "▶ Watch demo",
                width="stretch",
                help=self._button_help("▶ Watch demo"),
            ):
                st.session_state[SessionKey.SHOW_DEMO] = True
                st.session_state[SessionKey.DEMO_STEP] = 0
                st.session_state[SessionKey.SHOW_WELCOME] = False
                st.rerun()

        st.sidebar.divider()
        st.sidebar.header("Game controls")

        if st.sidebar.button(
            "New Game",
            width="stretch",
            help=self._button_help("New Game"),
        ):
            self._start_new_game(
                message="A new board is ready. MENACE keeps everything it learned before."
            )

        st.sidebar.markdown("#### Learning journey")

        if self._previous_learning_available():
            continue_label = "Continue previous learning"
            if st.sidebar.button(
                continue_label,
                key="continue_previous_learning",
                width="stretch",
                help=self._button_help(continue_label),
            ):
                self._continue_previous_learning()
        else:
            st.sidebar.caption(
                "No previous saved learning is available yet."
            )

        reset_label = "Start MENACE from the beginning"
        reset_confirmed = st.sidebar.checkbox(
            "I understand this permanently deletes the saved learning journey",
            key="confirm_full_learning_reset",
            help=(
                "Required before the full reset button is enabled. New Game "
                "does not delete learning."
            ),
        )
        if st.sidebar.button(
            reset_label,
            key="reset_learning_journey",
            width="stretch",
            help=self._button_help(reset_label),
            disabled=not reset_confirmed,
        ):
            self._start_learning_from_zero()

        source = st.session_state.get(
            SessionKey.LEARNING_SOURCE,
            "Fresh start",
        )
        notice = st.session_state.get(
            SessionKey.PERSISTENCE_NOTICE,
            "",
        )
        st.sidebar.info(f"**Current journey:** {source}\n\n{notice}")

        st.sidebar.divider()
        if not self._is_simple_mode():
            if st.sidebar.button(
                "Save Model",
                width="stretch",
                help=self._button_help("Save Model"),
            ):
                self._save_model()
            if st.sidebar.button(
                "Load Model",
                width="stretch",
                help=self._button_help("Load Model"),
            ):
                self._load_model()
            st.sidebar.divider()

        self._render_sidebar_summary()

    def _render_sidebar_summary(self) -> None:
        """Render current MENACE progress for the active interface context."""
        menace = st.session_state[SessionKey.MENACE]
        summary = menace.training_summary()

        if self._is_simple_mode():
            labels = self.config.simple_page_labels
            active_page = st.session_state.get(SessionKey.ACTIVE_PAGE, labels[0])
            if active_page == labels[0]:
                human_games = int(st.session_state.get(SessionKey.INTERACTIVE_GAMES, 0))
                human_wins = int(st.session_state.get(SessionKey.INTERACTIVE_WINS, 0))
                human_losses = int(st.session_state.get(SessionKey.INTERACTIVE_LOSSES, 0))
                human_draws = int(st.session_state.get(SessionKey.INTERACTIVE_DRAWS, 0))
                st.sidebar.markdown("### Your Play results")
                st.sidebar.metric("Human games finished", human_games)
                st.sidebar.caption(
                    f"You won: {human_wins} · MENACE won: {human_losses} · Draws: {human_draws}"
                )
                st.sidebar.caption(
                    "Practice outcomes are kept on the Practice and Results pages."
                )
            else:
                st.sidebar.markdown("### MENACE practice so far")
                st.sidebar.metric("Practice games played", summary["games_played"])
                st.sidebar.metric("Practice games won", summary["wins"])
                st.sidebar.metric("Practice win rate", f"{summary['win_rate']:.1%}")
                st.sidebar.caption(
                    "These are practice results only; human games are not included."
                )
        else:
            st.sidebar.markdown("### MENACE Summary")
            st.sidebar.write(
                {
                    "games_played": summary["games_played"],
                    "wins": summary["wins"],
                    "losses": summary["losses"],
                    "draws": summary["draws"],
                    "win_rate": f"{summary['win_rate']:.1%}",
                    "matchboxes": summary["matchboxes"],
                    "total_beads": summary["total_beads"],
                    "use_symmetry": summary["use_symmetry"],
                }
            )

    def _set_active_page(self, page_name: str) -> None:
        """
        Change the visible application page.

        The page name is stored in session state so programmatic navigation
        remains reliable across Streamlit reruns.
        """
        labels = (
            self.config.simple_page_labels
            if self._is_simple_mode()
            else self.config.technical_page_labels
        )
        if page_name != labels[0]:
            st.session_state[SessionKey.SHOW_DEMO] = False
            st.session_state[SessionKey.DEMO_STEP] = 0
            st.session_state[SessionKey.TECH_SHOW_DEMO] = False
            st.session_state[SessionKey.TECH_DEMO_STEP] = 0

        st.session_state[SessionKey.ACTIVE_PAGE] = page_name
        st.session_state[SessionKey.REQUESTED_PAGE] = None
