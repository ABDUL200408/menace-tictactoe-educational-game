"""Streamlit session-state keys and initialisation for the MENACE interface."""

from __future__ import annotations


import pandas as pd
import streamlit as st

from src.board import Board
from src.menace import MENACEPlayer
class SessionKey:
    """Central names used for MENACE Streamlit session-state values."""
    MENACE = 'menace_player'
    BOARD = 'board'
    HISTORY = 'training_history'
    COMPARISON = 'comparison_results'
    MESSAGE = 'game_message'
    EXPLANATION = 'last_explanation'
    REWARD_FEEDBACK = 'reward_feedback'
    LAST_MATCHBOX_STATE = 'last_matchbox_state'
    LAST_HUMAN_MOVE = 'last_human_move'
    SYMMETRY_MAPPINGS = 'symmetry_mappings'
    EXPORTED_TRAINING_FIGURES = 'exported_training_figures'
    EXPORTED_COMPARISON_FIGURES = 'exported_comparison_figures'
    LAST_TRAINING_SUMMARY = 'last_training_summary'
    DISPLAY_MODE = 'display_mode'
    SHOW_WELCOME = 'show_welcome'
    LAST_GAME_RESULT = 'last_game_result'
    LAST_BEAD_CHANGES = 'last_bead_changes'
    SHOW_DEMO = 'show_demo'
    DEMO_STEP = 'demo_step'
    DEMO_OUTCOME = 'demo_outcome'
    ACTIVE_PAGE = 'active_page'
    REQUESTED_PAGE = 'requested_page'
    PLAY_READY_BANNER = 'play_ready_banner'
    LEARNING_SOURCE = 'learning_source'
    PERSISTENCE_NOTICE = 'persistence_notice'
    INTERACTIVE_GAMES = 'interactive_games'
    INTERACTIVE_WINS = 'interactive_wins'
    INTERACTIVE_LOSSES = 'interactive_losses'
    INTERACTIVE_DRAWS = 'interactive_draws'
    LAST_TRAINING_CONTEXT = 'last_training_context'
    TRAINING_NOTICE = 'training_notice'
    TECH_SHOW_DEMO = 'technical_show_demo'
    TECH_DEMO_STEP = 'technical_demo_step'
    TECH_DEMO_OUTCOME = 'technical_demo_outcome'
    SYMMETRY_DEMO_STEP = 'symmetry_demo_step'
    PLAY_MOVE_HISTORY = 'play_move_history'
    LEARNING_BASELINE = 'learning_baseline'
    LAST_HUMAN_MEMORY = 'last_human_memory'
    TRAINING_SINCE_HUMAN_GAME = 'training_since_human_game'
    GAME_START_MEMORY = 'game_start_memory'
    LEARNING_EVENTS = 'learning_events'
    PERSISTENCE_ID = 'persistence_id'
    PERSISTENCE_LABEL = 'persistence_label'


class SessionMixin:
    """Initialise and manage persistent values used across Streamlit reruns."""

    def _initialise_session_state(self) -> None:
        """Initialise required Streamlit session-state values when absent."""
        if SessionKey.MENACE not in st.session_state:
            st.session_state[SessionKey.MENACE] = self._new_menace()
        if SessionKey.BOARD not in st.session_state:
            st.session_state[SessionKey.BOARD] = Board.empty()
        if SessionKey.MESSAGE not in st.session_state:
            st.session_state[SessionKey.MESSAGE] = 'Start by choosing a lettered square.'
        if SessionKey.EXPLANATION not in st.session_state:
            st.session_state[SessionKey.EXPLANATION] = 'MENACE explanations will appear after MENACE makes a move.'
        if SessionKey.REWARD_FEEDBACK not in st.session_state:
            st.session_state[SessionKey.REWARD_FEEDBACK] = 'MENACE will explain what it learned when the game ends.'
        if SessionKey.LAST_MATCHBOX_STATE not in st.session_state:
            st.session_state[SessionKey.LAST_MATCHBOX_STATE] = None
        if SessionKey.LAST_HUMAN_MOVE not in st.session_state:
            st.session_state[SessionKey.LAST_HUMAN_MOVE] = None
        if SessionKey.SYMMETRY_MAPPINGS not in st.session_state:
            st.session_state[SessionKey.SYMMETRY_MAPPINGS] = []
        if SessionKey.HISTORY not in st.session_state:
            # A new browser/server session begins with a fresh learning journey.
            # Previous evidence is loaded only when the learner explicitly chooses
            # "Continue previous learning", keeping the model and evidence aligned.
            st.session_state[SessionKey.HISTORY] = pd.DataFrame()
        if SessionKey.COMPARISON not in st.session_state:
            st.session_state[SessionKey.COMPARISON] = pd.DataFrame()
        if SessionKey.EXPORTED_TRAINING_FIGURES not in st.session_state:
            st.session_state[SessionKey.EXPORTED_TRAINING_FIGURES] = []
        if SessionKey.EXPORTED_COMPARISON_FIGURES not in st.session_state:
            st.session_state[SessionKey.EXPORTED_COMPARISON_FIGURES] = {}
        if SessionKey.LAST_TRAINING_SUMMARY not in st.session_state:
            st.session_state[SessionKey.LAST_TRAINING_SUMMARY] = None
        # The application deliberately provides one consistent learner view.
        st.session_state[SessionKey.DISPLAY_MODE] = self.config.simple_mode_name
        if SessionKey.SHOW_WELCOME not in st.session_state:
            st.session_state[SessionKey.SHOW_WELCOME] = True
        if SessionKey.LAST_GAME_RESULT not in st.session_state:
            st.session_state[SessionKey.LAST_GAME_RESULT] = None
        if SessionKey.LAST_BEAD_CHANGES not in st.session_state:
            st.session_state[SessionKey.LAST_BEAD_CHANGES] = []
        if SessionKey.SHOW_DEMO not in st.session_state:
            st.session_state[SessionKey.SHOW_DEMO] = False
        if SessionKey.DEMO_STEP not in st.session_state:
            st.session_state[SessionKey.DEMO_STEP] = 0
        if SessionKey.DEMO_OUTCOME not in st.session_state:
            st.session_state[SessionKey.DEMO_OUTCOME] = "MENACE wins"
        if SessionKey.ACTIVE_PAGE not in st.session_state:
            labels = self.config.simple_page_labels
            st.session_state[SessionKey.ACTIVE_PAGE] = labels[0]
        if SessionKey.REQUESTED_PAGE not in st.session_state:
            st.session_state[SessionKey.REQUESTED_PAGE] = None
        if SessionKey.PLAY_READY_BANNER not in st.session_state:
            st.session_state[SessionKey.PLAY_READY_BANNER] = False
        if SessionKey.LEARNING_SOURCE not in st.session_state:
            st.session_state[SessionKey.LEARNING_SOURCE] = "Fresh start"
        if SessionKey.PERSISTENCE_NOTICE not in st.session_state:
            st.session_state[SessionKey.PERSISTENCE_NOTICE] = (
                "MENACE is starting from zero in this session."
            )
        if SessionKey.INTERACTIVE_GAMES not in st.session_state:
            st.session_state[SessionKey.INTERACTIVE_GAMES] = 0
        if SessionKey.INTERACTIVE_WINS not in st.session_state:
            st.session_state[SessionKey.INTERACTIVE_WINS] = 0
        if SessionKey.INTERACTIVE_LOSSES not in st.session_state:
            st.session_state[SessionKey.INTERACTIVE_LOSSES] = 0
        if SessionKey.INTERACTIVE_DRAWS not in st.session_state:
            st.session_state[SessionKey.INTERACTIVE_DRAWS] = 0
        if SessionKey.LAST_TRAINING_CONTEXT not in st.session_state:
            st.session_state[SessionKey.LAST_TRAINING_CONTEXT] = None
        if SessionKey.TRAINING_NOTICE not in st.session_state:
            st.session_state[SessionKey.TRAINING_NOTICE] = None
        if SessionKey.TECH_SHOW_DEMO not in st.session_state:
            st.session_state[SessionKey.TECH_SHOW_DEMO] = False
        if SessionKey.TECH_DEMO_STEP not in st.session_state:
            st.session_state[SessionKey.TECH_DEMO_STEP] = 0
        if SessionKey.TECH_DEMO_OUTCOME not in st.session_state:
            st.session_state[SessionKey.TECH_DEMO_OUTCOME] = "MENACE wins"
        if SessionKey.SYMMETRY_DEMO_STEP not in st.session_state:
            st.session_state[SessionKey.SYMMETRY_DEMO_STEP] = 0
        if SessionKey.PLAY_MOVE_HISTORY not in st.session_state:
            st.session_state[SessionKey.PLAY_MOVE_HISTORY] = []
        if SessionKey.LEARNING_BASELINE not in st.session_state:
            st.session_state[SessionKey.LEARNING_BASELINE] = (
                self._learning_memory_snapshot(
                    st.session_state[SessionKey.MENACE]
                )
            )
        if SessionKey.LAST_HUMAN_MEMORY not in st.session_state:
            st.session_state[SessionKey.LAST_HUMAN_MEMORY] = None
        if SessionKey.TRAINING_SINCE_HUMAN_GAME not in st.session_state:
            st.session_state[SessionKey.TRAINING_SINCE_HUMAN_GAME] = None
        if SessionKey.GAME_START_MEMORY not in st.session_state:
            st.session_state[SessionKey.GAME_START_MEMORY] = (
                self._learning_memory_snapshot(
                    st.session_state[SessionKey.MENACE]
                )
            )
        if SessionKey.LEARNING_EVENTS not in st.session_state:
            st.session_state[SessionKey.LEARNING_EVENTS] = []

    @staticmethod
    def _learning_memory_snapshot(menace: MENACEPlayer) -> dict[str, int]:
        """Return cumulative learning counters without modifying MENACE state."""
        matchboxes = getattr(menace, "matchboxes", {})
        return {
            "games": int(getattr(menace, "games_played", 0)),
            "matchboxes": len(matchboxes),
            "beads": sum(box.total_beads() for box in matchboxes.values()),
        }

    def _new_menace(self) -> MENACEPlayer:
        """Create a fresh MENACE player from the central project configuration."""
        menace_defaults = self.config.project_config.menace
        return MENACEPlayer(
            mark=menace_defaults.mark,
            name=menace_defaults.name,
            seed=menace_defaults.random_seed,
            use_symmetry=menace_defaults.use_symmetry,
            initial_beads=menace_defaults.initial_beads,
            minimum_beads=menace_defaults.minimum_beads,
            reward_win=menace_defaults.reward_win,
            reward_draw=menace_defaults.reward_draw,
            penalty_loss=menace_defaults.penalty_loss,
        )
