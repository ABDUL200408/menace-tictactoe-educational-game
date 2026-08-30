"""Streamlit persistence controls for MENACE learning journeys."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.menace import MENACEPlayer, Matchbox
from src.persistence import ModelPersistence
from ui.session import SessionKey

class PersistenceControlsMixin:
    """Provide save, load, reset, and learning-evidence persistence controls."""

    def _previous_learning_available(self) -> bool:
        """Return whether a saved MENACE model can be continued."""
        return self.config.model_path.exists()

    def _next_versioned_backup_path(self) -> Path:
        """
        Return the next unused numbered training-batch snapshot path.

        ``menace_model.json`` remains the cumulative model. Numbered snapshot
        files represent individual training batches:
            - ``.bak1`` = first batch only
            - ``.bak2`` = second batch only
            - ``.bak3`` = third batch only
        """
        index = 1
        while True:
            candidate = Path(f"{self.config.model_path}.bak{index}")
            if not candidate.exists():
                return candidate
            index += 1

    @staticmethod
    def _matchbox_bead_snapshot(
        menace: MENACEPlayer,
    ) -> dict[str, dict[int, int]]:
        """Capture bead counts immediately before one training batch."""
        return {
            str(state): {
                int(move): int(count)
                for move, count in box.beads.items()
            }
            for state, box in menace.matchboxes.items()
        }

    def _save_isolated_training_batch(
        self,
        *,
        menace: MENACEPlayer,
        before_beads: dict[str, dict[int, int]],
        summary: dict[str, Any],
        requested_games: int,
    ) -> Path:
        """
        Save one numbered model containing only the latest batch's evidence.

        The live ``menace_model.json`` remains cumulative. The numbered snapshot
        stores:
            - outcomes from this batch only;
            - matchboxes whose bead counts changed in this batch;
            - the net bead change contributed by this batch, expressed from the
              normal initial bead count.

        The batch model is an evidence snapshot, not the model used to continue
        training. Later batches may depend on learning retained in the cumulative
        model, so the numbered files document each batch's contribution rather
        than creating independent retraining experiments.
        """
        import copy

        batch_model = copy.deepcopy(menace)
        initial_beads = int(getattr(menace, "initial_beads", 3))
        minimum_beads = int(getattr(menace, "minimum_beads", 1))

        isolated_matchboxes: dict[str, Matchbox] = {}
        for state, current_box in menace.matchboxes.items():
            state_key = str(state)
            previous = before_beads.get(state_key, {})
            isolated_beads: dict[int, int] = {}
            changed = state_key not in before_beads

            for move, current_count in current_box.beads.items():
                move_key = int(move)
                current_value = int(current_count)
                previous_value = int(
                    previous.get(move_key, initial_beads)
                )
                net_change = current_value - previous_value

                if net_change != 0:
                    changed = True

                # Re-express only this batch's net contribution from the normal
                # starting bead count, while respecting MENACE's minimum.
                isolated_beads[move_key] = max(
                    minimum_beads,
                    initial_beads + net_change,
                )

            if changed:
                isolated_matchboxes[state_key] = Matchbox(
                    state=state_key,
                    beads=isolated_beads,
                )

        batch_model.matchboxes = isolated_matchboxes

        batch_games = int(
            summary.get(
                "games",
                summary.get("menace_games_played", requested_games),
            )
            or requested_games
        )
        batch_wins = int(
            summary.get("menace_wins", summary.get("wins", 0)) or 0
        )
        batch_losses = int(
            summary.get(
                "opponent_wins",
                summary.get("random_wins", summary.get("losses", 0)),
            )
            or 0
        )
        batch_draws = int(summary.get("draws", 0) or 0)

        batch_model.games_played = batch_games
        batch_model.wins = batch_wins
        batch_model.losses = batch_losses
        batch_model.draws = batch_draws

        backup_path = self._next_versioned_backup_path()
        ModelPersistence(backup_path).save(batch_model)
        return backup_path

    def _save_learning_bundle(self, *, show_message: bool = False) -> None:
        """
        Save the current MENACE model and its visible evidence as one bundle.

        Training and comparison code already writes the CSV evidence. This helper
        guarantees that the model producing the evidence is saved alongside it.
        """
        persistence = ModelPersistence(self.config.model_path)
        persistence.save(st.session_state[SessionKey.MENACE])

        history = self._normalise_history(
            st.session_state.get(SessionKey.HISTORY, pd.DataFrame())
        )
        if not history.empty:
            self.config.training_log_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            history.to_csv(self.config.training_log_path, index=False)

        comparison = st.session_state.get(
            SessionKey.COMPARISON,
            pd.DataFrame(),
        )
        if comparison is not None and not comparison.empty:
            self.config.comparison_results_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            comparison.to_csv(
                self.config.comparison_results_path,
                index=False,
            )

        st.session_state[SessionKey.LEARNING_SOURCE] = "Current saved learning"
        st.session_state[SessionKey.PERSISTENCE_NOTICE] = (
            "This learning journey has been saved and can be continued later."
        )
        if show_message:
            st.sidebar.success(
                "MENACE and its evidence were saved together."
                if self._is_simple_mode()
                else "Model and evidence bundle saved."
            )

    def _continue_previous_learning(self) -> None:
        """Load the saved MENACE model and matching evidence together."""
        try:
            if not self.config.model_path.exists():
                raise FileNotFoundError(
                    "No saved MENACE model was found. Start a new practice run first."
                )

            persistence = ModelPersistence(self.config.model_path)
            menace = persistence.load()
            history = self._load_csv(self.config.training_log_path)
            comparison = self._load_csv(
                self.config.comparison_results_path
            )

            st.session_state[SessionKey.MENACE] = menace
            st.session_state[SessionKey.LEARNING_BASELINE] = (
                self._learning_memory_snapshot(menace)
            )
            st.session_state[SessionKey.HISTORY] = history
            st.session_state[SessionKey.COMPARISON] = comparison
            st.session_state[SessionKey.LAST_TRAINING_SUMMARY] = None
            st.session_state[SessionKey.LAST_TRAINING_CONTEXT] = None
            st.session_state[SessionKey.LAST_HUMAN_MEMORY] = None
            st.session_state[SessionKey.TRAINING_SINCE_HUMAN_GAME] = None
            st.session_state[SessionKey.LEARNING_EVENTS] = []
            st.session_state[SessionKey.GAME_START_MEMORY] = (
                self._learning_memory_snapshot(menace)
            )
            st.session_state[SessionKey.LEARNING_SOURCE] = (
                "Continued previous learning"
            )
            st.session_state[SessionKey.PERSISTENCE_NOTICE] = (
                "MENACE and the evidence shown on the pages come from the "
                "same saved learning journey."
            )
            self._start_new_game(
                message=(
                    "Previous learning loaded. A new board is ready, and MENACE "
                    "keeps the matchboxes and beads it learned before."
                    if self._is_simple_mode()
                    else "Saved model and matching evidence loaded."
                )
            )
            st.sidebar.success(
                "Previous MENACE learning loaded."
                if self._is_simple_mode()
                else "Saved learning bundle loaded."
            )
        except Exception as exc:
            st.sidebar.error(f"Could not continue previous learning: {exc}")

    def _clear_saved_learning_files(self) -> tuple[list[Path], list[Path]]:
        """Delete every saved artefact belonging to the current participant."""
        import shutil

        session_dirs = [
            self.config.model_path.parent,
            self.config.training_log_path.parent,
        ]
        removed: list[Path] = []
        failed: list[Path] = []

        for directory in session_dirs:
            if not directory.exists():
                continue
            try:
                shutil.rmtree(directory)
                removed.append(directory)
            except OSError:
                failed.append(directory)

        failed.extend(
            directory
            for directory in session_dirs
            if directory.exists() and directory not in failed
        )
        return removed, failed

    def _start_learning_from_zero(self) -> None:
        """
        Reset the active model, visible evidence, board, and saved learning files.

        This is deliberately different from New Game, which only clears the board.
        """
        removed, failed = self._clear_saved_learning_files()
        if failed:
            names = ", ".join(str(path) for path in failed[:5])
            if len(failed) > 5:
                names += f", and {len(failed) - 5} more"
            st.session_state[SessionKey.PERSISTENCE_NOTICE] = (
                "Fresh start was not completed because saved files could not "
                f"be deleted: {names}. Close any program using these files and "
                "try again."
            )
            st.sidebar.error(
                "MENACE was not reset because some saved learning files could "
                f"not be deleted: {names}"
            )
            return
        st.session_state[SessionKey.MENACE] = self._new_menace()
        st.session_state[SessionKey.LEARNING_BASELINE] = (
            self._learning_memory_snapshot(
                st.session_state[SessionKey.MENACE]
            )
        )
        st.session_state[SessionKey.HISTORY] = pd.DataFrame()
        st.session_state[SessionKey.COMPARISON] = pd.DataFrame()
        st.session_state[SessionKey.LAST_TRAINING_SUMMARY] = None
        st.session_state[SessionKey.EXPORTED_TRAINING_FIGURES] = []
        st.session_state[SessionKey.EXPORTED_COMPARISON_FIGURES] = {}
        st.session_state[SessionKey.LEARNING_SOURCE] = "Fresh start"
        st.session_state[SessionKey.INTERACTIVE_GAMES] = 0
        st.session_state[SessionKey.INTERACTIVE_WINS] = 0
        st.session_state[SessionKey.INTERACTIVE_LOSSES] = 0
        st.session_state[SessionKey.INTERACTIVE_DRAWS] = 0
        st.session_state[SessionKey.LAST_TRAINING_CONTEXT] = None
        st.session_state[SessionKey.LAST_HUMAN_MEMORY] = None
        st.session_state[SessionKey.TRAINING_SINCE_HUMAN_GAME] = None
        st.session_state[SessionKey.LEARNING_EVENTS] = []
        st.session_state[SessionKey.GAME_START_MEMORY] = (
            self._learning_memory_snapshot(
                st.session_state[SessionKey.MENACE]
            )
        )
        st.session_state[SessionKey.PERSISTENCE_NOTICE] = (
            "MENACE is at the beginning: no saved practice results, "
            "comparison results, learned matchboxes, or exported figures are "
            f"active. Removed {len(removed)} saved location(s)."
        )
        self._start_new_game(
            message=(
                "MENACE is starting from zero. It has no previous practice "
                "results or learned matchboxes."
                if self._is_simple_mode()
                else "MENACE, training history, comparison evidence, and saved "
                "learning files have been reset."
            )
        )

        st.rerun()

    def _save_model(self) -> None:
        """Save MENACE and its matching evidence as one consistent bundle."""
        try:
            self._save_learning_bundle(show_message=True)
        except Exception as exc:
            st.sidebar.error(f'Save failed: {exc}')

    def _load_model(self) -> None:
        """Load MENACE and its matching evidence as one consistent bundle."""
        self._continue_previous_learning()

    def _load_csv(self, path: Path) -> pd.DataFrame:
        """Load a CSV evidence file safely for interface display."""
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
