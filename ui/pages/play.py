"""Streamlit Play page for human interaction with MENACE."""

from __future__ import annotations

import re
from typing import Any, Optional

import pandas as pd
import streamlit as st

from src.board import Board
from src.menace import MENACEPlayer, Matchbox
from ui.session import SessionKey

class PlayPageMixin:
    """Render the Play page and coordinate human-versus-MENACE interaction."""

    def _start_new_game(self, message: str='New game started.') -> None:
        """
        Start a new board and clear incomplete MENACE decisions.

        This prevents stale move decisions from an unfinished game being
        rewarded or punished later.
        """
        menace = st.session_state[SessionKey.MENACE]
        menace.reset_game_memory()
        st.session_state[SessionKey.BOARD] = Board.empty()
        st.session_state[SessionKey.MESSAGE] = message
        st.session_state[SessionKey.EXPLANATION] = 'MENACE explanations will appear after MENACE makes a move.'
        st.session_state[SessionKey.REWARD_FEEDBACK] = 'MENACE will explain what it learned when the game ends.'
        st.session_state[SessionKey.LAST_MATCHBOX_STATE] = None
        st.session_state[SessionKey.LAST_HUMAN_MOVE] = None
        st.session_state[SessionKey.SYMMETRY_MAPPINGS] = []
        st.session_state[SessionKey.PLAY_MOVE_HISTORY] = []
        st.session_state[SessionKey.LAST_GAME_RESULT] = None
        st.session_state[SessionKey.LAST_BEAD_CHANGES] = []
        st.session_state[SessionKey.GAME_START_MEMORY] = (
            self._learning_memory_snapshot(menace)
        )
        st.rerun()

    def _render_play_page(self) -> None:
        """
        Render gameplay with a responsive learner view and supporting evidence.

        The learner-facing layout uses one vertical reading path so the board,
        matchbox, explanations, and reward cards remain readable on laptops,
        tablets, small browser windows, and exported pages.
        """
        if st.session_state.get(SessionKey.PLAY_READY_BANNER, False):
            st.success(
                "🎮 **MENACE is ready!** It kept everything learned during "
                "practice. The board is new—try to beat it now."
            )
            st.toast("MENACE is ready to play!", icon="🎮")
            st.session_state[SessionKey.PLAY_READY_BANNER] = False

        if self._is_simple_mode():
            st.subheader("Play with MENACE")
            st.write(
                "You are **X** and MENACE is **O**. Choose a lettered square. "
                "Try to win, or see whether MENACE can block you."
            )

            self._render_simple_symmetry_explainer()

            # 1. Play board — compact and given the full page width.
            with st.container(border=True):
                self._render_board()
                board = st.session_state[SessionKey.BOARD]
                if board.is_terminal():
                    st.caption(
                        "The game has finished. Start a new game to play again."
                    )
                else:
                    st.caption(
                        "Empty squares use letters A–I. "
                        "No internal indexes are shown."
                    )

                self._render_move_history()

            # 2. Latest decision and matchbox — full width for a readable
            #    responsive bead-card grid.
            with st.container(border=True):
                st.markdown("### Inside the machine")
                self._render_latest_move_summary()
                self._render_current_matchbox_visualisation()

            # 3. End-of-game learning — full width for before/after bead cards.
            with st.container(border=True):
                st.markdown("### What MENACE learned from this game")
                st.info(st.session_state[SessionKey.REWARD_FEEDBACK])
                self._render_simple_reward_visual()

            # 4. Live shared memory is shown after the frozen human-game evidence
            #    so the page remains chronological when later practice changes
            #    the same MENACE model.
            with st.container(border=True):
                st.markdown("### Shared memory after all learning")
                self._render_learned_memory_before_game()

            # Optional evidence remains collapsed to keep the main learner path concise.
            self._render_simple_evidence_panel("play")
            return

        st.subheader("Play Against MENACE")
        st.write(
            "You play as **X**. MENACE plays as **O**. This view exposes the "
            "canonical state, symmetry mapping, matchbox coordinates and exact probabilities."
        )
        st.caption(
            "The playable board uses the same lettered squares A–I as the demo. "
            "MENACE still stores and transforms positions internally as indexes "
            "0–8; only the learner-facing Play-page labels have changed."
        )
        left, right = st.columns([1.35, 1])
        with left:
            self._render_board()
            st.markdown("### Reward / Punishment Feedback")
            st.info(st.session_state[SessionKey.REWARD_FEEDBACK])
            self._render_current_matchbox_visualisation()
        with right:
            st.info(st.session_state[SessionKey.MESSAGE])
            st.markdown("### MENACE Decision Explanation")
            st.write(
                self._letterise_play_text(
                    st.session_state[SessionKey.EXPLANATION]
                )
            )
            st.caption(
                "This formatted explanation is the primary decision summary. "
                "The full raw log is available in the collapsed section beside "
                "the matchbox evidence."
            )
            self._render_symmetry_plain_language_note()
            self._render_symmetry_mapping_explanations()
            board = st.session_state[SessionKey.BOARD]
            st.markdown("### Current Board")
            st.code(self._lettered_board_text(board))
            latest = self._latest_symmetry_mapping_record()
            if latest is not None:
                st.markdown("### Board Before Latest MENACE Move")
                st.code(
                    self._lettered_board_text(
                        Board.from_string(str(latest["board_before"]))
                    )
                )
            st.markdown("### Current Board Explanation After MENACE Move")
            st.code(self._letterise_play_text(board.explain_state()))

    def _render_learned_memory_before_game(self) -> None:
        """Show the shared live memory and distinguish it from the last game."""
        menace = st.session_state[SessionKey.MENACE]
        matchboxes = getattr(menace, "matchboxes", {})
        total_beads = sum(box.total_beads() for box in matchboxes.values())
        human_games = int(st.session_state.get(SessionKey.INTERACTIVE_GAMES, 0))

        st.markdown(
            f"""
            <div class="shared-memory-banner">
              <div class="shared-memory-title">🧠 MENACE's shared current memory</div>
              <div class="shared-memory-text">This is the current opponent you play against. Human play and practice can both change its matchboxes.</div>
              <div class="memory-source-grid">
                <div class="memory-source human"><b>👤 {human_games:,}</b><span>Human games</span></div>
                <div class="memory-source practice"><b>📦 {len(matchboxes):,}</b><span>Current matchboxes</span></div>
                <div class="memory-source total"><b>🟢 {total_beads:,}</b><span>Current beads</span></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        training_change = st.session_state.get(SessionKey.TRAINING_SINCE_HUMAN_GAME)
        if training_change:
            st.warning(
                "**MENACE practised after your last human game.** The completed "
                "game, result and reward shown below have not changed. Only the "
                "shared live memory changed. Detailed practice totals and "
                "outcomes remain on the Practice and Results pages."
            )

        with st.expander(
            f"📦 Open the shared live memory — {len(matchboxes):,} matchboxes",
            expanded=not bool(st.session_state[SessionKey.PLAY_MOVE_HISTORY]),
        ):
            if not matchboxes:
                st.info(
                    "MENACE has no learned matchboxes yet. Give it practice or "
                    "finish a game, then its saved board situations will appear here."
                )
                return

            st.write(
                "This is a **live view**, not a frozen record of your last game. "
                "Starting a New Game clears only the board. Practice training can "
                "change these same matchboxes and beads."
            )
            metrics = st.columns(3)
            metrics[0].metric("Matchboxes currently in memory", f"{len(matchboxes):,}")
            metrics[1].metric("Beads in memory", f"{total_beads:,}")
            metrics[2].metric(
                "Human games finished",
                f"{int(st.session_state.get(SessionKey.INTERACTIVE_GAMES, 0)):,}",
            )
            st.info(
                f"**What these totals mean:** {len(matchboxes):,} matchboxes "
                "means MENACE currently remembers that many distinct canonical "
                "board situations. The "
                f"**{total_beads:,} beads in memory** are the sum of the beads "
                "inside all of those matchboxes—not only the matchboxes used in "
                "the latest game. A matchbox stores one board situation; its "
                "beads determine the weighted chance of choosing each legal move."
            )
            self._render_memory_calculation(total_beads)

            sort_choice = st.selectbox(
                "Arrange learned board situations by",
                ("Most beads", "Fewest beads", "Board position"),
                key="play_learned_memory_sort",
                help="This changes only the display order; it does not change MENACE.",
            )
            if sort_choice == "Most beads":
                ordered_states = sorted(
                    matchboxes,
                    key=lambda state: (-matchboxes[state].total_beads(), state),
                )
            elif sort_choice == "Fewest beads":
                ordered_states = sorted(
                    matchboxes,
                    key=lambda state: (matchboxes[state].total_beads(), state),
                )
            else:
                ordered_states = sorted(matchboxes)
            state_labels = {
                state: (
                    f"Board state: {self._memory_board_one_line(state)} "
                    f"— {matchboxes[state].total_beads():,} beads"
                )
                for state in ordered_states
            }
            selected_state = st.selectbox(
                "Open the matchbox for a learned board situation",
                options=ordered_states,
                format_func=lambda state: state_labels[state],
                key="play_learned_memory_matchbox",
                help=(
                    "Each item is one canonical board situation. Rotated or "
                    "reflected versions may share this same matchbox."
                ),
            )
            box = matchboxes[selected_state]

            left, right = st.columns([1, 1.6])
            with left:
                st.markdown("**Board situation stored by this matchbox**")
                st.code(self._memory_board_grid(selected_state), language=None)
                st.caption(
                    "This is the canonical stored view. X and O are occupied "
                    "squares; A–I are available matchbox moves."
                )
            with right:
                probabilities = box.probabilities()
                rows = [
                    {
                        "Square": self._square_label(int(move)),
                        "Beads": int(count),
                        "Chance": f"{probabilities[int(move)]:.1%}",
                    }
                    for move, count in sorted(box.beads.items())
                ]
                st.markdown("**MENACE's current memory for this board situation**")
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                st.caption(
                    "Chance = beads for this move ÷ total beads in this "
                    f"matchbox. Here the denominator is {box.total_beads():,}. "
                    "The percentage displayed in each row uses that row's "
                    "actual bead count."
                )

            snapshot = st.session_state.get(SessionKey.LAST_HUMAN_MEMORY)
            if snapshot and training_change:
                previous_box = snapshot.get("matchboxes", {}).get(selected_state)
                if previous_box is not None:
                    comparison_rows = []
                    for move in sorted(set(previous_box) | set(box.beads)):
                        before = int(previous_box.get(move, 0))
                        after = int(box.beads.get(move, 0))
                        comparison_rows.append({
                            "Square": self._square_label(int(move)),
                            "After last human game": before,
                            "Current after practice": after,
                            "Change": after - before,
                        })
                    st.markdown("**How practice changed this same matchbox**")
                    st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)
                    st.caption(
                        f"Total beads in this matchbox: {sum(previous_box.values()):,} "
                        f"after your last human game → {box.total_beads():,} now."
                    )
                else:
                    st.info(
                        "This matchbox did not exist immediately after your last "
                        "human game. It was created later during practice."
                    )

            st.caption(
                "A count such as D = 9 proves that this stored move currently "
                "has nine beads. It does not prove one particular earlier win: "
                "the count can reflect several wins, draws, losses, training "
                "games, and symmetry-shared board positions."
            )

    def _render_memory_calculation(self, current_total: int) -> None:
        """Explain the cumulative bead total from recorded learning events."""
        baseline = st.session_state.get(SessionKey.LEARNING_BASELINE, {})
        opening = int(baseline.get("beads", 0))
        events = list(st.session_state.get(SessionKey.LEARNING_EVENTS, []))

        st.markdown("#### How was “Beads in memory” calculated?")
        st.write(
            "**Beads in memory** is the total number of beads across every "
            "matchbox currently stored by MENACE. Each learning event changes "
            "that shared total in two possible ways: it may create new "
            "matchboxes with initial beads, and it may reward or penalise the "
            "moves MENACE selected. Matchboxes that already existed contribute "
            "no new initial beads, even though MENACE can use and update them."
        )
        st.code(
            "Previous memory\n"
            "+ initial beads placed in newly created matchboxes\n"
            "+ actual reward or penalty applied to selected moves\n"
            "= memory after the learning event"
        )
        if not events:
            st.code(f"Journey-start memory = {opening:,} beads")
            return

        rows = []
        for event in events:
            rows.append(
                {
                    "Learning event": str(event["label"]),
                    "Before": int(event["before"]),
                    "New matchbox beads (human game)": int(event.get("created", 0)),
                    "Net result / practice change": int(event.get("reinforcement", 0)),
                    "After": int(event["after"]),
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        terms = [f"{opening:,}"]
        for event in events:
            delta = int(event["after"]) - int(event["before"])
            terms.append(f"{delta:+,}")
        st.code(
            "Journey start + each learning event = current memory\n"
            + " ".join(terms)
            + f" = {current_total:,} beads"
        )

        latest = events[-1]
        if latest.get("kind") == "human":
            created = int(latest.get("created", 0))
            reinforcement = int(latest.get("reinforcement", 0))
            box_totals_before = list(latest.get("box_totals_before", []))
            used_count = len(box_totals_before)
            result_name = str(latest.get("result", "")).strip().lower()
            if not result_name:
                label_result = re.search(
                    r"\((?:MENACE\s+)?(win|draw|loss)\)\s*$",
                    str(latest.get("label", "")),
                    flags=re.IGNORECASE,
                )
                result_name = (
                    label_result.group(1).lower() if label_result else ""
                )
            result_label = (
                f"{result_name} "
                if result_name in {"win", "draw", "loss"}
                else ""
            )
            st.code(
                f"{latest['label']}:\n"
                f"{int(latest['before']):,} beads before the game\n"
                f"+ {created:,} initial beads from newly created matchboxes\n"
                f"{reinforcement:+,} actual {result_label}reward/penalty "
                f"across {used_count:,} selected MENACE "
                f"{'move' if used_count == 1 else 'moves'}\n"
                f"= {int(latest['after']):,} beads after the game"
            )
            if created == 0 and used_count:
                st.info(
                    f"**Why are new-matchbox beads 0?** MENACE still used "
                    f"{used_count:,} "
                    f"{'matchbox' if used_count == 1 else 'matchboxes'} in this "
                    "game, but every one already existed in the shared memory "
                    "from earlier human play or practice. Therefore, no initial "
                    "beads were created. The existing matchboxes could still "
                    f"receive the {reinforcement:+,} result update shown above."
                )
            elif created > 0:
                st.info(
                    f"**Why were {created:,} initial beads added?** At least one "
                    "board situation in this game had not been stored before. "
                    "MENACE created a matchbox for each new situation and placed "
                    "the configured initial number of beads on every legal move. "
                    "This initial-bead amount is separate from the result reward "
                    "or penalty."
                )

            if box_totals_before and latest.get("box_totals_after"):
                before_text = " + ".join(
                    f"{value:,}" for value in box_totals_before
                )
                after_text = " + ".join(
                    f"{value:,}" for value in latest["box_totals_after"]
                )
                used_before = sum(box_totals_before)
                used_after = sum(latest["box_totals_after"])
                unchanged_beads = int(latest["before"]) + created - used_before
                st.markdown("**Matchboxes used in the latest human game**")
                st.code(
                    "Used matchboxes before reinforcement:\n"
                    f"{before_text} = {used_before:,} beads\n\n"
                    "Used matchboxes after reinforcement:\n"
                    f"{after_text} = {used_after:,} beads\n\n"
                    "Change in the used matchboxes:\n"
                    f"{used_after:,} − {used_before:,} = "
                    f"{used_after - used_before:+,} beads"
                )
                st.write(
                    f"These {used_count:,} matchboxes are only the matchboxes "
                    "MENACE used in this game. Their "
                    f"**{used_before:,} beads before reinforcement** are not the "
                    f"whole memory of **{int(latest['before']):,} beads**. "
                    "The other stored matchboxes were not selected in this game "
                    "and therefore did not receive this game's result update."
                )
                st.code(
                    "Second check using the whole memory:\n"
                    f"{unchanged_beads:,} beads outside the used matchboxes\n"
                    f"+ {used_after:,} beads in the used matchboxes after learning\n"
                    f"= {int(latest['after']):,} beads in current memory"
                )

    def _memory_board_grid(self, state: str) -> str:
        """Return one stored canonical state as a learner-facing 3×3 grid."""
        cells = [
            value if value != Board.EMPTY else self._square_label(index)
            for index, value in enumerate(state)
        ]
        return (
            f"{cells[0]} | {cells[1]} | {cells[2]}\n"
            "---------\n"
            f"{cells[3]} | {cells[4]} | {cells[5]}\n"
            "---------\n"
            f"{cells[6]} | {cells[7]} | {cells[8]}"
        )

    def _memory_board_one_line(self, state: str) -> str:
        """Return a compact label for a stored canonical board state."""
        cells = [value if value != Board.EMPTY else "·" for value in state]
        return " / ".join(
            "".join(cells[start:start + 3])
            for start in range(0, Board.BOARD_SIZE, 3)
        )

    def _render_simple_symmetry_explainer(self) -> None:
        """Show the eight square symmetries as a learner-controlled demo."""
        with st.container(border=True):
            st.markdown("### 🔄 How MENACE recognises turned boards")
            st.write(
                "The same pattern can appear in a different direction. MENACE "
                "turns or reflects equivalent boards into one standard version, "
                "called the **canonical state**, and reuses one matchbox."
            )

            all_steps = self._symmetry_demo_steps()
            original_step = all_steps[0]
            steps = all_steps[1:]
            step_index = int(
                st.session_state.get(SessionKey.SYMMETRY_DEMO_STEP, 0)
            ) % len(steps)
            step = steps[step_index]

            st.progress(step_index / (len(steps) - 1))
            st.caption(f"Transformation {step_index + 1} of {len(steps)}")

            original_column, arrow_column, transformed_column = st.columns(
                [1, 0.16, 1]
            )
            with original_column:
                st.markdown(f"**{original_step['title']}**")
                st.code(
                    self._symmetry_demo_board_text(original_step["x_square"])
                )
                st.caption("This original board remains fixed for comparison.")
            with arrow_column:
                st.markdown("### →")
            with transformed_column:
                st.markdown(f"**{step['title']}**")
                st.code(self._symmetry_demo_board_text(step["x_square"]))
                st.caption("Only this equivalent board changes when you continue.")

            st.info(step["explanation"])

            previous, reset, following = st.columns(3)
            with previous:
                if st.button(
                    "← Previous",
                    key="symmetry_demo_previous",
                    disabled=step_index == 0,
                    help="Show the previous board transformation.",
                ):
                    st.session_state[SessionKey.SYMMETRY_DEMO_STEP] = step_index - 1
                    st.rerun()
            with reset:
                if st.button(
                    "↺ First transformation",
                    key="symmetry_demo_original",
                    disabled=step_index == 0,
                    help="Return the right-hand board to the first reflection.",
                ):
                    st.session_state[SessionKey.SYMMETRY_DEMO_STEP] = 0
                    st.rerun()
            with following:
                next_label = (
                    "Next →" if step_index < len(steps) - 1 else "Start again ↺"
                )
                if st.button(
                    next_label,
                    key="symmetry_demo_next",
                    help="Show the next equivalent rotation or reflection.",
                ):
                    st.session_state[SessionKey.SYMMETRY_DEMO_STEP] = (
                        step_index + 1
                    ) % len(steps)
                    st.rerun()

            st.caption(
                "With only one X in a corner, different transformations can "
                "occasionally look the same. They are still different geometric "
                "operations, and a fuller board pattern makes the difference visible."
            )

            st.success(
                "All eight equivalent views → one canonical state → one shared "
                "matchbox. Learning from any orientation improves the same memory."
            )

    @staticmethod
    def _symmetry_demo_board_text(x_square: str) -> str:
        """Return a letter-labelled demo board with X in one visible square."""
        cells = ["X" if letter == x_square else letter for letter in "ABCDEFGHI"]
        return (
            f"{cells[0]} | {cells[1]} | {cells[2]}\n"
            "---------\n"
            f"{cells[3]} | {cells[4]} | {cells[5]}\n"
            "---------\n"
            f"{cells[6]} | {cells[7]} | {cells[8]}"
        )

    @staticmethod
    def _symmetry_demo_steps() -> list[dict[str, str]]:
        """Return the identity and seven dihedral symmetries in demo order."""
        return [
            {
                "title": "Original board — 0° rotation",
                "x_square": "A",
                "explanation": "This is the pattern exactly as the learner first sees it.",
            },
            {
                "title": "Reflection 1 — vertical mirror",
                "x_square": "C",
                "explanation": "The board is reflected left-to-right across its vertical centre line.",
            },
            {
                "title": "Rotation 1 — 90° clockwise",
                "x_square": "C",
                "explanation": "The complete board is turned one quarter-turn clockwise.",
            },
            {
                "title": "Rotation 2 — 180°",
                "x_square": "I",
                "explanation": "The complete board is turned one half-turn.",
            },
            {
                "title": "Rotation 3 — 270° clockwise",
                "x_square": "G",
                "explanation": "The board is turned three quarter-turns clockwise (or 90° anticlockwise).",
            },
            {
                "title": "Reflection 2 — horizontal mirror",
                "x_square": "G",
                "explanation": "The board is reflected top-to-bottom across its horizontal centre line.",
            },
            {
                "title": "Reflection 3 — main diagonal",
                "x_square": "A",
                "explanation": "The board is reflected across the diagonal running from A to I.",
            },
            {
                "title": "Reflection 4 — anti-diagonal",
                "x_square": "I",
                "explanation": "The board is reflected across the diagonal running from C to G.",
            },
        ]

    def _render_latest_move_summary(self) -> None:
        """Explain the latest MENACE move and, when finished, its bead update."""
        latest = self._latest_symmetry_mapping_record()
        st.markdown("### What just happened?")

        if latest is None:
            st.info(
                "Choose an empty lettered square. MENACE will answer, "
                "then this box will explain its choice."
                if self._is_simple_mode()
                else
                "Make a move as X. MENACE will respond as O and this box "
                "will summarise the decision."
            )
            return

        rows = latest.get("probability_rows", [])
        selected_row = next((row for row in rows if row.get("selected")), None)
        selected_original = latest.get("selected_original_move")
        visible_square = self._move_label(selected_original)

        if self._is_simple_mode():
            if not rows or selected_row is None:
                st.success(f"MENACE placed **O** in square **{visible_square}**.")
            else:
                bead_counts = [int(row.get("beads", 0) or 0) for row in rows]
                selected_beads = int(selected_row.get("beads", 0) or 0)
                if len(set(bead_counts)) == 1:
                    reason = (
                        "Every available square had the same number of beads, "
                        "so MENACE chose between them fairly."
                    )
                elif selected_beads == max(bead_counts):
                    reason = (
                        "This square had one of the biggest groups of beads, "
                        "so MENACE liked it more."
                    )
                else:
                    reason = (
                        "Past games changed the bead groups, so some squares "
                        "were more likely than others."
                    )
                st.success(
                    f"MENACE placed **O** in square **{visible_square}**. {reason}"
                )

            update_sentence = self._simple_result_update_sentence()
            if update_sentence:
                st.info(update_sentence)
            else:
                st.caption(
                    "Squares with more beads are more likely to be chosen next time."
                )
            return

        probability = (
            float(selected_row.get("probability", 0.0))
            if selected_row is not None else None
        )
        beads = selected_row.get("beads") if selected_row is not None else None
        selected_matchbox = latest.get("selected_matchbox_move")
        selected_matchbox_label = self._move_label(selected_matchbox)
        transform = latest.get("transform_before")
        if probability is None:
            st.success(
                f"MENACE placed O in visible square {visible_square}; "
                f"the canonical matchbox move was {selected_matchbox_label}."
            )
        else:
            st.success(
                f"MENACE placed O in visible square **{visible_square}**. "
                f"This came from canonical matchbox move "
                f"**{selected_matchbox_label}**, "
                f"which currently has **{beads} beads** and a "
                f"**{probability:.1%}** probability."
            )
            st.caption(
                "The decision used the board before MENACE moved, "
                f"with canonical transform `{transform}`."
            )

    def _render_symmetry_plain_language_note(self) -> None:
        """Explain canonical state and symmetry in simple language."""
        if self._is_simple_mode():
            st.info(self.config.symmetry_explanation)
        else:
            st.info(
                self.config.symmetry_explanation
                + ' The canonical transform records the exact rotation or '
                'reflection used to map the visible board to the stored state.'
            )

    def _render_occupied_play_cell(
        self,
        index: int,
        value: str,
        last_menace_move: Optional[int],
        last_human_move: Optional[int],
    ) -> None:
        """Render one occupied Play-board square with the demo colours."""
        classes = ["live-board-cell"]
        caption = ""

        if value == "X":
            classes.append("human")
            if index == last_human_move:
                classes.append("latest-human")
                caption = "Your latest move"
        elif value == "O":
            classes.append("menace")
            if index == last_menace_move:
                classes.append("latest-menace")
                caption = "MENACE chose this square"

        star = (
            "<span class='live-board-star'>★</span>"
            if index == last_menace_move and value == "O"
            else ""
        )
        caption_html = (
            f"<span class='live-board-caption'>{caption}</span>"
            if caption
            else ""
        )
        st.markdown(
            f"<div class='{' '.join(classes)}' "
            f"aria-label='Square {self._square_label(index)}: {value}'>"
            f"{star}{value}{caption_html}</div>",
            unsafe_allow_html=True,
        )

    def _render_board(self) -> None:
        """
        Render the playable 3×3 board.

        The learner-facing board uses coloured, rounded boxes: X is blue, O is
        green, and MENACE's latest move is highlighted. Internal integer indexes
        are retained for callbacks and MENACE calculations.
        """
        board = st.session_state[SessionKey.BOARD]
        st.markdown("### Board")
        st.caption(
            "Choose one of the lettered squares."
            if self._is_simple_mode()
            else
            "Choose one of the lettered squares A–I."
        )

        latest = self._latest_symmetry_mapping_record()
        last_menace_move = (
            latest.get("selected_original_move")
            if latest is not None
            else None
        )
        last_human_move = st.session_state.get(SessionKey.LAST_HUMAN_MOVE)

        if self._is_simple_mode():
            with st.container(key="play_board"):
                for row in range(3):
                    cols = st.columns(3)
                    for col in range(3):
                        index = row * 3 + col
                        value = board.cells[index]

                        with cols[col]:
                            if value == Board.EMPTY:
                                if st.button(
                                    self._square_label(index),
                                    key=f"cell_{index}",
                                    disabled=board.is_terminal(),
                                    width="stretch",
                                    help=f"Play X in square {self._square_label(index)}",
                                ):
                                    self._handle_human_move(index)
                            else:
                                self._render_occupied_play_cell(
                                    index=index,
                                    value=value,
                                    last_menace_move=last_menace_move,
                                    last_human_move=last_human_move,
                                )
            return

        for row in range(3):
            cols = st.columns(3)
            for col in range(3):
                index = row * 3 + col
                value = board.cells[index]
                if value == Board.EMPTY:
                    label = self._square_label(index)
                elif index == last_menace_move and value == "O":
                    label = "O (last)"
                else:
                    label = value

                disabled = value != Board.EMPTY or board.is_terminal()
                if cols[col].button(
                    label,
                    key=f"cell_{index}",
                    disabled=disabled,
                    width="stretch",
                ):
                    self._handle_human_move(index)

    def _handle_human_move(self, move: int) -> None:
        """Handle a human move, then allow MENACE to respond."""
        board = st.session_state[SessionKey.BOARD]
        menace = st.session_state[SessionKey.MENACE]
        try:
            board.make_move(move, 'X')
            st.session_state[SessionKey.LAST_HUMAN_MOVE] = move
            self._append_play_move(
                player="You",
                mark="X",
                move=move,
                explanation="You selected this visible board square.",
            )
            if board.is_terminal():
                self._finish_game()
                st.rerun()
            board_before_menace = board.copy()
            menace_move, explanation = menace.choose_move_with_explanation(board)
            self._record_symmetry_mapping_for_last_decision(board_before_menace=board_before_menace, selected_original_move=menace_move)
            board.make_move(menace_move, 'O')
            latest_mapping = self._latest_symmetry_mapping_record()
            transform = (
                latest_mapping.get("transform_before", "identity")
                if latest_mapping else "identity"
            )
            canonical_move = (
                latest_mapping.get("selected_matchbox_move")
                if latest_mapping else None
            )
            self._append_play_move(
                player="MENACE",
                mark="O",
                move=menace_move,
                explanation=(
                    f"Selected from matchbox square "
                    f"{self._move_label(canonical_move)} after {transform}."
                ),
            )
            self._notify(f'MENACE chose square {self._move_label(menace_move)}.', icon='🎯')
            st.session_state[SessionKey.EXPLANATION] = explanation
            st.session_state[SessionKey.LAST_MATCHBOX_STATE] = self._extract_last_matchbox_state()
            if board.is_terminal():
                self._finish_game()
        except Exception as exc:
            st.error('Something went wrong while playing this move.')
            if not self._is_simple_mode():
                st.exception(exc)
        st.rerun()

    def _append_play_move(
        self,
        *,
        player: str,
        mark: str,
        move: int,
        explanation: str,
    ) -> None:
        """Append one learner-facing move without exposing internal indexes."""
        history = list(
            st.session_state.get(SessionKey.PLAY_MOVE_HISTORY, [])
        )
        history.append(
            {
                "Turn": len(history) + 1,
                "Player": player,
                "Mark": mark,
                "Square": self._move_label(move),
                "Explanation": explanation,
            }
        )
        st.session_state[SessionKey.PLAY_MOVE_HISTORY] = history

    def _render_move_history(self) -> None:
        """Render a compact chronological log directly below the play board."""
        st.divider()
        st.markdown("### Move history")
        history = st.session_state.get(SessionKey.PLAY_MOVE_HISTORY, [])
        if not history:
            st.caption(
                "Your moves and MENACE's replies will appear here in order."
            )
            return

        compact_rows = [
            {
                "Turn": row["Turn"],
                "Player": row["Player"],
                "Mark": row["Mark"],
                "Square": row["Square"],
            }
            for row in history
        ]
        st.dataframe(
            pd.DataFrame(compact_rows),
            width="stretch",
            hide_index=True,
        )
        with st.expander("Show explanations for each move", expanded=False):
            for row in history:
                icon = "👤" if row["Mark"] == "X" else "🤖"
                st.markdown(
                    f"**{icon} Turn {row['Turn']}: {row['Player']} "
                    f"played {row['Mark']} in square {row['Square']}**"
                )
                st.caption(row["Explanation"])

    def _record_symmetry_mapping_for_last_decision(self, board_before_menace: Board, selected_original_move: int) -> None:
        """Store the symmetry mapping used for the latest MENACE move.

        The mapping must be generated from the board state *before* MENACE
        places its mark. This avoids the common explanation error where the UI
        shows the canonical transform for the board after MENACE has already
        moved, which may be a different transform.
        """
        menace = st.session_state[SessionKey.MENACE]
        decision = menace.decisions_this_game[-1] if menace.decisions_this_game else None
        move_number = len(st.session_state.get(SessionKey.SYMMETRY_MAPPINGS, [])) + 1
        selected_matchbox_move = getattr(decision, 'canonical_move', None)
        probability_rows = self._decision_probability_rows(decision) if decision is not None else []
        try:
            explanation = self._build_symmetry_mapping_explanation(board_before_menace=board_before_menace, selected_original_move=selected_original_move, selected_matchbox_move=selected_matchbox_move)
        except Exception as exc:
            explanation = f'Symmetry mapping could not be generated: {exc}'
        before_analysis = board_before_menace.canonical_analysis()
        record = {'move_number': move_number, 'board_before': board_before_menace.to_string(), 'transform_before': before_analysis.transform_name, 'canonical_state_before': before_analysis.canonical_state, 'selected_original_move': selected_original_move, 'selected_matchbox_move': selected_matchbox_move, 'matchbox_state': getattr(decision, 'state', None), 'probability_rows': probability_rows, 'explanation': explanation}
        mappings = list(st.session_state.get(SessionKey.SYMMETRY_MAPPINGS, []))
        mappings.append(record)
        st.session_state[SessionKey.SYMMETRY_MAPPINGS] = mappings

    def _build_symmetry_mapping_explanation(self, board_before_menace: Board, selected_original_move: int, selected_matchbox_move: Optional[int]=None) -> str:
        """Build a letter-labelled explanation of one MENACE symmetry mapping."""
        canonical = board_before_menace.canonical_analysis()
        canonical_to_original = {
            canonical_move: canonical.to_original_move(canonical_move)
            for canonical_move in range(Board.BOARD_SIZE)
        }
        original_to_canonical = {
            original_move: canonical.to_canonical_move(original_move)
            for original_move in range(Board.BOARD_SIZE)
        }
        legal_original_moves = board_before_menace.available_moves()
        legal_matchbox_moves = [
            original_to_canonical[move] for move in legal_original_moves
        ]
        if selected_matchbox_move is None:
            selected_matchbox_move = original_to_canonical[selected_original_move]

        def format_letter_grid(grid: list[list[int]]) -> str:
            return "\n".join(
                " ".join(self._move_label(value) for value in row)
                for row in grid
            )

        canonical_grid = [
            list(canonical.transform.transform[0:3]),
            list(canonical.transform.transform[3:6]),
            list(canonical.transform.transform[6:9]),
        ]
        lines = [
            f'Canonical transform: {canonical.transform_name}',
            f'Canonical state: {canonical.canonical_state!r}',
            f'Original board state before MENACE: {board_before_menace.to_string()!r}',
            '',
            'Visible board squares:',
            format_letter_grid([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
            '',
            'Canonical / matchbox square grid:',
            format_letter_grid(canonical_grid),
            '',
            'Canonical matchbox square -> visible board square:',
        ]
        for matchbox_move in sorted(legal_matchbox_moves):
            original_square = canonical_to_original[matchbox_move]
            marker = (
                '  <-- selected by MENACE'
                if matchbox_move == selected_matchbox_move
                else ''
            )
            lines.append(
                f'- Matchbox square {self._move_label(matchbox_move)} -> '
                f'visible square {self._move_label(original_square)}{marker}'
            )
        lines.extend([
            '',
            f'Selected visible square: {self._move_label(selected_original_move)}',
            f'Corresponding matchbox square: {self._move_label(selected_matchbox_move)}',
        ])
        return '\n'.join(lines)

    def _render_symmetry_mapping_explanations(self) -> None:
        """Render advanced symmetry mapping details for each MENACE move."""
        mappings = st.session_state.get(SessionKey.SYMMETRY_MAPPINGS, [])
        if not mappings:
            return
        with st.expander('Advanced symmetry details for MENACE moves', expanded=False):
            st.caption("Each entry uses the board state before MENACE moved. This is the exact state used for MENACE's decision. The current board shown below may have a different canonical transform because MENACE has already placed O after this decision.")
            summary_rows = [
                {
                    'MENACE move': record['move_number'],
                    'board before MENACE': record['board_before'],
                    'transform before': record.get('transform_before'),
                    'canonical state before': record.get('canonical_state_before'),
                    'visible square': self._move_label(
                        record['selected_original_move']
                    ),
                    'matchbox square': self._move_label(
                        record['selected_matchbox_move']
                    ),
                    'matchbox state': record['matchbox_state'],
                }
                for record in mappings
            ]
            st.dataframe(pd.DataFrame(summary_rows), width="stretch")
            for record in mappings:
                st.markdown(f"#### MENACE move {record['move_number']}")
                st.code(record['explanation'])

    def _render_simple_reward_visual(self) -> None:
        """
        Show the final result and bead changes as learner-facing green bead diagrams.

        This panel is displayed only after a game finishes. It complements the
        textual explanation by showing the exact before/after bead count for
        every MENACE move that received a reward or penalty.
        """
        if not self._is_simple_mode():
            return

        result = st.session_state.get(SessionKey.LAST_GAME_RESULT)
        rows = st.session_state.get(SessionKey.LAST_BEAD_CHANGES, [])
        if not result or not rows:
            st.caption(
                "The green bead-change diagrams will appear here when the game ends."
            )
            return

        outcome_title = {
            "win": "🏆 MENACE won and received a reward",
            "loss": "🎉 You won, so MENACE reduced some beads",
            "draw": "🤝 The game was a draw, so MENACE made a smaller update",
        }.get(result, "MENACE updated its beads")

        outcome_explanation = {
            "win": (
                "The selected moves helped MENACE win. Each one receives extra "
                "green beads, making it more likely in the same situation next time."
            ),
            "loss": (
                "The selected moves did not lead to a win. MENACE removes some "
                "beads where possible, making those choices less likely next time."
            ),
            "draw": (
                "A draw gives a smaller reward. MENACE adds fewer green beads than "
                "it would receive after a win."
            ),
        }.get(result, "MENACE changed the bead groups for the moves it used.")

        st.markdown(f"#### {outcome_title}")
        st.write(outcome_explanation)

        valid_rows = [
            row
            for row in rows
            if row.get("original_move") is not None
            and row.get("before") is not None
            and row.get("after") is not None
        ]
        if not valid_rows:
            st.caption("No complete bead-change records were available.")
            return

        total_change = sum(int(row.get("change") or 0) for row in valid_rows)
        summary_cols = st.columns(3)
        summary_cols[0].metric("MENACE moves updated", len(valid_rows))
        summary_cols[1].metric(
            "Total bead change",
            f"{total_change:+d}",
        )
        summary_cols[2].metric(
            "Result",
            {"win": "Win", "loss": "Loss", "draw": "Draw"}.get(
                result,
                result.title(),
            ),
        )

        cards: list[str] = []
        for row in valid_rows:
            square = self._move_label(int(row["original_move"]))
            before = int(row["before"])
            after = int(row["after"])
            change = int(row.get("change") or (after - before))

            before_beads = "".join(
                "<span class='reward-bead before-bead'></span>"
                for _ in range(before)
            )
            after_beads = "".join(
                "<span class='reward-bead after-bead'></span>"
                for _ in range(after)
            )

            if change > 0:
                change_text = f"+{change} beads"
                change_class = "positive"
                learning_text = "This move is now more likely."
            elif change < 0:
                change_text = f"{change} beads"
                change_class = "negative"
                learning_text = "This move is now less likely."
            else:
                change_text = "No change"
                change_class = "neutral"
                learning_text = "This move keeps the same bead count."

            cards.append(
                "<div class='reward-change-card'>"
                f"<div class='reward-square-title'>Square {square}</div>"
                "<div class='reward-change-row'>"
                "<div class='reward-side'>"
                "<div class='reward-label'>Before</div>"
                f"<div class='reward-beads'>{before_beads}</div>"
                f"<div class='reward-count'>{before} beads</div>"
                "</div>"
                "<div class='reward-arrow'>→</div>"
                "<div class='reward-side'>"
                "<div class='reward-label'>After</div>"
                f"<div class='reward-beads'>{after_beads}</div>"
                f"<div class='reward-count'>{after} beads</div>"
                "</div>"
                "</div>"
                f"<div class='reward-delta {change_class}'>{change_text}</div>"
                f"<div class='reward-learning-text'>{learning_text}</div>"
                "</div>"
            )

        st.markdown(
            "<div class='reward-change-grid'>"
            + "".join(cards)
            + "</div>",
            unsafe_allow_html=True,
        )

        if result == "win":
            st.success(
                "The larger green bead groups make these winning choices easier "
                "for MENACE to pick when it sees the same board situations again."
            )
        elif result == "loss":
            st.info(
                "The smaller green bead groups make these choices less likely, "
                "although MENACE always keeps at least its minimum bead count."
            )
        else:
            st.info(
                "The draw reward is smaller than the win reward, so the increase "
                "is more cautious."
            )

    def _learn_from_interactive_game(self, result: str) -> None:
        """
        Reinforce beads after a human game without changing practice counters.

        Automated practice counters and human-play outcomes are kept separate:

        - bead learning still happens normally;
        - automated-practice counters remain unchanged;
        - human-game outcomes are tracked independently in session state.
        """
        menace = st.session_state[SessionKey.MENACE]

        practice_games = menace.games_played
        practice_wins = menace.wins
        practice_losses = menace.losses
        practice_draws = menace.draws

        # Apply the normal MENACE reinforcement update to its matchboxes.
        menace.learn_from_result(result)

        # Restore counters reserved for automated practice evidence.
        menace.games_played = practice_games
        menace.wins = practice_wins
        menace.losses = practice_losses
        menace.draws = practice_draws

        # Retain separate, non-visual session statistics for interactive play.
        st.session_state[SessionKey.INTERACTIVE_GAMES] += 1
        if result == "win":
            st.session_state[SessionKey.INTERACTIVE_WINS] += 1
        elif result == "loss":
            st.session_state[SessionKey.INTERACTIVE_LOSSES] += 1
        else:
            st.session_state[SessionKey.INTERACTIVE_DRAWS] += 1

    def _finish_game(self) -> None:
        """Apply reinforcement and synchronise every learner-facing explanation."""
        board = st.session_state[SessionKey.BOARD]
        menace = st.session_state[SessionKey.MENACE]
        winner = board.winner()
        start_memory = dict(
            st.session_state.get(
                SessionKey.GAME_START_MEMORY,
                self._learning_memory_snapshot(menace),
            )
        )
        before_rows = self._bead_rows_from_current_decisions(menace)
        before_learning = self._learning_memory_snapshot(menace)
        used_states = list(dict.fromkeys(row["state"] for row in before_rows))
        box_totals_before = [
            menace.matchboxes[state].total_beads()
            for state in used_states
            if state in menace.matchboxes
        ]

        if winner == "O":
            result = "win"
            message = (
                "MENACE won! Helpful choices gain more beads."
                if self._is_simple_mode() else
                "MENACE wins and is rewarded."
            )
        elif winner == "X":
            result = "loss"
            message = (
                "You won! MENACE may lose a bead from choices that did not help."
                if self._is_simple_mode() else
                "Human wins. MENACE is punished."
            )
        else:
            result = "draw"
            message = (
                "It is a draw. MENACE makes a smaller bead update."
                if self._is_simple_mode() else
                "Draw game. MENACE receives a small draw reward."
            )

        # ``result`` is intentionally stored from MENACE's perspective because
        # it drives reinforcement: win rewards MENACE, loss penalises MENACE.
        # The learner-facing label names the winner explicitly so that
        # "Human game (win)" cannot be mistaken for a human victory.
        display_result = {
            "win": "MENACE win",
            "loss": "Human win",
            "draw": "Draw",
        }[result]

        self._learn_from_interactive_game(result)
        after_rows = self._bead_rows_after_learning(menace, before_rows)
        after_learning = self._learning_memory_snapshot(menace)
        box_totals_after = [
            menace.matchboxes[state].total_beads()
            for state in used_states
            if state in menace.matchboxes
        ]
        events = st.session_state.setdefault(SessionKey.LEARNING_EVENTS, [])
        events.append(
            {
                "kind": "human",
                "result": result,
                "label": (
                    f"Human game "
                    f"{int(st.session_state[SessionKey.INTERACTIVE_GAMES]):,} "
                    f"({display_result})"
                ),
                "before": int(start_memory.get("beads", 0)),
                "created": (
                    int(before_learning["beads"])
                    - int(start_memory.get("beads", 0))
                ),
                "reinforcement": (
                    int(after_learning["beads"])
                    - int(before_learning["beads"])
                ),
                "after": int(after_learning["beads"]),
                "box_totals_before": box_totals_before,
                "box_totals_after": box_totals_after,
            }
        )
        st.session_state[SessionKey.LAST_GAME_RESULT] = result
        st.session_state[SessionKey.LAST_BEAD_CHANGES] = after_rows
        st.session_state[SessionKey.LAST_HUMAN_MEMORY] = {
            "summary": self._learning_memory_snapshot(menace),
            "practice_games": int(menace.games_played),
            "human_games": int(st.session_state[SessionKey.INTERACTIVE_GAMES]),
            "matchboxes": {
                str(state): {int(move): int(count) for move, count in box.beads.items()}
                for state, box in menace.matchboxes.items()
            },
        }
        st.session_state[SessionKey.TRAINING_SINCE_HUMAN_GAME] = None
        self._refresh_decision_records_after_learning()
        st.session_state[SessionKey.MESSAGE] = message
        st.session_state[SessionKey.REWARD_FEEDBACK] = self._build_reward_feedback(
            result=result,
            before_after_rows=after_rows,
        )

        # Human play changes MENACE's bead memory too. Save immediately at the
        # end of each completed game so "Continue previous learning" remains
        # available after the browser tab is closed and reopened.
        try:
            self._save_learning_bundle(show_message=False)
        except Exception as exc:
            st.session_state[SessionKey.PERSISTENCE_NOTICE] = (
                "MENACE learned from this game, but the saved learning journey "
                f"could not be updated: {exc}"
            )

        self._notify(message, icon="🏁")

    def _extract_last_matchbox_state(self) -> Optional[str]:
        """Return the state key for the latest MENACE decision."""
        menace = st.session_state[SessionKey.MENACE]
        if not menace.decisions_this_game:
            return None
        return menace.decisions_this_game[-1].state

    def _bead_rows_from_current_decisions(self, menace: MENACEPlayer) -> list[dict[str, Any]]:
        """Capture bead counts before reinforcement."""
        rows: list[dict[str, Any]] = []
        for decision in menace.decisions_this_game:
            matchbox = menace.matchboxes.get(decision.state)
            if matchbox is None:
                continue
            rows.append({'state': decision.state, 'canonical_move': decision.canonical_move, 'original_move': decision.move, 'before': matchbox.beads.get(decision.canonical_move)})
        return rows

    def _bead_rows_after_learning(self, menace: MENACEPlayer, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add bead counts after reinforcement."""
        for row in rows:
            matchbox = menace.matchboxes.get(str(row['state']))
            canonical_move = int(row['canonical_move'])
            if matchbox is not None:
                row['after'] = matchbox.beads.get(canonical_move)
                row['change'] = None if row.get('before') is None or row.get('after') is None else int(row['after']) - int(row['before'])
            else:
                row['after'] = None
                row['change'] = None
        return rows

    def _build_reward_feedback(self, result: str, before_after_rows: list[dict[str, Any]]) -> str:
        """Build readable feedback showing how bead counts changed."""
        if not before_after_rows:
            return 'The game ended before MENACE made a choice, so there were no beads to change.' if self._is_simple_mode() else f'Game result for MENACE: {result}. No MENACE decision was recorded for reinforcement.'
        if self._is_simple_mode():
            intro = {'win':'MENACE won, so it added beads to the choices that helped.','loss':'MENACE lost, so it removed a bead from the choices it used.','draw':'The game was a draw, so MENACE added a small reward.'}.get(result, 'MENACE updated its beads.')
            lines = [intro, '', 'Bead changes:']
            for row in before_after_rows:
                lines.append(f"- Square {self._move_label(int(row['original_move']))}: {row['before']} bead(s) → {row['after']} bead(s).")
            lines.append('\nNext time, moves with more beads will be more likely to be chosen.')
            return '\n'.join(lines)
        lines = [f'Game result for MENACE: {result}.', '', 'MENACE updates the bead count for the moves it selected during this game:']
        for row in before_after_rows:
            original_label = self._move_label(int(row['original_move']))
            canonical_label = self._move_label(int(row['canonical_move']))
            lines.append(
                f"- State {row['state']!r}, visible square {original_label}, "
                f"canonical matchbox square {canonical_label}: "
                f"{row['before']} beads -> {row['after']} beads "
                f"(change: {row['change']})."
            )
        lines.append('\nThis is the key learning mechanism: future choices become more or less likely because the bead counts changed.')
        return '\n'.join(lines)

    def _decision_probability_rows(self, decision: Any) -> list[dict[str, Any]]:
        """Return rows linking matchbox moves to visible board squares for one decision.

        This is based on the board state before MENACE moved, so it matches the
        decision explanation exactly. It does not use the board after MENACE has
        placed O.
        """
        if hasattr(decision, 'matchbox_probability_rows'):
            rows = list(decision.matchbox_probability_rows())
            return [dict(row) for row in rows]
        original_to_canonical = getattr(decision, 'original_to_canonical', {})
        canonical_to_original = {canonical_move: original_move for original_move, canonical_move in original_to_canonical.items()}
        bead_snapshot = getattr(decision, 'bead_snapshot', {})
        probabilities = getattr(decision, 'probabilities', {})
        selected = getattr(decision, 'canonical_move', None)
        rows: list[dict[str, Any]] = []
        for matchbox_move in sorted(bead_snapshot):
            rows.append({'matchbox_move': int(matchbox_move), 'original_board_square': canonical_to_original.get(matchbox_move), 'beads': int(bead_snapshot[matchbox_move]), 'probability': float(probabilities.get(matchbox_move, 0.0)), 'selected': matchbox_move == selected})
        return rows

    def _format_decision_matchbox_explanation(self, record: dict[str, Any]) -> str:
        """Build a letter-labelled matchbox explanation for the latest decision."""
        rows = record.get('probability_rows', [])
        selected_original = record.get('selected_original_move')
        selected_matchbox = record.get('selected_matchbox_move')
        lines = [
            'This is the matchbox MENACE used before placing O.',
            f"Board before MENACE: {record.get('board_before')!r}",
            f"Canonical transform: {record.get('transform_before')}",
            f"Matchbox state: {record.get('matchbox_state')!r}",
            f"Selected visible square: {self._move_label(selected_original)}",
            f"Selected matchbox square: {self._move_label(selected_matchbox)}",
            '',
            'Move probabilities at decision time:',
        ]
        for row in rows:
            marker = '  <-- selected' if row.get('selected') else ''
            lines.append(
                f"- Matchbox square "
                f"{self._move_label(row.get('matchbox_move'))} -> visible square "
                f"{self._move_label(row.get('original_board_square'))}: "
                f"{row.get('beads')} beads "
                f"({float(row.get('probability', 0.0)):.1%} chance){marker}"
            )
        return '\n'.join(lines)

    def _temporary_matchbox_from_record(self, record: dict[str, Any]) -> Optional[Matchbox]:
        """Build a temporary Matchbox for charting the latest decision snapshot."""
        rows = record.get('probability_rows', [])
        state = record.get('matchbox_state')
        if not state or not rows:
            return None
        beads = {int(row['matchbox_move']): int(row['beads']) for row in rows}
        try:
            return Matchbox(state=str(state), beads=beads)
        except Exception:
            return None

    def _render_current_matchbox_visualisation(self) -> None:
        """Show the latest matchbox using current, post-learning bead values."""
        latest = self._latest_symmetry_mapping_record()
        st.markdown(
            "### The matchbox MENACE just used"
            if self._is_simple_mode()
            else "### Matchbox Used for Latest MENACE Decision"
        )

        if latest is None:
            st.info(
                "You’ll see green bead groups here once MENACE has moved."
                if self._is_simple_mode()
                else
                "After MENACE responds, this section will show the exact "
                "matchbox used for the decision."
            )
            return

        rows = latest.get("probability_rows", [])
        box = self._temporary_matchbox_from_record(latest)
        if self._is_simple_mode():
            if box is None:
                st.info("The bead box could not be displayed for this move.")
                return
            move_mapping = {
                int(row["matchbox_move"]): int(row["original_board_square"])
                for row in rows
                if row.get("original_board_square") is not None
            }
            selected_move = latest.get("selected_matchbox_move")
            html = self.visualiser.educational_matchbox_html(
                box,
                move_to_board_square=move_mapping,
                selected_move=(
                    int(selected_move) if selected_move is not None else None
                ),
                square_labels=self.config.square_labels,
                title="Each bead group points to one lettered square",
                show_probabilities=False,
            )
            st.markdown(html, unsafe_allow_html=True)
            st.caption(
                "A bigger bead group means MENACE likes that square more. "
                "The green border and ‘MENACE chose this square’ label show its choice."
            )
            if st.session_state.get(SessionKey.LAST_GAME_RESULT):
                st.caption(
                    "This is the final matchbox MENACE used. "
                    "The reward section below shows every MENACE move from the whole game."
                )
            return

        with st.expander(
            "Show raw log for this decision",
            expanded=False,
        ):
            st.code(self._format_decision_matchbox_explanation(latest))

        st.markdown("### Exportable evidence for this decision")
        st.caption(
            "The table and plots below are the structured evidence used for "
            "technical inspection, comparison and report-ready figures."
        )

        if rows:
            df = pd.DataFrame(rows).copy()
            if "matchbox_move" in df.columns:
                df["matchbox_move"] = df["matchbox_move"].map(
                    lambda value: self._move_label(int(value))
                )
            if "original_board_square" in df.columns:
                df["original_board_square"] = df[
                    "original_board_square"
                ].map(
                    lambda value: self._move_label(int(value))
                    if pd.notna(value) else "?"
                )
            if "probability" in df.columns:
                df["probability"] = df["probability"].map(
                    lambda value: f"{float(value):.1%}"
                )
            df = df.rename(
                columns={
                    "matchbox_move": "Matchbox square",
                    "original_board_square": "Visible square",
                    "beads": "Beads",
                    "probability": "Probability",
                    "selected": "Selected",
                }
            )
            st.dataframe(df, width="stretch", hide_index=True)
        if box is not None:
            try:
                bead_figure = self._letterise_matchbox_figure(
                    self.visualiser.matchbox_beads(box)
                )
                probability_figure = self._letterise_matchbox_figure(
                    self.visualiser.matchbox_probabilities(box)
                )
                self._render_exportable_figure(
                    bead_figure,
                    name="play_matchbox_beads",
                    chart_key="play_matchbox_beads",
                )
                self._render_exportable_figure(
                    probability_figure,
                    name="play_matchbox_probabilities",
                    chart_key="play_matchbox_probabilities",
                )
            except Exception as exc:
                st.warning(f"Could not draw latest decision charts: {exc}")

    def _current_matchbox(self) -> Optional[Matchbox]:
        """
        Return the matchbox for the current board.

        The call to explain_state deliberately creates the matchbox if needed,
        which is useful for educational inspection in the interface.
        """
        board = st.session_state[SessionKey.BOARD]
        menace = st.session_state[SessionKey.MENACE]
        if board.is_terminal():
            return None
        state = menace._state_key(board)
        menace.explain_state(board)
        return menace.matchboxes.get(state)
