"""Streamlit helpers for the MENACE learner demonstration."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import streamlit as st

from ui.session import SessionKey

class DemoMixin:
    """Provide the interactive learner demonstration used by the interface."""

    def _demo_frames(
        self,
        outcome: str = "MENACE wins",
    ) -> list[dict[str, Any]]:
        """
        Return one complete, fixed MENACE learning story for the selected ending.

        The learner can view three valid reinforcement outcomes:
            - MENACE wins: selected moves gain the full win reward.
            - Draw: selected moves gain the smaller draw reward.
            - MENACE loses: selected moves lose one bead where possible.
        """
        common_frames = [
            {
                "board": [" "] * 9,
                "title": "1. You choose a square",
                "text": (
                    "You play as X. Choose any empty square labelled A to I, "
                    "just like in the real game."
                ),
                "last": None,
                "matchbox_title": "Before MENACE needs a matchbox",
                "matchbox_note": (
                    "MENACE waits until you make the first move. "
                    "Then it looks for a matchbox that matches the new board."
                ),
                "beads": {},
            },
            {
                "board": ["X", " ", " ", " ", " ", " ", " ", " ", " "],
                "title": "2. MENACE recognises the pattern",
                "text": (
                    "You placed X in square A. With symmetry switched on, MENACE "
                    "checks turned and reflected versions, converts equivalent "
                    "patterns to one canonical state, and opens the shared matchbox."
                ),
                "last": None,
                "matchbox_title": "One matchbox for equivalent patterns",
                "matchbox_note": (
                    "This avoids keeping separate boxes for the same pattern in "
                    "different directions. Every bead group points to an available "
                    "lettered square."
                ),
                "beads": {
                    "B": 3, "C": 3, "D": 3, "E": 3,
                    "F": 3, "G": 3, "H": 3, "I": 3,
                },
            },
            {
                "board": ["X", " ", " ", " ", "O", " ", " ", " ", " "],
                "title": "3. MENACE picks a bead",
                "text": (
                    "MENACE picked one bead from the group for square E, "
                    "so it placed O in E. The chosen square is highlighted."
                ),
                "last": 4,
                "matchbox_title": "The bead MENACE picked",
                "matchbox_note": (
                    "All groups still have 3 beads, so every available move "
                    "had the same chance. MENACE happened to pick E."
                ),
                "beads": {
                    "B": 3, "C": 3, "D": 3, "E": 3,
                    "F": 3, "G": 3, "H": 3, "I": 3,
                },
                "selected": "E",
            },
        ]

        if outcome == "Draw":
            return common_frames + [
                {
                    "board": ["X", "O", "X", " ", "O", " ", " ", "X", " "],
                    "title": "4. The game continues",
                    "text": (
                        "Both players keep blocking possible winning lines. "
                        "MENACE remembers the beads it selected."
                    ),
                    "last": 1,
                    "matchbox_title": "Another matchbox",
                    "matchbox_note": (
                        "MENACE used another matchbox and selected square B."
                    ),
                    "beads": {"B": 3, "D": 3, "F": 3, "G": 3, "I": 3},
                    "selected": "B",
                },
                {
                    "board": ["X", "O", "X", "X", "O", "O", "O", "X", "X"],
                    "title": "5. The game ends in a draw",
                    "text": (
                        "Every square is filled and neither player has three in "
                        "a row. The game is a draw."
                    ),
                    "last": 6,
                    "matchbox_title": "Small reward after the draw",
                    "matchbox_note": (
                        "Each MENACE move receives 1 extra bead. A draw is useful, "
                        "but the reward is smaller than for a win."
                    ),
                    "beads": {"E": 4, "B": 4, "G": 4},
                    "selected": "G",
                    "result": {
                        "outcome": "Draw",
                        "headline": "🤝 The game ends in a draw.",
                        "moves_rewarded": ["E", "B", "G"],
                        "change": 1,
                        "before": 3,
                        "after": 4,
                        "reason": (
                            "The selected moves helped MENACE avoid losing. "
                            "Each move receives the smaller draw reward."
                        ),
                        "future": (
                            "These moves become a little more likely next time "
                            "because their bead groups are slightly larger."
                        ),
                    },
                },
            ]

        if outcome == "MENACE loses":
            return common_frames + [
                {
                    "board": ["X", " ", " ", "X", "O", " ", " ", " ", "O"],
                    "title": "4. The game continues",
                    "text": (
                        "You placed X in D and MENACE answered in I. "
                        "MENACE remembers both beads it selected."
                    ),
                    "last": 8,
                    "matchbox_title": "Another matchbox",
                    "matchbox_note": (
                        "MENACE used another matchbox and selected square I."
                    ),
                    "beads": {"B": 3, "C": 3, "F": 3, "G": 3, "H": 3, "I": 3},
                    "selected": "I",
                },
                {
                    "board": ["X", " ", " ", "X", "O", " ", "X", " ", "O"],
                    "title": "5. You win and MENACE learns",
                    "text": (
                        "You placed X in G and completed the line A–D–G. "
                        "MENACE has lost this game."
                    ),
                    "last": 6,
                    "matchbox_title": "Penalty after the loss",
                    "matchbox_note": (
                        "Each selected MENACE move loses 1 bead where possible. "
                        "Those moves will be less likely in a future game."
                    ),
                    "beads": {"E": 2, "I": 2},
                    "selected": "I",
                    "result": {
                        "outcome": "MENACE loses",
                        "headline": "❌ You win with the line A–D–G.",
                        "moves_rewarded": ["E", "I"],
                        "change": -1,
                        "before": 3,
                        "after": 2,
                        "reason": (
                            "The selected moves did not prevent the loss. "
                            "Each selected move loses 1 bead."
                        ),
                        "future": (
                            "These moves become less likely next time because "
                            "their bead groups are smaller."
                        ),
                    },
                },
            ]

        return common_frames + [
            {
                "board": ["X", "X", "O", " ", "O", " ", " ", " ", " "],
                "title": "4. The game continues",
                "text": (
                    "You chose B and MENACE answered in C. "
                    "MENACE remembers every bead it picked during this game."
                ),
                "last": 2,
                "matchbox_title": "Another matchbox",
                "matchbox_note": (
                    "This board looks different, so MENACE uses another "
                    "matchbox. It picked the bead for square C."
                ),
                "beads": {"C": 3, "D": 3, "F": 3, "G": 3, "H": 3, "I": 3},
                "selected": "C",
            },
            {
                "board": ["X", "X", "O", "X", "O", " ", "O", " ", " "],
                "title": "5. MENACE wins and learns",
                "text": (
                    "MENACE placed O in G and completed a winning line: "
                    "C–E–G. The game is now finished."
                ),
                "last": 6,
                "matchbox_title": "Reward after the win",
                "matchbox_note": (
                    "Each MENACE move that helped this win receives 3 extra beads. "
                    "Those moves will be easier to pick in a future game."
                ),
                "beads": {"E": 6, "C": 6, "G": 6},
                "selected": "G",
                "result": {
                    "outcome": "MENACE wins",
                    "headline": "🏆 MENACE wins with the line C–E–G.",
                    "moves_rewarded": ["E", "C", "G"],
                    "change": 3,
                    "before": 3,
                    "after": 6,
                    "reason": (
                        "The selected moves helped MENACE win. "
                        "Each move receives 3 extra beads."
                    ),
                    "future": (
                        "These moves become more likely next time because "
                        "their bead groups are larger."
                    ),
                },
            },
        ]

    def _render_demo_board(
        self,
        cells: list[str],
        last_move: Optional[int],
    ) -> None:
        """
        Draw the demonstration board using the same A–I labels as the Play page.

        Empty squares show their learner-facing letter. Played squares show X or O.
        The latest MENACE move is highlighted so the demonstration remains visually
        consistent with the interactive game.
        """
        if len(cells) != 9:
            raise ValueError("Demo board must contain exactly nine cells.")

        parts = ["<div class='demo-board'>"]
        for index, value in enumerate(cells):
            classes = ["demo-cell"]
            if value == "X":
                classes.append("human")
            if value == "O":
                classes.append("menace")
            if index == last_move:
                classes.append("menace-last")

            symbol = value if value.strip() else self._square_label(index)
            accessible_label = (
                f"Square {self._square_label(index)}: {value}"
                if value.strip()
                else f"Empty square {self._square_label(index)}"
            )
            parts.append(
                f"<div class='{' '.join(classes)}' "
                f"aria-label='{accessible_label}'>{symbol}</div>"
            )

        parts.append("</div>")
        st.markdown("".join(parts), unsafe_allow_html=True)

    def _render_demo_matchbox(
        self,
        title: str,
        note: str,
        beads: dict[str, int],
        selected: Optional[str] = None,
    ) -> None:
        """Draw a learner-facing matchbox with one visible circle per bead."""
        cards: list[str] = []
        for square, count in beads.items():
            selected_class = " selected" if square == selected else ""
            selected_text = (
                "<div class='selected-label'>MENACE chose this square</div>"
                if square == selected
                else ""
            )
            circles = "".join(
                "<span class='reward-bead after-bead' aria-hidden='true'></span>"
                for _ in range(int(count))
            )
            cards.append(
                f"<div class='move-card{selected_class}'>"
                f"<div class='square-title'>Square {square}</div>"
                f"<div class='bead-row' aria-label='{count} beads'>{circles}</div>"
                f"<div class='move-caption'>{count} bead(s)</div>"
                f"{selected_text}</div>"
            )

        content = (
            "".join(cards)
            if cards
            else (
                "<div class='demo-empty-matchbox'>"
                "No matchbox is needed until the first X is placed."
                "</div>"
            )
        )
        st.markdown(
            "<div class='menace-matchbox'>"
            f"<div class='matchbox-title'>{title}</div>"
            f"<div class='matchbox-help'>{note}</div>"
            f"<div class='matchbox-grid'>{content}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    def _render_demo_result(self, result: dict[str, Any]) -> None:
        """Explain a win, draw, or loss and show the exact bead change."""
        affected = self._friendly_join(
            [str(square) for square in result["moves_rewarded"]]
        )

        outcome = str(result["outcome"])
        if outcome == "MENACE wins":
            st.success(result["headline"])
        elif outcome == "Draw":
            st.info(result["headline"])
        else:
            st.warning(result["headline"])

        st.markdown("#### What did MENACE learn?")
        st.write(
            f"The moves in squares **{affected}** were the MENACE choices "
            f"remembered from this game. {result['reason']}"
        )

        columns = st.columns(len(result["moves_rewarded"]))
        for column, square in zip(columns, result["moves_rewarded"]):
            with column:
                st.markdown(f"**Square {square}**")
                before_class = "before-bead"
                after_class = (
                    "after-bead"
                    if result["change"] > 0
                    else "learning-bead-after-loss"
                )
                st.markdown(
                    f"<div class='demo-bead-change'>"
                    f"<div><strong>Before</strong><br>"
                    f"{''.join(f'<span class=\"reward-bead {before_class}\"></span>' for _ in range(result['before']))}"
                    f"<br>{result['before']} beads</div>"
                    f"<div class='demo-arrow'>→</div>"
                    f"<div><strong>After</strong><br>"
                    f"{''.join(f'<span class=\"reward-bead {after_class}\"></span>' for _ in range(result['after']))}"
                    f"<br>{result['after']} beads</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.info(result["future"])

    def _render_watch_demo(self) -> None:
        """Render a complete, user-controlled demonstration of one learning cycle."""
        if not self._is_simple_mode():
            return
        if not st.session_state.get(SessionKey.SHOW_DEMO, False):
            return

        outcome_options = ["MENACE wins", "Draw", "MENACE loses"]
        selected_outcome = st.session_state.get(
            SessionKey.DEMO_OUTCOME,
            outcome_options[0],
        )
        if selected_outcome not in outcome_options:
            selected_outcome = outcome_options[0]
            st.session_state[SessionKey.DEMO_OUTCOME] = selected_outcome

        frames = self._demo_frames(selected_outcome)
        step = max(
            0,
            min(
                int(st.session_state.get(SessionKey.DEMO_STEP, 0)),
                len(frames) - 1,
            ),
        )
        frame = frames[step]

        with st.container(border=True):
            st.markdown("### ▶ Watch how MENACE plays and learns")
            st.caption(
                "Choose an ending to see how a win, draw, or loss changes "
                "MENACE's beads."
            )
            st.info(
                "This walkthrough uses lettered squares A–I. Symmetry is enabled: "
                "equivalent turned or reflected boards share one canonical state "
                "and one matchbox."
            )
            selected_outcome = st.radio(
                "Choose the demo ending",
                outcome_options,
                index=outcome_options.index(selected_outcome),
                horizontal=True,
                key=SessionKey.DEMO_OUTCOME,
            )
            # If the selected story changed, restart at step 1.
            previous_story = st.session_state.get("_last_demo_outcome")
            if previous_story != selected_outcome:
                st.session_state["_last_demo_outcome"] = selected_outcome
                st.session_state[SessionKey.DEMO_STEP] = 0
                step = 0
                frames = self._demo_frames(selected_outcome)

            st.progress(
                (step + 1) / len(frames),
                text=f"Step {step + 1} of {len(frames)}",
            )

            board_col, explanation_col = st.columns([1, 1.25])
            with board_col:
                st.markdown(f"#### {frame['title']}")
                self._render_demo_board(frame["board"], frame["last"])

            with explanation_col:
                st.markdown("#### What just happened?")
                st.info(frame["text"])
                self._render_demo_matchbox(
                    title=frame["matchbox_title"],
                    note=frame["matchbox_note"],
                    beads=frame.get("beads", {}),
                    selected=frame.get("selected"),
                )

            if frame.get("result"):
                st.divider()
                self._render_demo_result(frame["result"])

            if step == len(frames) - 1:
                st.success(
                    "Demo complete — you have seen one full MENACE learning cycle."
                )
                st.info(
                    "Now switch to Play Against MENACE below to see the same "
                    "canonical states, symmetry mappings and probabilities operate "
                    "in a live game."
                )

            left, middle, right = st.columns(3)
            if left.button(
                "Previous",
                disabled=step == 0,
                key="demo_previous",
            ):
                st.session_state[SessionKey.DEMO_STEP] = step - 1
                st.rerun()

            if middle.button(
                "Next step" if step < len(frames) - 1 else "Play now",
                key="demo_next",
                type="primary",
            ):
                if step < len(frames) - 1:
                    st.session_state[SessionKey.DEMO_STEP] = step + 1
                else:
                    st.session_state[SessionKey.SHOW_DEMO] = False
                    st.session_state[SessionKey.SHOW_WELCOME] = False
                st.rerun()

            if right.button("Close demo", key="demo_close"):
                st.session_state[SessionKey.SHOW_DEMO] = False
                st.rerun()





