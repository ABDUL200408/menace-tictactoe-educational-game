"""Streamlit page explaining how MENACE learns."""

from __future__ import annotations


import streamlit as st

from ui.session import SessionKey

class LearningPageMixin:
    """Render the MENACE learning explanation and related navigation controls."""

    def _open_play_from_learning(self) -> None:
        """Open the Play page from the learning explanation."""
        labels = (
            self.config.simple_page_labels
            if self._is_simple_mode()
            else self.config.technical_page_labels
        )
        self._set_active_page(labels[0])

    def _open_practice_from_learning(self) -> None:
        """Open the practice page from the learning explanation."""
        labels = (
            self.config.simple_page_labels
            if self._is_simple_mode()
            else self.config.technical_page_labels
        )
        self._set_active_page(labels[1])

    @staticmethod
    def _learning_bead_row(count: int, css_class: str = "") -> str:
        """Return HTML for a visible row of bead circles in the learning explanation."""
        return "".join(
            f"<span class='learning-bead {css_class}'></span>"
            for _ in range(max(0, int(count)))
        )

    def _render_learning_page(self) -> None:
        """Explain MENACE learning visually, with optional detailed technical information."""
        menace = st.session_state[SessionKey.MENACE]
        board = st.session_state[SessionKey.BOARD]

        if self._is_simple_mode():
            st.subheader("How MENACE Learns")
            st.write(
                "MENACE learns by playing many games, remembering its choices, "
                "and changing the number of beads inside its matchboxes."
            )

            st.markdown(
                """
                <div class="learning-story-grid">
                    <div class="learning-story-card">
                        <div class="learning-story-number">1</div>
                        <div class="learning-story-icon">🙂</div>
                        <div class="learning-story-title">Starts as a beginner</div>
                        <div class="learning-story-text">
                            At first, MENACE does not know which moves are best.
                        </div>
                    </div>
                    <div class="learning-story-card">
                        <div class="learning-story-number">2</div>
                        <div class="learning-story-icon">📦</div>
                        <div class="learning-story-title">Finds a matchbox</div>
                        <div class="learning-story-text">
                            Each canonical board pattern has one shared matchbox.
                        </div>
                    </div>
                    <div class="learning-story-card">
                        <div class="learning-story-number">3</div>
                        <div class="learning-story-icon">🟢</div>
                        <div class="learning-story-title">Picks one bead</div>
                        <div class="learning-story-text">
                            Each bead points to a possible square. MENACE picks one by chance.
                        </div>
                    </div>
                    <div class="learning-story-card">
                        <div class="learning-story-number">4</div>
                        <div class="learning-story-icon">⭐</div>
                        <div class="learning-story-title">Learns from the result</div>
                        <div class="learning-story-text">
                            Helpful moves gain beads. Poor moves may lose a bead.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("### A simple example")
            before = self._learning_bead_row(3, "learning-bead-before")
            after_win = self._learning_bead_row(6, "learning-bead-after")
            after_draw = self._learning_bead_row(4, "learning-bead-after-draw")
            after_loss = self._learning_bead_row(2, "learning-bead-after-loss")

            st.markdown(
                "<div class='learning-example-grid'>"
                "<div class='learning-example-card learning-win-card'>"
                "<div class='learning-example-heading'>🏆 After a win</div>"
                "<div class='learning-example-text'>"
                "A helpful move receives more beads."
                "</div>"
                "<div class='learning-bead-change-row'>"
                "<div><strong>Before</strong>"
                f"<div class='learning-bead-line'>{before}</div>"
                "<span>3 beads</span></div>"
                "<div class='learning-change-arrow'>→</div>"
                "<div><strong>After</strong>"
                f"<div class='learning-bead-line'>{after_win}</div>"
                "<span>6 beads</span></div>"
                "</div>"
                "<div class='learning-example-result'>"
                "That move becomes more likely next time."
                "</div></div>"
                "<div class='learning-example-card learning-draw-card'>"
                "<div class='learning-example-heading'>🤝 After a draw</div>"
                "<div class='learning-example-text'>"
                "A draw gives a smaller reward than a win."
                "</div>"
                "<div class='learning-bead-change-row'>"
                "<div><strong>Before</strong>"
                f"<div class='learning-bead-line'>{before}</div>"
                "<span>3 beads</span></div>"
                "<div class='learning-change-arrow'>→</div>"
                "<div><strong>After</strong>"
                f"<div class='learning-bead-line'>{after_draw}</div>"
                "<span>4 beads</span></div>"
                "</div>"
                "<div class='learning-example-result'>"
                "The move becomes a little more likely next time."
                "</div></div>"
                "<div class='learning-example-card learning-loss-card'>"
                "<div class='learning-example-heading'>❌ After a loss</div>"
                "<div class='learning-example-text'>"
                "A move that did not help may lose one bead."
                "</div>"
                "<div class='learning-bead-change-row'>"
                "<div><strong>Before</strong>"
                f"<div class='learning-bead-line'>{before}</div>"
                "<span>3 beads</span></div>"
                "<div class='learning-change-arrow'>→</div>"
                "<div><strong>After</strong>"
                f"<div class='learning-bead-line'>{after_loss}</div>"
                "<span>2 beads</span></div>"
                "</div>"
                "<div class='learning-example-result'>"
                "That move becomes less likely next time."
                "</div></div>"
                "</div>",
                unsafe_allow_html=True,
            )

            st.info(
                "MENACE does not follow a fixed list of rules. It chooses using "
                "bead-based chances, so it may still make different choices in "
                "the same situation."
            )

            st.markdown("### Look inside the matchbox for the current board")
            matchbox = self._current_matchbox()
            if matchbox is not None:
                probabilities = matchbox.probabilities()
                rows = [
                    {
                        "matchbox_move": move,
                        "original_board_square": move,
                        "beads": beads,
                        "probability": probabilities.get(move, 0.0),
                        "selected": False,
                    }
                    for move, beads in sorted(matchbox.beads.items())
                ]
                record = {
                    "matchbox_state": matchbox.state,
                    "probability_rows": rows,
                    "selected_matchbox_move": None,
                }
                box = self._temporary_matchbox_from_record(record)
                if box is not None:
                    st.markdown(
                        self.visualiser.educational_matchbox_html(
                            box,
                            square_labels=self.config.square_labels,
                            title="One matchbox from MENACE's memory",
                            show_probabilities=False,
                        ),
                        unsafe_allow_html=True,
                    )
            else:
                menace = st.session_state[SessionKey.MENACE]
                remembered = len(getattr(menace, "matchboxes", {}))
                st.caption(
                    "Play until MENACE chooses a move to select the matchbox "
                    "for that current board position. "
                    f"MENACE currently has {remembered:,} matchboxes in its "
                    "shared memory."
                )

            st.success(
                "After many games, useful moves usually have larger bead groups, "
                "so MENACE is more likely to choose them."
            )

            with st.container(border=True):
                st.markdown("### Try it yourself")
                st.write(
                    "Play a game to watch one decision, or give MENACE more "
                    "practice to see its memory grow."
                )
                left, right = st.columns(2)
                left.button(
                    "▶ Play with MENACE",
                    key="learning_go_to_play",
                    type="primary",
                    on_click=self._open_play_from_learning,
                    width="stretch",
                )
                right.button(
                    "🎮 Help MENACE practise",
                    key="learning_go_to_practice",
                    on_click=self._open_practice_from_learning,
                    width="stretch",
                )
            self._render_simple_evidence_panel("learning")
            return

        st.subheader("How MENACE Learns")
        st.caption(
            "The detailed explanation shows the same learning process using the internal "
            "state, weighted probabilities, reinforcement updates, and symmetry."
        )

        st.markdown("### Learning process")
        st.markdown(
            "1. The current board is converted to a canonical state.\n"
            "2. That state identifies one matchbox.\n"
            "3. Every legal canonical move has a bead count.\n"
            "4. Bead counts are normalised into a probability distribution.\n"
            "5. MENACE samples one move from that distribution.\n"
            "6. The move is mapped back to the original board orientation.\n"
            "7. At the end of the game, the selected beads are reinforced or reduced."
        )

        st.markdown("### Why symmetry is used")
        st.info(
            "Rotated and reflected versions of the same board share one canonical "
            "matchbox. This reduces duplicate states and allows experience learned "
            "in one orientation to be reused in equivalent orientations."
        )

        st.markdown("### Current Matchbox Explanation")
        st.code(menace.explain_state(board))

        matchbox = self._current_matchbox()
        if matchbox is not None:
            try:
                st.markdown("### Current bead-based move probabilities")
                st.caption(
                    "The chart shows the weighted probabilities for the current "
                    "matchbox. Larger bead counts produce higher selection probability."
                )
                self._render_exportable_figure(
                    self.visualiser.matchbox_probabilities(matchbox),
                    name="how_menace_learns_probabilities",
                    chart_key="how_menace_learns_probability_chart",
                )
            except Exception:
                st.info(
                    "The probability chart could not be rendered, but the current "
                    "bead counts and probabilities are shown in the explanation above."
                )
        else:
            st.info(
                "No current matchbox is available yet. Play a move or begin a "
                "new game to create the current decision state."
            )
