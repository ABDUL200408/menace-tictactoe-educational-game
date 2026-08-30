"""Streamlit Results page for MENACE training evidence and learning summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.statistics import StatisticsTracker
from src.trainer import Trainer
from src.visualisation import build_textual_visual_summary
from ui.session import SessionKey

class ResultsPageMixin:
    """Render MENACE training results, statistical summaries, and learning evidence."""

    def _open_practice_from_results(self) -> None:
        """Open the practice page from the Results & Learning page."""
        labels = (
            self.config.simple_page_labels
            if self._is_simple_mode()
            else self.config.technical_page_labels
        )
        self._set_active_page(labels[1])

    def _render_results_page(self) -> None:
        """Render the learner-facing results summary and supporting statistical evidence."""
        st.subheader(
            "See What MENACE Learned"
            if self._is_simple_mode()
            else "Results Dashboard"
        )
        history = self._normalise_history(
            st.session_state[SessionKey.HISTORY]
        )
        if history.empty:
            st.warning(
                "MENACE needs some practice games first."
                if self._is_simple_mode()
                else "No training history available. Train MENACE first."
            )
            if self._is_simple_mode():
                st.button(
                    "🎮 Help MENACE practise",
                    key="results_empty_go_to_practice",
                    type="primary",
                    on_click=self._open_practice_from_results,
                )
            return

        try:
            tracker = StatisticsTracker(history)
            summary = tracker.summary()
            latest = history.iloc[-1]

            if self._is_simple_mode():
                games = int(latest["game"])
                wins = int(latest["menace_wins"])
                losses = int(latest["opponent_wins"])
                draws = int(latest["draws"])
                overall_rate = float(summary.win_rate)
                loss_rate = losses / games if games else 0.0
                draw_rate = draws / games if games else 0.0

                phases = tracker.split_early_late(fraction=0.25)
                early_row = phases.loc[phases["phase"] == "early"].iloc[0]
                late_row = phases.loc[phases["phase"] == "late"].iloc[0]
                early_rate = float(early_row["mean_win_rate"])
                late_rate = float(late_row["mean_win_rate"])
                early_games = int(early_row["games"])
                late_games = int(late_row["games"])

                matchboxes = int(
                    latest.get(
                        "matchboxes",
                        getattr(summary, "final_matchboxes", 0) or 0,
                    )
                    or 0
                )
                total_beads = int(latest.get("total_beads", 0) or 0)
                context = st.session_state.get(SessionKey.LAST_TRAINING_CONTEXT)

                st.write(
                    "Here you can see what changed after MENACE practised."
                )
                st.caption(
                    "Latest training batch against The Guesser (Random). "
                    "Learning was active during these practice games."
                )

                cards = [
                    (
                        "green",
                        "🎮",
                        "Practice games",
                        f"{games:,}",
                        "How many games MENACE used to learn.",
                    ),
                    (
                        "green",
                        "🏆",
                        "MENACE wins",
                        f"{wins:,}",
                        "Games won by MENACE.",
                    ),
                    (
                        "loss",
                        "❌",
                        "Losses",
                        f"{losses:,}",
                        "Games MENACE lost to the guessing opponent.",
                    ),
                    (
                        "draw",
                        "🤝",
                        "Draws",
                        f"{draws:,}",
                        "Games that finished without a winner.",
                    ),
                ]
                card_html = "".join(
                    (
                        f"<div class='results-summary-card {colour_class}'>"
                        f"<div class='results-summary-icon'>{icon}</div>"
                        f"<div class='results-summary-label'>{label}</div>"
                        f"<div class='results-summary-value'>{value}</div>"
                        f"<div class='results-summary-help'>{help_text}</div>"
                        "</div>"
                    )
                    for colour_class, icon, label, value, help_text in cards
                )
                st.markdown(
                    f"<div class='results-summary-grid'>{card_html}</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("### What do the game results show?")
                st.markdown(
                    "<div class='results-outcome-bar' "
                    f"aria-label='MENACE won {wins} games, drew {draws}, "
                    f"and the opponent won {losses}'>"
                    f"<span class='results-win' style='width:{overall_rate * 100:.4f}%'></span>"
                    f"<span class='results-draw' style='width:{draw_rate * 100:.4f}%'></span>"
                    f"<span class='results-loss' style='width:{loss_rate * 100:.4f}%'></span>"
                    "</div>"
                    "<div class='results-outcome-legend'>"
                    f"<span>🏆 MENACE wins: {wins:,}</span>"
                    f"<span>🤝 Draws: {draws:,}</span>"
                    f"<span>❌ Opponent wins: {losses:,}</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Green shows MENACE wins, gold shows draws, and red shows "
                    "opponent wins."
                )

                change_points = late_rate - early_rate
                if change_points > 0:
                    learning_message = (
                        f"The later group of games was "
                        f"{abs(change_points) * 100:.1f} percentage points higher "
                        "than the early group in this run."
                    )
                    learning_state = "higher"
                    direction = "↑"
                elif change_points < 0:
                    learning_message = (
                        f"The later group of games was "
                        f"{abs(change_points) * 100:.1f} percentage points lower "
                        "than the early group in this run. Small falls can occur "
                        "because MENACE chooses moves probabilistically."
                    )
                    learning_state = "lower"
                    direction = "↓"
                else:
                    learning_message = (
                        "The early and later groups had the same win rate in this run."
                    )
                    learning_state = "same"
                    direction = "→"

                st.markdown("### How did performance change during practice?")
                st.markdown(
                    "<div class='results-learning-card'>"
                    "<div class='results-learning-stage'>"
                    "<div class='results-learning-face'>🎮</div>"
                    "<div class='results-learning-title'>Early practice phase</div>"
                    f"<div class='results-learning-rate'>{early_rate:.1%}</div>"
                    "<div class='results-learning-note'>Win rate across early games</div>"
                    "</div>"
                    f"<div class='results-learning-arrow'>{direction}</div>"
                    "<div class='results-learning-stage'>"
                    "<div class='results-learning-face'>📊</div>"
                    "<div class='results-learning-title'>Late practice phase</div>"
                    f"<div class='results-learning-rate'>{late_rate:.1%}</div>"
                    "<div class='results-learning-note'>Win rate across later games</div>"
                    "</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

                if learning_state == "higher":
                    st.success(learning_message)
                else:
                    st.info(learning_message)

                st.write(
                    f"The late-phase win rate was **{late_rate:.1%}**. "
                    "This means MENACE won about "
                    f"**{round(late_rate * 100):d} out of every 100 games** "
                    "in the later group of this training run."
                )
                st.caption(
                    f"This compares the first {early_games:,} games with the last "
                    f"{late_games:,} games. The overall win rate for all {games:,} "
                    f"games was {overall_rate:.1%}. "
                    "They are descriptive results, not a guarantee that performance "
                    "must always rise. A negative change is possible and should be "
                    "reported honestly rather than hidden."
                )

                if matchboxes or total_beads:
                    if context:
                        before_matchboxes = int(context.get("before_matchboxes", matchboxes))
                        before_beads = int(context.get("before_beads", total_beads))
                        after_matchboxes = int(context.get("after_matchboxes", matchboxes))
                        after_beads = int(context.get("after_beads", total_beads))
                        st.markdown("### How did memory change in this practice round?")
                        memory_change = pd.DataFrame(
                            [
                                {
                                    "Memory measure": "Matchboxes",
                                    "Immediately before practice": before_matchboxes,
                                    "After practice": after_matchboxes,
                                    "Change": after_matchboxes - before_matchboxes,
                                },
                                {
                                    "Memory measure": "Beads",
                                    "Immediately before practice": before_beads,
                                    "After practice": after_beads,
                                    "Change": after_beads - before_beads,
                                },
                            ]
                        )
                        st.dataframe(memory_change, width="stretch", hide_index=True)
                        st.caption(
                            "These are the before-and-after values for the same latest "
                            "practice round summarised above, not the beginning of the "
                            "whole learning journey."
                        )
                    else:
                        st.markdown("### MENACE's current saved memory")
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {"Memory measure": "Matchboxes", "Current value": matchboxes},
                                    {"Memory measure": "Beads", "Current value": total_beads},
                                ]
                            ),
                            width="stretch",
                            hide_index=True,
                        )
                        st.caption(
                            "This saved result contains the final memory values. "
                            "Run another practice round to display an exact "
                            "before-and-after change table."
                        )

                with st.container(border=True):
                    st.markdown("### What could you try next?")
                    st.write(
                        "Give MENACE more practice, then return here to see "
                        "whether its final win rate changes."
                    )
                    st.button(
                        "🎮 Give MENACE more practice",
                        key="results_go_to_practice",
                        type="primary",
                        on_click=self._open_practice_from_results,
                    )

                self._render_simple_evidence_panel("results")
                st.caption(
                    "Detailed evidence includes statistical analysis, raw tables, "
                    "and export options."
                )
                return

            st.caption(
                "These plots and statistics are based on the same training "
                "run summarised on the Train page."
            )
            analysis_frame = tracker.export_analysis_frame()
            cols = st.columns(5)
            cols[0].metric("Games", int(latest["game"]))
            cols[1].metric("Wins", int(latest["menace_wins"]))
            cols[2].metric("Losses", int(latest["opponent_wins"]))
            cols[3].metric("Draws", int(latest["draws"]))
            cols[4].metric("Win Rate", f"{float(latest['win_rate']):.1%}")

            with st.expander(
                "Statistical analysis for report",
                expanded=True,
            ):
                st.write(tracker.educational_interpretation())
                st.dataframe(analysis_frame, width="stretch")
                stability = tracker.rate_stability()
                gain = tracker.learning_gain()
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Late − Early Win-rate Change",
                    f"{gain.absolute_improvement:+.1%}",
                )
                c2.metric(
                    "Win-rate Std Dev",
                    f"{stability['std_win_rate']:.3f}",
                )
                c3.metric(
                    "Final Matchboxes",
                    summary.final_matchboxes or 0,
                )

            figures = self.visualiser.training_dashboard_figures(history)
            for name, figure in figures.items():
                self._render_exportable_figure(
                    figure,
                    name=f"results_{name}",
                    chart_key=f"training_chart_{name}",
                )

            st.markdown("### Statistical Interpretation")
            st.info(
                "The late-minus-early value is a descriptive change within one "
                "training run. Positive values indicate a higher late-phase rate; "
                "negative values indicate a lower late-phase rate. Because MENACE "
                "uses stochastic action selection, either result is possible and "
                "does not by itself establish statistical significance."
            )
            st.write(build_textual_visual_summary(history))

            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button(
                    label="Download Training CSV",
                    data=Trainer.evidence_frame(history).to_csv(index=False),
                    file_name="training_log.csv",
                    mime="text/csv",
                )
            with col_b:
                analysis_csv = analysis_frame.to_csv(index=False)
                st.download_button(
                    label="Download Training Analysis CSV",
                    data=analysis_csv,
                    file_name="training_analysis.csv",
                    mime="text/csv",
                )

            if st.button("Export Results Figures as HTML"):
                paths = self.visualiser.export_training_evidence(history)
                st.session_state[SessionKey.EXPORTED_TRAINING_FIGURES] = [
                    str(path) for path in paths
                ]
                st.success(f"Exported {len(paths)} figure(s).")

            for path_str in st.session_state[
                SessionKey.EXPORTED_TRAINING_FIGURES
            ]:
                path = Path(path_str)
                if path.exists():
                    st.download_button(
                        label=f"Download {path.name}",
                        data=path.read_bytes(),
                        file_name=path.name,
                        mime="text/html",
                        key=f"download_training_{path.name}",
                    )
        except Exception as exc:
            st.error(f"Could not build results dashboard: {exc}")
