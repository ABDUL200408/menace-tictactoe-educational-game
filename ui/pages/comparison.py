"""Streamlit comparison page."""

from __future__ import annotations

from typing import Any
import secrets
from itertools import count

import pandas as pd
import streamlit as st

from src.menace import MENACEPlayer
from src.players import HeuristicPlayer, MinimaxPlayer, RandomPlayer
from src.statistics import OpponentComparator
from src.trainer import ComparisonConfig, ComparisonExperiment, OpponentSpec, Trainer
from src.visualisation import build_comparison_textual_summary
from ui.session import SessionKey

class ComparisonPageMixin:
    """Render and handle the comparison page."""

    def _render_comparison_opponent_intro_cards(self) -> None:
        """Draw the three learner-facing opponent cards used on the Compare page."""
        cards = [
            (
                "guesser",
                "🎲",
                "The Guesser (Random)",
                "Picks an available square<br>without following a plan.",
            ),
            (
                "rules",
                "📋",
                "The Rule Follower (Heuristic)",
                "Tries useful rules such as<br>winning, blocking and<br>choosing strong squares.",
            ),
            (
                "thinker",
                "🧠",
                "The Careful Thinker (Minimax)",
                "Looks ahead through possible<br>moves to find the strongest<br>choice.",
            ),
        ]
        html = "".join(
            (
                f"<div class='comparison-intro-card {style_name}'>"
                f"<div class='comparison-intro-icon'>{icon}</div>"
                f"<div class='comparison-intro-title'>{title}</div>"
                f"<div class='comparison-intro-text'>{description}</div>"
                "</div>"
            )
            for style_name, icon, title, description in cards
        )
        st.markdown(
            f"<div class='comparison-intro-grid'>{html}</div>",
            unsafe_allow_html=True,
        )

    @staticmethod
    def _comparison_rate(value: int, games: int) -> float:
        """Return a safe percentage for a comparison result."""
        return (value / games * 100.0) if games > 0 else 0.0

    def _render_comparison_result_card(
        self,
        *,
        icon: str,
        title: str,
        games: int,
        wins: int,
        draws: int,
        losses: int,
    ) -> str:
        """Build one learner-facing comparison result card with a coloured outcome bar."""
        win_rate = self._comparison_rate(wins, games)
        draw_rate = self._comparison_rate(draws, games)
        loss_rate = self._comparison_rate(losses, games)

        return (
            "<div class='comparison-score-card'>"
            f"<div class='comparison-score-icon'>{icon}</div>"
            f"<div class='comparison-score-title'>{title}</div>"
            f"<div class='comparison-score-main'>MENACE won {wins} of {games} games</div>"
            f"<div class='comparison-score-rate'>MENACE win rate: {win_rate:.0f}%</div>"
            "<div class='comparison-stacked-bar' "
            f"aria-label='MENACE won {wins} games, {draws} games were draws, and the opponent won {losses} games'>"
            f"<span class='comparison-bar-win' style='width:{win_rate:.4f}%'></span>"
            f"<span class='comparison-bar-draw' style='width:{draw_rate:.4f}%'></span>"
            f"<span class='comparison-bar-loss' style='width:{loss_rate:.4f}%'></span>"
            "</div>"
            "<div class='comparison-score-list'>"
            f"<div>🏆 MENACE wins: {wins}</div>"
            f"<div>🤝 Draws: {draws}</div>"
            f"<div>❌ Opponent wins: {losses}</div>"
            "</div>"
            "</div>"
        )

    def _open_practice_from_comparison(self) -> None:
        """Open the MENACE practice page from the comparison workflow."""
        labels = (
            self.config.simple_page_labels
            if self._is_simple_mode()
            else self.config.technical_page_labels
        )
        self._set_active_page(labels[1])

    def _render_comparison_page(self) -> None:
        """Render the comparison page in learner or technical form."""
        if self._is_simple_mode():
            st.subheader("Try Different Opponents")
            st.write(
                "MENACE first practises, then plays against three different "
                "computer players. This helps us see which playing style it "
                "finds easiest or hardest."
            )

            self._render_comparison_opponent_intro_cards()

            st.warning(
                "The Careful Thinker considers many future moves, so larger "
                "tests can take longer."
            )
            st.caption(
                "This test uses a separate copy of MENACE. It does not erase "
                "or replace the MENACE used on the Play page."
            )
            st.info(
                "What is measured? First, the separate MENACE copy learns during "
                "the selected practice games against The Guesser (Random). Next, "
                "learning is switched off and the trained copy is evaluated against "
                "Random, Heuristic and Minimax. The percentages below are evaluation "
                "win rates from MENACE's point of view—not training rates."
            )

            with st.container(border=True):
                st.markdown("### Choose the size of the test")

                trained_games = st.slider(
                    "Practice games before the test",
                    100,
                    min(10000, self.config.max_training_games),
                    min(
                        self.config.default_comparison_training_games,
                        min(10000, self.config.max_training_games),
                    ),
                    100,
                    key="comparison_training_games",
                )
                st.caption(
                    "More practice can help MENACE improve, but takes longer."
                )

                evaluation_games = st.slider(
                    "Games against each opponent",
                    10,
                    min(1000, self.config.max_comparison_evaluation_games),
                    min(
                        self.config.default_comparison_evaluation_games,
                        min(1000, self.config.max_comparison_evaluation_games),
                    ),
                    10,
                    key="comparison_evaluation_games",
                )
                st.caption(
                    "More games give a clearer result because one lucky game "
                    "has less effect."
                )

                repetitions = st.slider(
                    "Repeat the whole test",
                    1,
                    min(100, self.config.max_comparison_repetitions),
                    1,
                    1,
                    key="comparison_repetitions",
                )
                st.caption(
                    "Repeating checks whether a similar result happens again."
                )

            button_label = "Start the games"
        else:
            st.subheader("Comparative AI Evaluation")
            st.write(
                "Train a fresh MENACE model and evaluate it against Random, "
                "Heuristic and Minimax opponents."
            )
            trained_games = st.slider(
                "Training Games Before Evaluation",
                100,
                self.config.max_training_games,
                self.config.default_comparison_training_games,
                100,
                key="comparison_training_games",
            )
            evaluation_games = st.slider(
                "Evaluation Games Per Opponent",
                10,
                self.config.max_comparison_evaluation_games,
                self.config.default_comparison_evaluation_games,
                10,
                key="comparison_evaluation_games",
            )
            repetitions = st.slider(
                "Repetitions",
                1,
                self.config.max_comparison_repetitions,
                1,
                1,
                key="comparison_repetitions",
            )
            button_label = "Run AI Comparison"

        estimated_games = repetitions * (
            trained_games + evaluation_games * 3
        )
        st.info(
            f"The computer will play about **{estimated_games:,} games**. "
            "You do not have to watch them - they happen quickly."
            if self._is_simple_mode()
            else
            f"This will run approximately {estimated_games} total games, "
            "including training and evaluation."
        )

        if st.button(button_label, type="primary"):
            self._run_comparison(
                trained_games=trained_games,
                evaluation_games=evaluation_games,
                repetitions=repetitions,
            )

        results = st.session_state[SessionKey.COMPARISON]
        if not results.empty:
            self._render_comparison_results(results)
        else:
            st.info(
                "Start the games to see which opponent MENACE finds easiest "
                "and hardest."
                if self._is_simple_mode()
                else "No comparison results yet."
            )

        if self._is_simple_mode():
            self._render_simple_evidence_panel("comparison")

    def _run_comparison(
        self,
        trained_games: int,
        evaluation_games: int,
        repetitions: int,
    ) -> None:
        """Run the MENACE comparison and present messages appropriate to the active mode."""
        try:
            # A new seed makes every button press a genuinely new stochastic
            # experiment even when the sliders are unchanged. Store it so the
            # exact run remains reproducible from exported evidence.
            run_number = int(st.session_state.get("comparison_run_number", 0)) + 1
            run_seed = secrets.randbelow(2_147_483_646) + 1
            st.session_state["comparison_run_number"] = run_number
            st.session_state["comparison_run_seed"] = run_seed

            # Every repetition needs its own random streams. Reusing the same
            # seed here would make repeated trials exact duplicates and would
            # understate the natural variation in a stochastic experiment.
            menace_seeds = count(run_seed, 10)
            training_seeds = count(run_seed + 1, 10)
            random_evaluation_seeds = count(run_seed + 2, 10)
            progress = st.progress(
                0,
                text=(
                    "Getting the players ready..."
                    if self._is_simple_mode()
                    else "Preparing comparison experiment..."
                ),
            )
            experiment = ComparisonExperiment(
                menace_factory=lambda: MENACEPlayer(
                    mark="O",
                    seed=next(menace_seeds),
                    use_symmetry=self.config.use_symmetry,
                    reward_win=self.config.reward_win_default,
                    reward_draw=self.config.reward_draw_default,
                    penalty_loss=self.config.penalty_loss_default,
                ),
                training_opponent_factory=lambda: RandomPlayer(
                    mark="X",
                    seed=next(training_seeds),
                ),
                evaluation_opponents=[
                    OpponentSpec(
                        name="Random Baseline",
                        opponent_type="Random",
                        factory=lambda: RandomPlayer(
                            mark="X",
                            seed=next(random_evaluation_seeds),
                        ),
                    ),
                    OpponentSpec(
                        name="Heuristic AI",
                        opponent_type="Heuristic",
                        factory=lambda: HeuristicPlayer(mark="X"),
                    ),
                    OpponentSpec(
                        name="Minimax AI",
                        opponent_type="Minimax",
                        factory=lambda: MinimaxPlayer(mark="X"),
                    ),
                ],
                config=ComparisonConfig(
                    trained_games=trained_games,
                    evaluation_games=evaluation_games,
                    repetitions=repetitions,
                    seed=run_seed,
                    experiment_name=(
                        self.config.streamlit_comparison_experiment_name
                    ),
                ),
            )

            progress.progress(
                25,
                text=(
                    "MENACE is practising and playing..."
                    if self._is_simple_mode()
                    else "Running training and evaluation..."
                ),
            )
            with st.spinner(
                "The computer players are finishing their games..."
                if self._is_simple_mode()
                else "Running MENACE vs Random, Heuristic, and Minimax..."
            ):
                results = experiment.run()

            progress.progress(
                85,
                text=(
                    "Putting the results together..."
                    if self._is_simple_mode()
                    else "Saving comparison results..."
                ),
            )
            ComparisonExperiment.save_comparison(
                results,
                self.config.comparison_results_path,
            )
            self._save_comparison_analysis(results)
            st.session_state[SessionKey.COMPARISON] = results
            self._save_learning_bundle(show_message=False)
            progress.progress(
                100,
                text=(
                    "Games finished."
                    if self._is_simple_mode()
                    else "Comparison complete."
                ),
            )

            if self._is_simple_mode():
                st.success(
                    "The games are finished. Now you can see how well "
                    f"MENACE did against each kind of opponent. Test run "
                    f"**{run_number}** used seed **{run_seed}**."
                )
            else:
                st.success("Comparison experiment completed.")
                st.info(
                    f"Results saved to {self.config.comparison_results_path}"
                )
        except Exception as exc:
            st.error(f"Comparison failed: {exc}")
            if not self._is_simple_mode():
                st.exception(exc)

    def _save_comparison_analysis(self, results: pd.DataFrame) -> None:
        """Save the aggregated opponent ranking for the current comparison run."""
        comparator = OpponentComparator(results)
        ranking = comparator.ranking()
        self.config.comparison_analysis_path.parent.mkdir(parents=True, exist_ok=True)
        ranking.to_csv(self.config.comparison_analysis_path, index=False)

    def _comparison_best_hardest_text(self, ranking: pd.DataFrame, best_worst: dict[str, Any]) -> tuple[str, str, str]:
        """Return consistent best/hardest opponent text for AI comparison."""
        if ranking.empty or 'mean_win_rate' not in ranking.columns:
            return (str(best_worst.get('best_opponent_type', 'N/A')), str(best_worst.get('worst_opponent_type', 'N/A')), 'Comparison interpretation is unavailable because ranking data is incomplete.')
        ranking_copy = ranking.copy()
        ranking_copy['mean_win_rate'] = ranking_copy['mean_win_rate'].astype(float)
        best_rate = float(ranking_copy['mean_win_rate'].max())
        worst_rate = float(ranking_copy['mean_win_rate'].min())
        best_opponents = ranking_copy.loc[ranking_copy['mean_win_rate'] == best_rate, 'opponent_type'].astype(str).tolist()
        hardest_opponents = ranking_copy.loc[ranking_copy['mean_win_rate'] == worst_rate, 'opponent_type'].astype(str).tolist()
        best_label = ', '.join(best_opponents) if best_opponents else 'N/A'
        hardest_label = ', '.join(hardest_opponents) if hardest_opponents else 'N/A'
        minimax_note = ''
        if any((opponent.lower() == 'minimax' for opponent in hardest_opponents)):
            minimax_note = ' Minimax can be regarded as the most theoretically difficult opponent because it uses optimal search, while any other tied opponent is also challenging under this training setup.'
        interpretation = f'MENACE was evaluated against Random, Heuristic, and Minimax opponents. Based on mean win rate, it performed best against {best_label} ({best_rate:.1%}). It failed to win or performed worst against {hardest_label} ({worst_rate:.1%}).{minimax_note}'
        return (best_label, hardest_label, interpretation)

    def _render_simple_comparison_results(
        self,
        results: pd.DataFrame,
    ) -> None:
        """Render comparison results in the learner-facing design."""
        comparator = OpponentComparator(results)
        rows = comparator.evaluation_rows()
        if rows.empty:
            st.info("No finished test games were found.")
            return

        grouped = (
            rows.groupby("opponent_type", as_index=False)
            .agg(
                games=("games", "sum"),
                wins=("menace_wins", "sum"),
                losses=("opponent_wins", "sum"),
                draws=("draws", "sum"),
            )
        )
        grouped["win_rate"] = grouped["wins"] / grouped["games"]

        opponent_details = {
            "Random": ("🎲", "The Guesser (Random)"),
            "Heuristic": ("📋", "The Rule Follower (Heuristic)"),
            "Minimax": ("🧠", "The Careful Thinker (Minimax)"),
        }
        ordered_types = ["Random", "Heuristic", "Minimax"]
        grouped_lookup = {
            str(row["opponent_type"]): row
            for _, row in grouped.iterrows()
        }

        st.markdown("### How did MENACE do?")
        st.caption(
            "Every result below is shown from MENACE's point of view. "
            "‘Opponent wins’ means the named opponent defeated MENACE."
        )
        run_number = st.session_state.get("comparison_run_number")
        run_seed = st.session_state.get("comparison_run_seed")
        if run_number and run_seed:
            st.info(
                f"These are **evaluation** results from test run **{run_number}** "
                f"(seed **{run_seed}**). Learning was off during these games. "
                "Pressing **Start the games** again runs a new stochastic test, so "
                "the numbers can change even when the sliders stay the same. A "
                "larger number of games makes random variation smaller."
            )

        result_cards: list[str] = []
        for opponent_type in ordered_types:
            row = grouped_lookup.get(opponent_type)
            if row is None:
                continue
            icon, title = opponent_details[opponent_type]
            result_cards.append(
                self._render_comparison_result_card(
                    icon=icon,
                    title=title,
                    games=int(row["games"]),
                    wins=int(row["wins"]),
                    draws=int(row["draws"]),
                    losses=int(row["losses"]),
                )
            )

        st.markdown(
            "<div class='comparison-score-grid'>"
            + "".join(result_cards)
            + "</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "The coloured bars show green MENACE wins, gold draws, and red opponent wins."
        )

        best_rate = float(grouped["win_rate"].max())
        easiest_types = grouped.loc[
            grouped["win_rate"].sub(best_rate).abs() < 1e-12,
            "opponent_type",
        ].astype(str).tolist()
        hardest_types = self._hardest_opponent_types(grouped)

        easiest_names = [
            opponent_details.get(item, ("", item))[1]
            for item in easiest_types
        ]
        hardest_names = [
            opponent_details.get(item, ("", item))[1]
            for item in hardest_types
        ]

        st.success(
            "MENACE won most often against "
            f"**{self._friendly_join(easiest_names)}**."
        )

        if len(hardest_names) > 1:
            st.info(
                "It found "
                f"**{self._friendly_join(hardest_names)}** the hardest in "
                "this test. When two opponents have the same result, both "
                "are named."
            )
        elif hardest_names:
            st.info(
                f"It found **{hardest_names[0]}** the hardest in this test."
            )

        st.write(
            "This shows that practice can help MENACE, but players that follow "
            "strong rules or think ahead can still be difficult to beat."
        )

        with st.container(border=True):
            st.markdown("### What could you try next?")
            st.write(
                "Give MENACE more practice, then repeat the comparison to see "
                "whether the result changes."
            )
            st.button(
                "🎮 Give MENACE more practice",
                key="comparison_more_practice",
                type="primary",
                on_click=self._open_practice_from_comparison,
            )

        st.caption(
            "Teachers or parents can use this page to discuss three ways "
            "computers can play: guessing, following rules and thinking ahead."
        )

    def _render_comparison_results(self, results: pd.DataFrame) -> None:
        """Render comparison results for the active interface mode."""
        if self._is_simple_mode():
            self._render_simple_comparison_results(results)
            return

        st.markdown("## Comparison Summary")
        st.dataframe(results, width="stretch")
        try:
            comparator = OpponentComparator(results)
            evaluation_rows = comparator.evaluation_rows()
            ranking = comparator.ranking()
            best_worst = comparator.best_and_worst()
            best_label, hardest_label, comparison_interpretation = (
                self._comparison_best_hardest_text(ranking, best_worst)
            )

            display_columns = [
                "opponent_type",
                "opponent_name",
                "games",
                "menace_wins",
                "opponent_wins",
                "draws",
                "win_rate",
                "loss_rate",
                "draw_rate",
                "matchboxes",
                "total_beads",
            ]
            available_columns = [
                column
                for column in display_columns
                if column in evaluation_rows.columns
            ]
            st.markdown("### Evaluation Rows Only")
            st.dataframe(
                evaluation_rows[available_columns],
                width="stretch",
            )

            cols = st.columns(3)
            cols[0].metric(
                "Best Opponent Result",
                best_label,
                f"{float(best_worst['best_mean_win_rate']):.1%} mean win rate",
            )
            cols[1].metric(
                "Hardest / Tied Hardest",
                hardest_label,
                f"{float(best_worst['worst_mean_win_rate']):.1%} mean win rate",
            )
            cols[2].metric("Evaluation Rows", len(evaluation_rows))

            with st.expander(
                "Opponent ranking and aggregated comparison analysis",
                expanded=True,
            ):
                st.dataframe(ranking, width="stretch")
                st.info(comparison_interpretation)
                st.write(comparator.educational_interpretation())

            figures = self.visualiser.comparison_dashboard_figures(results)
            for name, figure in figures.items():
                self._render_exportable_figure(
                    figure,
                    name=f"comparison_{name}",
                    chart_key=f"comparison_chart_{name}",
                )

            st.markdown("### Interpretation")
            st.info(comparison_interpretation)
            st.write(build_comparison_textual_summary(results))

            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button(
                    label="Download Comparison CSV",
                    data=Trainer.evidence_frame(results).to_csv(index=False),
                    file_name="comparison_results.csv",
                    mime="text/csv",
                )
            with col_b:
                st.download_button(
                    label="Download Comparison Analysis CSV",
                    data=ranking.to_csv(index=False),
                    file_name="comparison_analysis.csv",
                    mime="text/csv",
                )

            if st.button("Export Comparison Charts as HTML"):
                exported = self.visualiser.export_comparison_dashboard_html(
                    results
                )
                st.session_state[
                    SessionKey.EXPORTED_COMPARISON_FIGURES
                ] = exported
                st.success(
                    f"Exported {len(exported)} comparison figure(s)."
                )

            exported = st.session_state.get(
                SessionKey.EXPORTED_COMPARISON_FIGURES,
                {},
            )
            if exported:
                st.markdown("### Exported Comparison Figure Paths")
                self._show_exported_paths(exported)
        except Exception as exc:
            st.error(f"Could not render comparison results: {exc}")
            st.exception(exc)
