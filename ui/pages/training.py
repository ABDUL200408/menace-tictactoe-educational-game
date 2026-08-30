"""Streamlit Train page for MENACE practice and training evidence."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.players import RandomPlayer
from src.statistics import StatisticsTracker
from src.trainer import Trainer, TrainingConfig
from ui.session import SessionKey

class TrainingPageMixin:
    """Render MENACE practice controls, progress summaries, and training evidence."""

    def _open_results_from_training(self) -> None:
        """Open the results page without changing MENACE's learned memory."""
        labels = (
            self.config.simple_page_labels
            if self._is_simple_mode()
            else self.config.technical_page_labels
        )
        self._set_active_page(labels[3])

    def _render_training_opponent_explanation(self) -> None:
        """State which opponent is used for learning and which are evaluation-only."""
        if self._is_simple_mode():
            st.markdown("### Practice opponent")
            st.success(
                "**The Guesser (Random Player)** — it chooses equally from the "
                "empty squares and does not learn. MENACE is the only player "
                "that changes its beads during these practice games."
            )
            st.caption(
                "Heuristic and Minimax are kept for the separate comparison "
                "page. They test MENACE after training; this practice button "
                "does not train against them."
            )
        else:
            st.info(
                "**Training opponent: Random Player.** Heuristic and Minimax "
                "are evaluation opponents on the Compare page, not "
                "training opponents in this experiment."
            )

    def _render_current_learning_status(self) -> None:
        """Explain exactly which memory the next practice batch will use."""
        menace = st.session_state[SessionKey.MENACE]
        current = self._learning_memory_snapshot(menace)
        baseline = st.session_state.get(SessionKey.LEARNING_BASELINE, current)
        source = str(
            st.session_state.get(SessionKey.LEARNING_SOURCE, "Fresh start")
        )
        added_practice = max(0, current["games"] - int(baseline.get("games", 0)))
        human_games = int(
            st.session_state.get(SessionKey.INTERACTIVE_GAMES, 0)
        )
        practice_games = int(current["games"])
        learning_total = human_games + practice_games

        if source == "Continued previous learning":
            st.success("### ✅ Continuing previous learning")
            st.write(
                "The saved MENACE model is loaded. Pressing the practice button "
                "will continue from this memory; it will not start MENACE again."
            )
        elif source == "Fresh start":
            st.info("### 🆕 Current learning journey: fresh start")
            st.write(
                "MENACE started without saved learning. Any practice games will "
                "build this new memory."
            )
        else:
            st.success("### 💾 Current learning journey is saved")
            st.write(
                "This is the active MENACE memory. Further practice will add "
                "to it, and the saved bundle can be continued later."
            )

        cols = st.columns(4)
        cols[0].metric(
            "Journey-start matchboxes",
            f"{int(baseline.get('matchboxes', 0)):,}",
            help="Matchboxes at the beginning of this learning journey, before human play or practice.",
        )
        cols[1].metric(
            "Journey-start beads",
            f"{int(baseline.get('beads', 0)):,}",
            help="Beads at the beginning of this learning journey, before human play or practice.",
        )
        cols[2].metric("Current matchboxes", f"{current['matchboxes']:,}")
        cols[3].metric("Current beads", f"{current['beads']:,}")
        st.caption(
            "Journey-start values describe the beginning of the whole learning "
            "journey. The results below separately show the memory immediately "
            "before and after the latest practice round."
        )
        st.write(
            f"**Added in this browser session:** {added_practice:,} practice "
            f"games and {human_games:,} completed human games."
        )
        st.markdown(
            f"""
            <div class="memory-source-grid training-counts">
              <div class="memory-source human"><b>👤 {human_games:,}</b><span>Human games</span></div>
              <div class="memory-source practice"><b>🤖 {practice_games:,}</b><span>All practice games</span></div>
              <div class="memory-source total"><b>⭐ {learning_total:,}</b><span>Total games influencing memory</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info(
            "The practice results and charts count **practice games only**. Human "
            "games are shown separately, although both activities update the same memory."
        )
        st.caption(
            "Beads are cumulative move weights. The model stores their current "
            "balances, not a separate historical label for every bead. Therefore "
            "one bead total cannot be divided exactly into wins, draws and losses."
        )

    def _render_simple_training_journey(self) -> None:
        """Show the practice cycle as a short visual story."""
        st.markdown(
            """
            <div class="training-journey" role="img"
                 aria-label="MENACE plays practice games, changes beads, remembers board situations, and becomes a better player">
                <div class="journey-step">
                    <div class="journey-icon">🤖</div>
                    <div class="journey-title">MENACE starts</div>
                    <div class="journey-text">A beginner with simple matchboxes.</div>
                </div>
                <div class="journey-arrow">→</div>
                <div class="journey-step">
                    <div class="journey-icon">🎮</div>
                    <div class="journey-title">Plays games</div>
                    <div class="journey-text">It practises again and again.</div>
                </div>
                <div class="journey-arrow">→</div>
                <div class="journey-step">
                    <div class="journey-icon">🟢</div>
                    <div class="journey-title">Changes beads</div>
                    <div class="journey-text">Good moves gain beads; poor moves may lose one.</div>
                </div>
                <div class="journey-arrow">→</div>
                <div class="journey-step">
                    <div class="journey-icon">📦</div>
                    <div class="journey-title">Builds memory</div>
                    <div class="journey-text">More board situations are remembered.</div>
                </div>
                <div class="journey-arrow">→</div>
                <div class="journey-step">
                    <div class="journey-icon">⭐</div>
                    <div class="journey-title">Gets better</div>
                    <div class="journey-text">Helpful choices become easier to pick.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


    def _open_play_after_training(self) -> None:
        """
        Start a fresh board and switch directly to the Play page.

        This is used as a Streamlit button callback. Callbacks run before the
        next script rerun, so the navigation widget can safely read the new
        ACTIVE_PAGE value when it is created.
        """
        play_labels = (
            self.config.simple_page_labels
            if self._is_simple_mode()
            else self.config.technical_page_labels
        )
        play_page = play_labels[0]

        self._start_new_game(
            message=(
                "MENACE is ready after practising. "
                "Choose an empty lettered square to test what it learned."
            )
        )

        self._set_active_page(play_page)
        st.session_state[SessionKey.PLAY_READY_BANNER] = True
        st.session_state[SessionKey.SHOW_WELCOME] = False
        st.session_state[SessionKey.SHOW_DEMO] = False

    def _render_simple_training_results(
        self,
        *,
        games: int,
        menace_wins: int,
        opponent_wins: int,
        draws: int,
        matchboxes: int,
        total_beads: int,
        history: pd.DataFrame,
    ) -> None:
        """Render child-friendly practice results using icons and diagrams."""
        st.markdown("### How did MENACE do in this practice round?")
        st.success(
            f"MENACE played **{games:,}** practice games against a simple "
            f"opponent that guesses moves. It won **{menace_wins:,}** times, "
            f"lost **{opponent_wins:,}** times, and drew **{draws:,}** times."
        )
        safe_games = max(1, games)
        rates = pd.DataFrame(
            [
                {
                    "Outcome": "MENACE wins",
                    "Calculation": f"{menace_wins:,} ÷ {games:,}",
                    "Rate": f"{menace_wins / safe_games:.1%}",
                },
                {
                    "Outcome": "MENACE losses",
                    "Calculation": f"{opponent_wins:,} ÷ {games:,}",
                    "Rate": f"{opponent_wins / safe_games:.1%}",
                },
                {
                    "Outcome": "Draws",
                    "Calculation": f"{draws:,} ÷ {games:,}",
                    "Rate": f"{draws / safe_games:.1%}",
                },
            ]
        )
        st.markdown("#### How the result percentages are calculated")
        st.dataframe(rates, width="stretch", hide_index=True)
        st.caption(
            "Rate = number of that outcome ÷ all games in this practice round. "
            f"Check: {menace_wins:,} + {opponent_wins:,} + {draws:,} = "
            f"{menace_wins + opponent_wins + draws:,} games."
        )

        cards = [
            ("green", "🎮", "Games this round", f"{games:,}", "New games added by this button press."),
            ("green", "🏆", "Wins", f"{menace_wins:,}", "Games MENACE won."),
            ("loss", "❌", "Losses", f"{opponent_wins:,}", "Games the guessing player won."),
            ("draw", "🤝", "Draws", f"{draws:,}", "Games with no winner."),
            (
                "green",
                "📦",
                "Board situations",
                f"{matchboxes:,}",
                "Different situations MENACE can recognise.",
            ),
        ]
        card_html = "".join(
            (
                f"<div class='training-result-card {colour_class}'>"
                f"<div class='training-result-icon'>{icon}</div>"
                f"<div class='training-result-label'>{label}</div>"
                f"<div class='training-result-value'>{value}</div>"
                f"<div class='training-result-help'>{help_text}</div>"
                "</div>"
            )
            for colour_class, icon, label, value, help_text in cards
        )
        st.markdown(
            f"<div class='training-result-grid'>{card_html}</div>",
            unsafe_allow_html=True,
        )

        # Concrete visualisation of memory growth.
        bead_dots = "".join(
            "<span class='training-bead'></span>" for _ in range(18)
        )
        box_icons = " ".join("📦" for _ in range(min(8, max(1, matchboxes // 40))))
        st.markdown(
            "<div class='training-memory-panel'>"
            "<div class='training-memory-block green'>"
            "<div class='training-memory-title'>📦 Matchboxes remembered</div>"
            f"<div class='training-box-icons'>{box_icons}</div>"
            f"<div class='training-memory-number'>{matchboxes:,}</div>"
            "<div class='training-memory-text'>"
            "How many different board situations MENACE can now recognise."
            "</div></div>"
            "<div class='training-memory-block green'>"
            "<div class='training-memory-title'>🟢 Beads in memory</div>"
            f"<div class='training-bead-cloud'>{bead_dots}</div>"
            f"<div class='training-memory-number'>{total_beads:,}</div>"
            "<div class='training-memory-text'>"
            "Bead counts encode how strongly MENACE currently prefers its moves."
            "</div></div>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.info(
            f"MENACE now has **{total_beads:,} beads** across its matchboxes. "
            "Open **See What MENACE Learned** for the early-versus-late "
            "comparison and detailed graphs."
        )
        st.button(
            "📈 View learning graphs",
            key="training_go_to_results",
            width="stretch",
            on_click=self._open_results_from_training,
            help=(
                "Open the results page for the latest practice round. "
                "This does not start another game or change MENACE's memory."
            ),
        )
        st.write(
            "After this practice, return to **Play** and see whether MENACE "
            "is harder to beat."
        )
        st.write(
            "If MENACE still seems easy to beat, you can give it even more practice."
        )

        st.markdown(
            "<div class='training-play-cta'>"
            "<div class='training-play-icon'>🎉</div>"
            "<div>"
            "<div class='training-play-title'>MENACE has finished practising!</div>"
            "<div class='training-play-text'>"
            "Its matchboxes and learned beads are kept. Start a fresh board "
            "and see whether it is harder to beat."
            "</div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.button(
            "▶ Play against MENACE now",
            key="prepare_game_after_training",
            type="primary",
            width="stretch",
            on_click=self._open_play_after_training,
            help=(
                "Open the Play page with a fresh board while keeping "
                "everything MENACE learned during practice."
            ),
        )

    def _render_training_page(self) -> None:
        """Render MENACE practice controls and the appropriate level of supporting detail."""
        if self._is_simple_mode():
            st.subheader("Help MENACE Practise")
            st.write(
                "MENACE needs practice, just like a person learning a new game. "
                "It plays practice games by itself and remembers which choices helped."
            )
            st.info(
                "It practises against a simple opponent that picks squares by guessing. "
                "After each game, MENACE gets more beads for good moves and can "
                "lose a bead for poor moves."
            )
            with st.container(border=True):
                self._render_training_opponent_explanation()

            with st.container(border=True):
                self._render_current_learning_status()

            with st.container(border=True):
                st.markdown("### What happens during practice?")
                self._render_simple_training_journey()

            with st.container(border=True):
                games = st.slider(
                    "How many practice games?",
                    100,
                    min(5000, self.config.max_training_games),
                    min(
                        self.config.default_training_games,
                        min(5000, self.config.max_training_games),
                    ),
                    100,
                )
                st.caption(
                    "More games means more practice for MENACE, but it takes longer."
                )
                reward_win = self.config.reward_win_default
                reward_draw = self.config.reward_draw_default
                penalty_loss = self.config.penalty_loss_default
                label = f"Start {games:,} practice games"
        else:
            st.subheader("Train MENACE")
            st.write(
                "Train the current MENACE model against a Random baseline "
                "and record quantitative evidence."
            )
            self._render_training_opponent_explanation()
            games = st.slider(
                "Training Games",
                100,
                self.config.max_training_games,
                self.config.default_training_games,
                100,
            )
            reward_win = st.number_input(
                "Reward for MENACE Win",
                value=self.config.reward_win_default,
            )
            reward_draw = st.number_input(
                "Reward for Draw",
                value=self.config.reward_draw_default,
            )
            penalty_loss = st.number_input(
                "Penalty for Loss",
                value=self.config.penalty_loss_default,
            )
            label = "Start Training"

        notice = st.session_state.pop(SessionKey.TRAINING_NOTICE, None)
        if notice:
            st.success(str(notice))

        if st.button(label, type="primary", key="start_practice_batch"):
            completed = self._train_menace(
                games=games,
                reward_win=int(reward_win),
                reward_draw=int(reward_draw),
                penalty_loss=int(penalty_loss),
            )
            if completed:
                # The sidebar is rendered before this page. A single rerun makes
                # its cumulative counters reflect the batch that has just ended.
                # Streamlit resets button state on the rerun, so training is not
                # executed a second time.
                st.rerun()

        latest_summary = st.session_state.get(SessionKey.LAST_TRAINING_SUMMARY)
        history = self._normalise_history(
            st.session_state.get(SessionKey.HISTORY, pd.DataFrame())
        )
        if latest_summary and not history.empty:
            with st.container(border=True):
                self._render_training_summary_panel(latest_summary, history)

        # Detailed charts and CSV evidence live on the Results page. Keeping
        # them in one place avoids showing the same analysis twice.

    def _train_menace(
        self,
        games: int,
        reward_win: int,
        reward_draw: int,
        penalty_loss: int,
    ) -> bool:
        """Train the current MENACE player and retain structured training evidence."""
        menace = st.session_state[SessionKey.MENACE]
        try:
            # The requested number is one additional training batch. Capture the
            # lifetime totals before it starts so the interface can distinguish
            # this run from MENACE's cumulative learning.
            previous_totals = {
                "games": int(menace.games_played),
                "wins": int(menace.wins),
                "losses": int(menace.losses),
                "draws": int(menace.draws),
            }
            before_memory = self._learning_memory_snapshot(menace)

            self._validate_training_inputs(
                games,
                reward_win,
                reward_draw,
                penalty_loss,
            )

            if self._is_simple_mode():
                status = st.status(
                    "🤖 MENACE is getting ready to practise...",
                    expanded=True,
                )
                status.write("🎮 Starting the practice games.")
                status.write("🟢 Watching which moves help and changing beads.")
                status.write("📦 Remembering new board situations.")
                spinner_text = f"Playing {games:,} quick practice games..."
            else:
                status = None
                spinner_text = f"Training MENACE for {games} games..."

            # Capture the cumulative model immediately before this batch.
            # It is used only to calculate the latest batch's net contribution.
            before_batch_beads = self._matchbox_bead_snapshot(menace)

            with st.spinner(spinner_text):
                trainer = Trainer(
                    menace_player=menace,
                    opponent=RandomPlayer(
                        mark="X",
                        seed=self.config.random_seed,
                    ),
                    config=TrainingConfig(
                        games=games,
                        reward_win=reward_win,
                        reward_draw=reward_draw,
                        penalty_loss=penalty_loss,
                        seed=self.config.random_seed,
                        output_path=self.config.training_log_path,
                        experiment_name=(
                            self.config.streamlit_training_experiment_name
                        ),
                    ),
                )
                report = trainer.run_with_report(train=True)
                history = self._normalise_history(report.history)
                trainer.save_history(history, self.config.training_log_path)
                self._save_training_analysis(history)

            st.session_state[SessionKey.HISTORY] = history
            st.session_state[SessionKey.LAST_TRAINING_SUMMARY] = (
                report.final_summary
            )
            st.session_state[SessionKey.LAST_TRAINING_CONTEXT] = {
                "batch_games": int(games),
                "previous_games": previous_totals["games"],
                "total_games": int(menace.games_played),
                "previous_wins": previous_totals["wins"],
                "total_wins": int(menace.wins),
                "previous_losses": previous_totals["losses"],
                "total_losses": int(menace.losses),
                "previous_draws": previous_totals["draws"],
                "total_draws": int(menace.draws),
                "before_matchboxes": before_memory["matchboxes"],
                "after_matchboxes": len(menace.matchboxes),
                "before_beads": before_memory["beads"],
                "after_beads": sum(
                    box.total_beads() for box in menace.matchboxes.values()
                ),
                "human_games": int(
                    st.session_state.get(SessionKey.INTERACTIVE_GAMES, 0)
                ),
            }
            st.session_state.setdefault(SessionKey.LEARNING_EVENTS, []).append(
                {
                    "kind": "practice",
                    "label": f"Practice round ({int(games):,} games)",
                    "before": int(before_memory["beads"]),
                    "created": 0,
                    "reinforcement": (
                        int(sum(box.total_beads() for box in menace.matchboxes.values()))
                        - int(before_memory["beads"])
                    ),
                    "after": int(
                        sum(box.total_beads() for box in menace.matchboxes.values())
                    ),
                }
            )
            human_snapshot = st.session_state.get(SessionKey.LAST_HUMAN_MEMORY)
            if human_snapshot is not None:
                origin = human_snapshot.get("summary", before_memory)
                st.session_state[SessionKey.TRAINING_SINCE_HUMAN_GAME] = {
                    "before_matchboxes": int(origin.get("matchboxes", 0)),
                    "after_matchboxes": len(menace.matchboxes),
                    "before_beads": int(origin.get("beads", 0)),
                    "after_beads": sum(box.total_beads() for box in menace.matchboxes.values()),
                    "practice_added": int(menace.games_played) - int(
                        human_snapshot.get("practice_games", 0)
                    ),
                }

            # Save this training batch as its own numbered evidence snapshot.
            batch_snapshot = self._save_isolated_training_batch(
                menace=menace,
                before_beads=before_batch_beads,
                summary=report.final_summary,
                requested_games=games,
            )

            # Keep the main trained model and its CSV evidence cumulative.
            self._save_learning_bundle(show_message=False)

            if status is not None:
                status.write("🏆 Counting wins, losses and draws.")
                status.write("⭐ Practice finished — MENACE has updated its memory.")
                status.update(
                    label=f"Practice complete: {games:,} games played",
                    state="complete",
                    expanded=False,
                )

            total_games = int(menace.games_played)
            st.session_state[SessionKey.TRAINING_NOTICE] = (
                (
                    f"Practice finished. This round added **{games:,} games**. "
                    f"MENACE has now completed **{total_games:,} practice games "
                    "in total**."
                )
                if self._is_simple_mode()
                else (
                    f"Training completed with **{games:,} additional games**. "
                    f"Cumulative practice total: **{total_games:,} games**."
                )
            )

            if not self._is_simple_mode():
                st.caption(
                    "This batch was saved separately as "
                    f"`{batch_snapshot.name}`. It contains this batch's outcomes "
                    "and net bead changes only; `menace_model.json` remains the "
                    "cumulative model used for continued learning."
                )
                st.download_button(
                    label="Download Training CSV",
                    data=Trainer.evidence_frame(history).to_csv(index=False),
                    file_name="training_log.csv",
                    mime="text/csv",
                )
            return True
        except Exception as exc:
            st.error(
                "MENACE could not finish practising."
                if self._is_simple_mode()
                else f"Training failed: {exc}"
            )
            if not self._is_simple_mode():
                st.exception(exc)
            return False

    def _render_training_summary_panel(
        self,
        summary: dict[str, Any],
        history: pd.DataFrame,
    ) -> None:
        """Present the latest training run with learner-facing and detailed evidence views."""
        if not summary:
            return

        games = int(
            summary.get("games", summary.get("menace_games_played", 0)) or 0
        )
        menace_wins = int(
            summary.get("menace_wins", summary.get("wins", 0)) or 0
        )
        opponent_wins = int(
            summary.get(
                "opponent_wins",
                summary.get("random_wins", summary.get("losses", 0)),
            )
            or 0
        )
        draws = int(summary.get("draws", 0) or 0)
        win_rate = float(summary.get("win_rate", 0.0) or 0.0)
        loss_rate = float(summary.get("loss_rate", 0.0) or 0.0)
        draw_rate = float(summary.get("draw_rate", 0.0) or 0.0)
        matchboxes = int(summary.get("matchboxes", 0) or 0)
        total_beads = int(summary.get("total_beads", 0) or 0)
        context = st.session_state.get(SessionKey.LAST_TRAINING_CONTEXT)
        if context:
            batch_games = int(context.get("batch_games", games))
            previous_games = int(context.get("previous_games", 0))
            cumulative_games = int(
                context.get("total_games", previous_games + batch_games)
            )
            before_matchboxes = int(context.get("before_matchboxes", 0))
            after_matchboxes = int(context.get("after_matchboxes", matchboxes))
            before_beads = int(context.get("before_beads", 0))
            after_beads = int(context.get("after_beads", total_beads))
            if self._is_simple_mode():
                human_games = int(context.get("human_games", 0))
                st.info(
                    f"**This practice round:** {batch_games:,} games  |  "
                    f"**Before this round:** {previous_games:,} games  |  "
                    f"**MENACE's total practice:** {cumulative_games:,} games\n\n"
                    "The results below describe only the most recent practice "
                    "round. MENACE keeps everything learned in earlier rounds."
                )
                st.caption(
                    f"Shared learning history now: {human_games:,} human games + "
                    f"{cumulative_games:,} practice games = "
                    f"{human_games + cumulative_games:,} games influencing memory. "
                    f"The outcomes below still total exactly {batch_games:,} because "
                    "they describe this practice round only."
                )
            else:
                st.info(
                    f"**Latest training batch:** {batch_games:,} games  |  "
                    f"**Prior cumulative total:** {previous_games:,} games  |  "
                    f"**New cumulative total:** {cumulative_games:,} games\n\n"
                    "The metrics and raw history below describe the latest batch; "
                    "the saved MENACE model retains cumulative learning."
                )

            st.markdown("### What this practice round changed")
            change_table = pd.DataFrame(
                [
                    {
                        "Memory measure": "Practice games",
                        "Before": previous_games,
                        "After": cumulative_games,
                        "Change": batch_games,
                    },
                    {
                        "Memory measure": "Learned matchboxes",
                        "Before": before_matchboxes,
                        "After": after_matchboxes,
                        "Change": after_matchboxes - before_matchboxes,
                    },
                    {
                        "Memory measure": "Beads in memory",
                        "Before": before_beads,
                        "After": after_beads,
                        "Change": after_beads - before_beads,
                    },
                ]
            )
            st.dataframe(change_table, width="stretch", hide_index=True)
            st.caption(
                "A positive bead change is the net result of rewards, draw "
                "rewards and loss penalties across this whole practice round."
            )

        if self._is_simple_mode():
            self._render_simple_training_results(
                games=games,
                menace_wins=menace_wins,
                opponent_wins=opponent_wins,
                draws=draws,
                matchboxes=matchboxes,
                total_beads=total_beads,
                history=history,
            )
            return

        opponent_name = str(summary.get("opponent_name", "Random Baseline"))
        st.markdown("### Latest Training Batch Interpretation")
        st.success(
            f"MENACE completed **{games}** training games against "
            f"**{opponent_name}**. It won **{menace_wins}** games "
            f"({win_rate:.1%}), lost **{opponent_wins}** ({loss_rate:.1%}), "
            f"and drew **{draws}** ({draw_rate:.1%}). During training it used "
            f"**{matchboxes}** matchboxes and accumulated "
            f"**{total_beads:,}** beads."
        )
        cols = st.columns(6)
        cols[0].metric("Games in This Batch", games)
        cols[1].metric("MENACE Wins", menace_wins, f"{win_rate:.1%}")
        cols[2].metric("Losses", opponent_wins, f"{loss_rate:.1%}")
        cols[3].metric("Draws", draws, f"{draw_rate:.1%}")
        cols[4].metric("Matchboxes", matchboxes)
        cols[5].metric("Total Beads", f"{total_beads:,}")
        st.caption(
            "These figures describe the latest training batch and provide report-ready evidence. Rates measure "
            "performance, while matchboxes and beads describe the learned representation."
        )
        with st.expander(
            "Show raw training data and configuration",
            expanded=False,
        ):
            st.write(summary)
            st.dataframe(history.tail(1000), width="stretch")

    def _validate_training_inputs(self, games: int, reward_win: int, reward_draw: int, penalty_loss: int) -> None:
        """Validate training values before creating the Trainer instance."""
        if games <= 0:
            raise ValueError('Training games must be greater than zero.')
        if reward_win < 0:
            raise ValueError('Reward for win must be non-negative.')
        if reward_draw < 0:
            raise ValueError('Reward for draw must be non-negative.')
        if penalty_loss > 0:
            raise ValueError('Penalty for loss should be zero or negative.')

    def _save_training_analysis(self, history: pd.DataFrame) -> None:
        """Save statistical analysis for the latest training run as CSV evidence."""
        tracker = StatisticsTracker(history)
        analysis = tracker.export_analysis_frame()
        self.config.training_analysis_path.parent.mkdir(parents=True, exist_ok=True)
        analysis.to_csv(self.config.training_analysis_path, index=False)
