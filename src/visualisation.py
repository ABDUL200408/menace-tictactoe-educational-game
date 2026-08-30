"""
visualisation.py

Creates visual evidence for the MENACE MSc AI project.

This module converts training, evaluation, and comparison results into
interactive figures and textual summaries for the Streamlit interface.

It generates learning curves, outcome charts, AI comparison figures,
matchbox visualisations, probability charts, board heatmaps, and
report-ready HTML/PNG exports.

This module is responsible only for visualisation. MENACE training is
handled by trainer.py, statistical analysis by statistics.py, and game
logic by board.py and menace.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover
    px = None
    go = None
    PLOTLY_IMPORT_ERROR = exc
else:
    PLOTLY_IMPORT_ERROR = None

from src.menace import Matchbox


@dataclass(frozen=True)
class VisualisationConfig:
    """Configuration for visualisation output."""

    title_prefix: str = "MENACE Training"
    moving_average_window: int = 100
    figure_output_dir: Path = Path("results/figures")
    export_png_enabled: bool = False
    max_visible_beads_per_move: int = 12
    square_labels: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H", "I")

    def __post_init__(self) -> None:
        if not self.title_prefix.strip():
            raise ValueError("title_prefix cannot be empty.")
        if self.moving_average_window <= 0:
            raise ValueError("moving_average_window must be greater than zero.")
        if self.max_visible_beads_per_move <= 0:
            raise ValueError("max_visible_beads_per_move must be greater than zero.")
        if len(self.square_labels) != 9 or any(not str(label).strip() for label in self.square_labels):
            raise ValueError("square_labels must contain nine non-empty labels.")
        if len(set(self.square_labels)) != 9:
            raise ValueError("square_labels must be unique.")


@dataclass(frozen=True)
class VisualSummary:
    """Summary of one training/evaluation history."""

    total_games: int
    menace_wins: int
    opponent_wins: int
    draws: int
    win_rate: float
    loss_rate: float
    draw_rate: float

    def to_dict(self) -> Dict[str, int | float]: 
        """Return a report-ready dictionary."""
        return {
            "total_games": self.total_games,
            "menace_wins": self.menace_wins,
            "opponent_wins": self.opponent_wins,
            "draws": self.draws,
            "win_rate": self.win_rate,
            "loss_rate": self.loss_rate,
            "draw_rate": self.draw_rate,
        }


@dataclass(frozen=True)
class EducationalMoveVisual:
    """Learner-facing description of one move inside a MENACE matchbox."""

    matchbox_move: int
    board_square: int
    square_label: str
    beads: int
    probability: float
    selected: bool = False

    def to_dict(self) -> Dict[str, int | float | str | bool]:
        """Return a table-ready representation."""
        return {
            "matchbox_move": self.matchbox_move,
            "board_square": self.board_square,
            "square_label": self.square_label,
            "beads": self.beads,
            "probability": self.probability,
            "selected": self.selected,
        }


class VisualisationError(RuntimeError):
    """Raised when visualisation cannot be produced."""


class Visualiser:
    """
    Creates Plotly figures from MENACE training and comparison data.

    Visualisation is kept separate from experiment execution so the same
    figures and summaries can be reused across the interface, exports,
    evaluation, and reporting.
    """

    REQUIRED_HISTORY_COLUMNS = {
        "game",
        "menace_wins",
        "opponent_wins",
        "draws",
        "win_rate",
        "loss_rate",
        "draw_rate",
    }

    REQUIRED_COMPARISON_COLUMNS = {
        "phase",
        "opponent_name",
        "opponent_type",
        "win_rate",
        "loss_rate",
        "draw_rate",
    }

    RATE_COLUMNS = {"win_rate", "loss_rate", "draw_rate"}

    # Fixed colour palette used consistently across interactive figures and exports.
    # Explicit colours keep traces visually stable across Plotly rendering contexts.
    ACADEMIC_COLOURWAY = [
        "#1f77b4",  # blue
        "#d62728",  # red
        "#2ca02c",  # green
        "#ff7f0e",  # orange
        "#9467bd",  # purple
        "#17becf",  # cyan
        "#8c564b",  # brown
    ]

    RATE_COLOUR_MAP = {
        "win_rate": "#1f77b4",
        "win_rate_ma": "#1f77b4",
        "Win Rate": "#1f77b4",
        "Mean Win Rate": "#1f77b4",
        "MENACE Wins": "#1f77b4",
        "loss_rate": "#d62728",
        "loss_rate_ma": "#d62728",
        "Loss Rate": "#d62728",
        "Mean Loss Rate": "#d62728",
        "Opponent Wins": "#d62728",
        "draw_rate": "#2ca02c",
        "draw_rate_ma": "#2ca02c",
        "Draw Rate": "#2ca02c",
        "Mean Draw Rate": "#2ca02c",
        "Draws": "#2ca02c",
    }

    OPPONENT_COLOUR_MAP = {
        "Random Baseline": "#1f77b4",
        "Random": "#1f77b4",
        "Heuristic AI": "#ff7f0e",
        "Heuristic": "#ff7f0e",
        "Minimax AI": "#d62728",
        "Minimax": "#d62728",
    }

    def __init__(self, config: Optional[VisualisationConfig] = None) -> None:
        self.config = config or VisualisationConfig()
        self._ensure_plotly_available()

    # ==========================================================
    # Training history visualisations
    # ==========================================================

    def learning_curve(self, history: pd.DataFrame):
        """Create a line chart showing MENACE win/loss/draw rates over training."""
        history = self._normalise_history(history)

        fig = px.line(
            history,
            x="game",
            y=["win_rate", "loss_rate", "draw_rate"],
            title=f"{self.config.title_prefix}: Learning Curve",
            labels={
                "game": "Training Game",
                "value": "Rate",
                "variable": "Metric",
            },
            color_discrete_map=self.RATE_COLOUR_MAP,
        )
        fig.for_each_trace(
            lambda trace: trace.update(
                name={
                    "win_rate": "MENACE win rate",
                    "loss_rate": "Opponent win rate",
                    "draw_rate": "Draw rate",
                }.get(trace.name, trace.name)
            )
        )
        self._format_rate_figure(fig, legend_title="Metric")
        self._fit_training_axis(fig, history)
        return fig

    def rolling_learning_curve(self, history: pd.DataFrame):
        """Create a smoothed learning curve using rolling averages."""
        history = self._normalise_history(history)
        df = history.copy()
        window = min(self.config.moving_average_window, max(1, len(df)))

        for column in ["win_rate", "loss_rate", "draw_rate"]:
            df[f"{column}_ma"] = df[column].rolling(
                window=window,
                min_periods=1,
            ).mean()

        fig = px.line(
            df,
            x="game",
            y=["win_rate_ma", "loss_rate_ma", "draw_rate_ma"],
            title=f"{self.config.title_prefix}: Rolling Average (window={window})",
            labels={
                "game": "Training Game",
                "value": "Rate",
                "variable": "Metric",
            },
            color_discrete_map=self.RATE_COLOUR_MAP,
        )
        fig.for_each_trace(
            lambda trace: trace.update(
                name={
                    "win_rate_ma": "MENACE win rate",
                    "loss_rate_ma": "Opponent win rate",
                    "draw_rate_ma": "Draw rate",
                }.get(trace.name, trace.name)
            )
        )
        self._format_rate_figure(fig, legend_title="Metric")
        self._fit_training_axis(fig, df)
        return fig

    @staticmethod
    def _fit_training_axis(figure: object, history: pd.DataFrame) -> None:
        """Keep the complete game range and legend visible in narrow exports."""
        final_game = int(history["game"].max()) if not history.empty else 1
        figure.update_xaxes(range=[0, final_game], title_text="Practice game")
        figure.update_layout(
            autosize=True,
            margin={"l": 70, "r": 35, "t": 80, "b": 105},
            legend={
                "orientation": "h",
                "yanchor": "top",
                "y": -0.22,
                "xanchor": "center",
                "x": 0.5,
            },
        )

    def outcome_counts(self, history: pd.DataFrame):
        """Create a bar chart of final wins, losses, and draws."""
        summary = self._summary_from_history(history)
        data = pd.DataFrame(
            [
                {"Outcome": "MENACE Wins", "Count": summary.menace_wins},
                {"Outcome": "Opponent Wins", "Count": summary.opponent_wins},
                {"Outcome": "Draws", "Count": summary.draws},
            ]
        )

        fig = px.bar(
            data,
            x="Outcome",
            y="Count",
            color="Outcome",
            title=f"{self.config.title_prefix}: Final Outcome Counts",
            text="Count",
            color_discrete_map=self.RATE_COLOUR_MAP,
        )
        self._apply_academic_theme(fig)
        fig.update_layout(showlegend=False)
        return fig

    def outcome_rates(self, history: pd.DataFrame):
        """Create a bar chart of final win/loss/draw rates."""
        summary = self._summary_from_history(history)
        data = pd.DataFrame(
            [
                {"Outcome": "Win Rate", "Rate": summary.win_rate},
                {"Outcome": "Loss Rate", "Rate": summary.loss_rate},
                {"Outcome": "Draw Rate", "Rate": summary.draw_rate},
            ]
        )

        fig = px.bar(
            data,
            x="Outcome",
            y="Rate",
            color="Outcome",
            title=f"{self.config.title_prefix}: Final Outcome Rates",
            text=data["Rate"].map(lambda value: f"{value:.1%}"),
            color_discrete_map=self.RATE_COLOUR_MAP,
        )
        self._format_rate_figure(fig, legend_title="")
        fig.update_layout(showlegend=False)
        return fig

    def early_late_training_comparison(
        self,
        history: pd.DataFrame,
        fraction: float = 0.25,
    ):
        """Compare disjoint early and late training periods using actual outcomes."""
        history = self._normalise_history(history)

        if not 0 < fraction <= 0.5:
            raise ValueError("fraction must be greater than 0 and at most 0.5.")

        count = max(1, int(len(history) * fraction))

        counters = history[["menace_wins", "opponent_wins", "draws"]].copy()
        increments = counters.diff()
        increments.iloc[0] = counters.iloc[0]
        increments = increments.clip(lower=0)

        def phase_rates(rows: pd.DataFrame) -> Dict[str, float]:
            wins = float(rows["menace_wins"].sum())
            losses = float(rows["opponent_wins"].sum())
            draws = float(rows["draws"].sum())
            games = wins + losses + draws

            if games <= 0:
                raise ValueError("No game outcomes found in the selected training phase.")

            return {
                "Mean Win Rate": wins / games,
                "Mean Loss Rate": losses / games,
                "Mean Draw Rate": draws / games,
            }

        early_rates = phase_rates(increments.head(count))
        late_rates = phase_rates(increments.tail(count))

        data = pd.DataFrame(
            [
                {"Phase": "Early Training", **early_rates},
                {"Phase": "Late Training", **late_rates},
            ]
        )

        melted = data.melt(
            id_vars=["Phase"],
            var_name="Metric",
            value_name="Rate",
        )

        fig = px.bar(
            melted,
            x="Phase",
            y="Rate",
            color="Metric",
            barmode="group",
            title="Early vs Late MENACE Training Performance",
            text=melted["Rate"].map(lambda value: f"{value:.1%}"),
            color_discrete_map=self.RATE_COLOUR_MAP,
        )
        self._format_rate_figure(fig, legend_title="Metric")
        return fig

    def matchbox_growth(self, history: pd.DataFrame):
        """Show how many distinct canonical matchboxes MENACE has encountered."""
        history = self._normalise_history(history)
        if "matchboxes" not in history.columns:
            raise ValueError("Training history does not contain matchbox counts.")
        hover_data = {
            "matchboxes": True,
            "total_beads": True,
        } if "total_beads" in history.columns else {"matchboxes": True}
        fig = px.line(
            history,
            x="game",
            y="matchboxes",
            markers=True,
            title="Growth of MENACE's Canonical Matchbox Memory",
            labels={"game": "Training Game", "matchboxes": "Matchboxes"},
            hover_data=hover_data,
        )
        self._set_single_trace_colour(fig, "#9467bd")
        self._apply_academic_theme(fig)
        return fig

    def training_dashboard_figures(self, history: pd.DataFrame) -> Dict[str, object]:
        """Build the standard set of training dashboard figures."""
        figures = {
            "learning_curve": self.learning_curve(history),
            "rolling_learning_curve": self.rolling_learning_curve(history),
            "outcome_counts": self.outcome_counts(history),
            "outcome_rates": self.outcome_rates(history),
            "early_late_comparison": self.early_late_training_comparison(history),
        }
        if "matchboxes" in history.columns:
            figures["matchbox_growth"] = self.matchbox_growth(history)
        return figures

    # ==========================================================
    # Comparison visualisations
    # ==========================================================

    def comparison_rates(self, comparison_results: pd.DataFrame):
        """Compare MENACE win/loss/draw rates against each opponent type."""
        df = self._evaluation_rows(comparison_results)
        melted = df.melt(
            id_vars=["opponent_name", "opponent_type"],
            value_vars=["win_rate", "loss_rate", "draw_rate"],
            var_name="Metric",
            value_name="Rate",
        )

        fig = px.bar(
            melted,
            x="opponent_type",
            y="Rate",
            color="Metric",
            barmode="group",
            title="MENACE Performance Against Different Opponents",
            labels={
                "opponent_type": "Opponent Type",
                "Rate": "Rate",
            },
            text=melted["Rate"].map(lambda value: f"{value:.1%}"),
            color_discrete_map=self.RATE_COLOUR_MAP,
        )
        self._format_rate_figure(fig, legend_title="Metric")
        return fig

    def comparison_win_rate(self, comparison_results: pd.DataFrame):
        """Show MENACE win rate against each opponent type."""
        df = self._evaluation_rows(comparison_results)
        fig = px.bar(
            df,
            x="opponent_type",
            y="win_rate",
            color="opponent_name",
            title="MENACE Win Rate by Opponent",
            labels={
                "opponent_type": "Opponent Type",
                "win_rate": "MENACE Win Rate",
                "opponent_name": "Opponent",
            },
            text=df["win_rate"].map(lambda value: f"{value:.1%}"),
            color_discrete_map=self.OPPONENT_COLOUR_MAP,
        )
        self._format_rate_figure(fig, legend_title="Opponent")
        return fig

    def comparison_loss_rate(self, comparison_results: pd.DataFrame):
        """Show MENACE loss rate against each opponent type."""
        df = self._evaluation_rows(comparison_results)
        fig = px.bar(
            df,
            x="opponent_type",
            y="loss_rate",
            color="opponent_name",
            title="MENACE Loss Rate by Opponent",
            labels={
                "opponent_type": "Opponent Type",
                "loss_rate": "MENACE Loss Rate",
                "opponent_name": "Opponent",
            },
            text=df["loss_rate"].map(lambda value: f"{value:.1%}"),
            color_discrete_map=self.OPPONENT_COLOUR_MAP,
        )
        self._format_rate_figure(fig, legend_title="Opponent")
        return fig

    def comparison_summary_table(self, comparison_results: pd.DataFrame) -> pd.DataFrame:
        """Return a clean evaluation-only comparison summary table."""
        df = self._evaluation_rows(comparison_results)
        columns = [
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
        available = [column for column in columns if column in df.columns]
        return df[available].copy()

    def comparison_dashboard_figures(
        self,
        comparison_results: pd.DataFrame,
    ) -> Dict[str, object]:
        """Build the standard set of comparison dashboard figures."""
        return {
            "comparison_rates": self.comparison_rates(comparison_results),
            "comparison_win_rate": self.comparison_win_rate(comparison_results),
            "comparison_loss_rate": self.comparison_loss_rate(comparison_results),
        }

    # ==========================================================
    # Repeated experiment visualisations
    # ==========================================================

    def repeated_experiment_boxplot(self, summaries: pd.DataFrame):
        """Visualise variation across repeated MENACE experiments."""
        self._validate_rate_frame(summaries)
        melted = summaries.melt(
            value_vars=["win_rate", "draw_rate", "loss_rate"],
            var_name="Metric",
            value_name="Rate",
        )

        fig = px.box(
            melted,
            x="Metric",
            y="Rate",
            color="Metric",
            title="Variation Across Repeated MENACE Experiments",
            points="all",
            color_discrete_map=self.RATE_COLOUR_MAP,
        )
        self._format_rate_figure(fig, legend_title="")
        return fig

    def repeated_comparison_boxplot(self, comparison_results: pd.DataFrame):
        """Visualise repeated comparison variation by opponent type."""
        df = self._evaluation_rows(comparison_results)
        fig = px.box(
            df,
            x="opponent_type",
            y="win_rate",
            color="opponent_name",
            points="all",
            title="Repeated Comparison: MENACE Win Rate by Opponent",
            labels={
                "opponent_type": "Opponent Type",
                "win_rate": "MENACE Win Rate",
            },
            color_discrete_map=self.OPPONENT_COLOUR_MAP,
        )
        self._format_rate_figure(fig, legend_title="")
        return fig

    # ==========================================================
    # Matchbox visualisations
    # ==========================================================

    def educational_matchbox_rows(
        self,
        matchbox: Matchbox,
        move_to_board_square: Optional[Mapping[int, int]] = None,
        selected_move: Optional[int] = None,
        square_labels: Optional[Sequence[str]] = None,
    ) -> list[EducationalMoveVisual]:
        """Build simple move, bead, and probability rows for learners."""
        self._validate_matchbox(matchbox)
        labels = tuple(square_labels or self.config.square_labels)
        if len(labels) != 9:
            raise ValueError("square_labels must contain exactly nine labels.")

        probabilities = matchbox.probabilities()
        rows: list[EducationalMoveVisual] = []
        for move, beads in sorted(matchbox.beads.items()):
            move_int = int(move)
            board_square = int(
                move_to_board_square.get(move_int, move_int)
                if move_to_board_square is not None
                else move_int
            )
            if not 0 <= board_square <= 8:
                raise ValueError("Mapped board squares must be between 0 and 8.")
            rows.append(
                EducationalMoveVisual(
                    matchbox_move=move_int,
                    board_square=board_square,
                    square_label=str(labels[board_square]),
                    beads=int(beads),
                    probability=float(probabilities.get(move_int, 0.0)),
                    selected=move_int == selected_move,
                )
            )
        return rows

    def educational_matchbox_table(
        self,
        matchbox: Matchbox,
        move_to_board_square: Optional[Mapping[int, int]] = None,
        selected_move: Optional[int] = None,
        square_labels: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Return a child-friendly table for one matchbox decision."""
        rows = self.educational_matchbox_rows(
            matchbox=matchbox,
            move_to_board_square=move_to_board_square,
            selected_move=selected_move,
            square_labels=square_labels,
        )
        return pd.DataFrame(
            [
                {
                    "Square": row.square_label,
                    "Beads": row.beads,
                    "Chance": f"{row.probability:.1%}",
                    "Chosen": "Yes" if row.selected else "",
                }
                for row in rows
            ]
        )

    def educational_matchbox_html(
        self,
        matchbox: Matchbox,
        move_to_board_square: Optional[Mapping[int, int]] = None,
        selected_move: Optional[int] = None,
        square_labels: Optional[Sequence[str]] = None,
        title: str = "Look inside MENACE's matchbox",
        show_probabilities: bool = True,
    ) -> str:
        """Create matchbox-style HTML with circles representing beads.

        The HTML is intended for ``st.markdown(..., unsafe_allow_html=True)``.
        It complements, rather than replaces, the technical Plotly charts.
        """
        rows = self.educational_matchbox_rows(
            matchbox=matchbox,
            move_to_board_square=move_to_board_square,
            selected_move=selected_move,
            square_labels=square_labels,
        )
        cards: list[str] = []
        limit = self.config.max_visible_beads_per_move
        for row in rows:
            visible = min(row.beads, limit)
            circles = "".join(
                '<span class="menace-bead" aria-hidden="true"></span>'
                for _ in range(visible)
            )
            overflow = (
                f'<span class="bead-overflow">+{row.beads - visible}</span>'
                if row.beads > visible
                else ""
            )
            selected_class = " selected" if row.selected else ""
            selected_text = (
                '<div class="selected-label">MENACE chose this square</div>'
                if row.selected
                else ""
            )
            caption = (
                f"{row.beads} bead(s) · {row.probability:.1%} chance"
                if show_probabilities
                else f"{row.beads} bead(s)"
            )
            cards.append(
                f'<div class="move-card{selected_class}">'
                f'<div class="square-title">Square {escape(row.square_label)}</div>'
                f'<div class="bead-row" aria-label="{row.beads} beads">{circles}{overflow}</div>'
                f'<div class="move-caption">{caption}</div>'
                f'{selected_text}</div>'
            )

        safe_title = escape(title)
        cards_html = "".join(cards)
        return f"""
<div class="menace-matchbox" role="group" aria-label="{safe_title}">
  <div class="matchbox-lid">{safe_title}</div>
  <div class="matchbox-help">A bigger bead group means MENACE likes that square more.</div>
  <div class="move-grid">{cards_html}</div>
</div>
<style>
.menace-matchbox {{background:#eef8f0;border:4px solid #267a3f;border-radius:16px;padding:16px;box-shadow:0 5px 14px rgba(0,0,0,.12);margin:.5rem 0 1rem 0;}}
.matchbox-lid {{background:#dff2e4;color:#185a2c;border:1px solid #87bc96;border-radius:9px;padding:10px 14px;font-size:1.05rem;font-weight:700;text-align:center;}}
.matchbox-help {{color:#315c3d;text-align:center;margin:10px 0 14px 0;font-size:.92rem;}}
.move-grid {{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px;}}
.move-card {{background:#fbfffc;border:2px solid #9ecaaa;border-radius:12px;padding:10px;min-height:112px;}}
.move-card.selected {{border:3px solid #24834b;background:#eefaf2;}}
.square-title {{font-weight:700;color:#3e2b18;margin-bottom:6px;}}
.bead-row {{min-height:45px;line-height:1.1;}}
.menace-bead {{display:inline-block;width:16px;height:16px;border-radius:50%;margin:3px;background:#2f9e55;border:1px solid #176b34;box-shadow:inset 0 1px 2px rgba(255,255,255,.6);}}
.bead-overflow {{display:inline-block;margin-left:5px;font-weight:700;color:#185a2c;vertical-align:middle;}}
.move-caption {{font-size:.84rem;color:#51483e;margin-top:5px;}}
.selected-label {{margin-top:7px;font-size:.82rem;font-weight:700;color:#146c3a;}}
</style>
"""

    def educational_decision_summary(
        self,
        matchbox: Matchbox,
        selected_move: int,
        move_to_board_square: Optional[Mapping[int, int]] = None,
        square_labels: Optional[Sequence[str]] = None,
    ) -> str:
        """Explain one MENACE choice in clear learner-facing language."""
        rows = self.educational_matchbox_rows(
            matchbox=matchbox,
            move_to_board_square=move_to_board_square,
            selected_move=selected_move,
            square_labels=square_labels,
        )
        selected = next((row for row in rows if row.selected), None)
        if selected is None:
            raise ValueError("selected_move is not present in the matchbox.")
        equal = len({round(row.probability, 12) for row in rows}) == 1
        reason = (
            "Every available move had the same number of beads, so each had the same chance."
            if equal
            else "Past games changed the bead numbers, so some moves were more likely than others."
        )
        return (
            f"MENACE chose square {selected.square_label}. "
            f"That choice had {selected.beads} bead(s), giving it a "
            f"{selected.probability:.1%} chance. {reason}"
        )

    def matchbox_beads(self, matchbox: Matchbox):
        """Visualise bead counts in one MENACE matchbox."""
        self._validate_matchbox(matchbox)
        data = self._matchbox_rows(matchbox)

        fig = px.bar(
            data,
            x="Move",
            y="Beads",
            title=f"Matchbox Bead Counts for State: {matchbox.state}",
            text="Beads",
        )
        self._apply_academic_theme(fig)
        self._set_single_trace_colour(fig, "#1f77b4")
        fig.update_layout(showlegend=False)
        return fig

    def matchbox_probabilities(self, matchbox: Matchbox):
        """Visualise move probabilities in one MENACE matchbox."""
        self._validate_matchbox(matchbox)
        probabilities = matchbox.probabilities()
        data = pd.DataFrame(
            [
                {"Move": str(move), "Probability": probability}
                for move, probability in sorted(probabilities.items())
            ]
        )

        fig = px.bar(
            data,
            x="Move",
            y="Probability",
            title=f"Move Probabilities for State: {matchbox.state}",
            text=data["Probability"].map(lambda value: f"{value:.1%}"),
        )
        self._format_rate_figure(fig, legend_title="")
        self._set_single_trace_colour(fig, "#1f77b4")
        fig.update_layout(showlegend=False)
        return fig

    def matchbox_board_heatmap(self, matchbox: Matchbox):
        """Visualise bead counts as a 3x3 board heatmap."""
        self._validate_matchbox(matchbox)
        values = self._board_values_from_move_counts(matchbox.beads)

        fig = go.Figure(
            data=go.Heatmap(
                z=values,
                text=values,
                texttemplate="%{text}",
                showscale=True,
                colorscale="Blues",
            )
        )
        fig.update_layout(
            title=f"Matchbox Bead Heatmap for State: {matchbox.state}",
            xaxis_title="Column",
            yaxis_title="Row",
        )
        self._apply_academic_theme(fig)
        return fig

    # ==========================================================
    # Move history visualisations
    # ==========================================================

    def move_frequency_heatmap(self, move_rows: pd.DataFrame):
        """Visualise frequency of moves from a move-history table."""
        if move_rows.empty:
            raise ValueError("move_rows cannot be empty.")
        if "move" not in move_rows.columns:
            raise ValueError("move_rows must contain a 'move' column.")

        counts = move_rows["move"].value_counts().to_dict()
        values = self._board_values_from_move_counts(counts)

        fig = go.Figure(
            data=go.Heatmap(
                z=values,
                text=values,
                texttemplate="%{text}",
                showscale=True,
                colorscale="Blues",
            )
        )
        fig.update_layout(
            title="Move Frequency Heatmap",
            xaxis_title="Column",
            yaxis_title="Row",
        )
        self._apply_academic_theme(fig)
        return fig

    # ==========================================================
    # Export support
    # ==========================================================

    def export_figure_html(self, figure: object, filename: str) -> Path:
        """Export one Plotly figure as an interactive HTML file."""
        output_path = self._safe_output_path(filename, ".html")
        self._require_write_html(figure)
        self._apply_academic_theme(figure)
        figure.write_html(
            str(output_path),
            include_plotlyjs="cdn",
            full_html=True,
            config={
                "displaylogo": False,
                "responsive": True,
                "toImageButtonOptions": {"format": "png", "scale": 2},
            },
        )
        return output_path

    def export_figure_png(self, figure: object, filename: str) -> Path:
        """Export one Plotly figure as a PNG image using Kaleido."""
        output_path = self._safe_output_path(filename, ".png")
        self._require_write_image(figure)
        try:
            self._apply_academic_theme(figure)
            figure.write_image(str(output_path))
        except Exception as exc:  # pragma: no cover - depends on kaleido/browser
            raise VisualisationError(
                "PNG export failed. Install and configure kaleido if PNG evidence is required."
            ) from exc
        return output_path

    def export_figures_html(self, figures: Mapping[str, object]) -> Dict[str, Path]:
        """Export multiple Plotly figures as HTML files."""
        self._validate_figure_mapping(figures)
        return {
            name: self.export_figure_html(figure, name)
            for name, figure in figures.items()
        }

    def export_figures_png(self, figures: Mapping[str, object]) -> Dict[str, Path]:
        """Export multiple Plotly figures as PNG images."""
        self._validate_figure_mapping(figures)
        return {
            name: self.export_figure_png(figure, name)
            for name, figure in figures.items()
        }

    def export_training_dashboard_html(self, history: pd.DataFrame) -> Dict[str, Path]:
        """Export all standard training figures as HTML report evidence."""
        figures = self.training_dashboard_figures(history)
        return self.export_figures_html(figures)

    def export_training_evidence(self, history: pd.DataFrame) -> list[Path]:
        """Export training figures as an ordered list of HTML file paths.

        The dashboard exporter returns a name-to-path mapping; this adapter
        exposes the same files in list form without duplicating export logic.
        """
        return list(self.export_training_dashboard_html(history).values())

    def export_comparison_dashboard_html(
        self,
        comparison_results: pd.DataFrame,
    ) -> Dict[str, Path]:
        """Export all standard comparison figures as HTML report evidence."""
        figures = self.comparison_dashboard_figures(comparison_results)
        return self.export_figures_html(figures)

    def export_training_dashboard_png(self, history: pd.DataFrame) -> Dict[str, Path]:
        """Export all standard training figures as PNG report evidence."""
        figures = self.training_dashboard_figures(history)
        return self.export_figures_png(figures)

    def export_comparison_dashboard_png(
        self,
        comparison_results: pd.DataFrame,
    ) -> Dict[str, Path]:
        """Export all standard comparison figures as PNG report evidence."""
        figures = self.comparison_dashboard_figures(comparison_results)
        return self.export_figures_png(figures)

    # ==========================================================
    # Text summaries
    # ==========================================================

    def training_text_summary(self, history: pd.DataFrame) -> str:
        """Build an accessible textual explanation of training results."""
        history = self._normalise_history(history)
        summary = self._summary_from_history(history)

        first = history.iloc[0]
        last = history.iloc[-1]
        improvement = float(last["win_rate"]) - float(first["win_rate"])
        if improvement > 0:
            change_text = f"increased by {improvement:.1%}"
        elif improvement < 0:
            change_text = f"decreased by {abs(improvement):.1%}"
        else:
            change_text = "did not change"

        return (
            f"MENACE completed {summary.total_games} games. "
            f"It won {summary.menace_wins}, lost {summary.opponent_wins}, "
            f"and drew {summary.draws}. The final win rate reached "
            f"{summary.win_rate:.1%}, with a loss rate of {summary.loss_rate:.1%} "
            f"and a draw rate of {summary.draw_rate:.1%}. "
            f"From the first logged point to the final logged point, the win "
            f"rate {change_text}. "
            "This should be interpreted critically because MENACE is stochastic "
            "and performance depends on opponent type, training length, reward "
            "settings, and symmetry handling."
        )

    def comparison_text_summary(self, comparison_results: pd.DataFrame) -> str:
        """Build an accessible textual explanation of comparative AI results."""
        df = self._evaluation_rows(comparison_results)
        grouped = (
            df.groupby("opponent_type", as_index=False)["win_rate"]
            .mean()
            .sort_values(["win_rate", "opponent_type"], ascending=[False, True])
        )
        best_rate = float(grouped["win_rate"].max())
        worst_rate = float(grouped["win_rate"].min())
        best_names = grouped.loc[
            grouped["win_rate"] == best_rate, "opponent_type"
        ].astype(str).tolist()
        worst_names = grouped.loc[
            grouped["win_rate"] == worst_rate, "opponent_type"
        ].astype(str).tolist()
        best_label = ", ".join(best_names)
        worst_label = ", ".join(worst_names)
        minimax_note = (
            " Minimax is theoretically the strongest opponent because it uses optimal search."
            if any(name.lower() == "minimax" for name in worst_names)
            else ""
        )
        return (
            "The comparison experiment tested MENACE against different computer opponents. "
            f"Its highest mean win rate was against {best_label} ({best_rate:.1%}). "
            f"Its lowest mean win rate was against {worst_label} ({worst_rate:.1%})."
            f"{minimax_note} These results compare learning-based, rule-based, "
            "and search-based AI strategies."
        )

    def evidence_manifest(
        self,
        exported_paths: Mapping[str, Path],
        title: str = "MENACE visual evidence manifest",
    ) -> pd.DataFrame:
        """Create a table listing exported figure evidence for report traceability."""
        if not exported_paths:
            raise ValueError("exported_paths cannot be empty.")

        return pd.DataFrame(
            [
                {
                    "evidence_set": title,
                    "figure_name": name,
                    "path": str(path),
                    "file_type": path.suffix.lstrip("."),
                }
                for name, path in exported_paths.items()
            ]
        )

    # ==========================================================
    # Internal helpers
    # ==========================================================

    def _summary_from_history(self, history: pd.DataFrame) -> VisualSummary:
        """Create a VisualSummary object from training history."""
        history = self._normalise_history(history)
        final = history.iloc[-1]

        return VisualSummary(
            total_games=int(final["game"]),
            menace_wins=int(final["menace_wins"]),
            opponent_wins=int(final["opponent_wins"]),
            draws=int(final["draws"]),
            win_rate=float(final["win_rate"]),
            loss_rate=float(final["loss_rate"]),
            draw_rate=float(final["draw_rate"]),
        )

    def _evaluation_rows(self, comparison_results: pd.DataFrame) -> pd.DataFrame:
        """Return validated evaluation rows from comparison results."""
        self._validate_comparison(comparison_results)
        df = comparison_results.copy()

        if "phase" in df.columns:
            df = df[df["phase"] == "evaluation"]

        if df.empty:
            raise ValueError("No evaluation rows found in comparison results.")

        return df

    def _normalise_history(self, history: pd.DataFrame) -> pd.DataFrame:
        """Return a validated copy of Trainer history data."""
        if history.empty:
            raise ValueError("History DataFrame cannot be empty.")

        df = history.copy()

        if "opponent_wins" not in df.columns and "random_wins" in df.columns:
            df["opponent_wins"] = df["random_wins"]

        self._validate_history(df)
        return df

    def _validate_history(self, history: pd.DataFrame) -> None:
        """Validate Trainer history DataFrame before plotting."""
        if history.empty:
            raise ValueError("History DataFrame cannot be empty.")

        missing = self.REQUIRED_HISTORY_COLUMNS.difference(history.columns)
        if missing:
            raise ValueError(
                "History DataFrame missing required columns: "
                + ", ".join(sorted(missing))
            )

        if (history["game"] <= 0).any():
            raise ValueError("All game values must be positive.")

        self._validate_rate_frame(history)

        totals = history["menace_wins"] + history["opponent_wins"] + history["draws"]
        if (totals != history["game"]).any():
            raise ValueError("Outcome counts must sum to the game number.")

    def _validate_comparison(self, comparison_results: pd.DataFrame) -> None:
        """Validate ComparisonExperiment DataFrame before plotting."""
        if comparison_results.empty:
            raise ValueError("Comparison results cannot be empty.")

        missing = self.REQUIRED_COMPARISON_COLUMNS.difference(
            comparison_results.columns
        )
        if missing:
            raise ValueError(
                "Comparison results missing required columns: "
                + ", ".join(sorted(missing))
            )

        self._validate_rate_frame(comparison_results)

    def _validate_rate_frame(self, dataframe: pd.DataFrame) -> None:
        """Validate that rate columns exist and contain values between 0 and 1."""
        if dataframe.empty:
            raise ValueError("DataFrame cannot be empty.")

        missing = self.RATE_COLUMNS.difference(dataframe.columns)
        if missing:
            raise ValueError(
                "DataFrame missing rate columns: " + ", ".join(sorted(missing))
            )

        for column in self.RATE_COLUMNS:
            invalid = (dataframe[column] < 0) | (dataframe[column] > 1)
            if invalid.any():
                raise ValueError(f"{column} values must be between 0 and 1.")

    @staticmethod
    def _validate_matchbox(matchbox: Matchbox) -> None:
        """Validate a matchbox before visualising it."""
        if not isinstance(matchbox, Matchbox):
            raise TypeError("Expected a Matchbox instance.")
        if not matchbox.beads:
            raise ValueError("Cannot visualise a matchbox without beads.")

    @staticmethod
    def _matchbox_rows(matchbox: Matchbox) -> pd.DataFrame:
        """Return a DataFrame representation of matchbox bead counts."""
        return pd.DataFrame(
            [
                {"Move": str(move), "Beads": beads}
                for move, beads in sorted(matchbox.beads.items())
            ]
        )

    @staticmethod
    def _board_values_from_move_counts(counts: Mapping[int, int]) -> list[list[int]]:
        """Convert move-index counts into a 3x3 board matrix."""
        values = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for move, count in counts.items():
            move_int = int(move)
            if not 0 <= move_int <= 8:
                continue
            row, col = divmod(move_int, 3)
            values[row][col] = int(count)
        return values

    def _safe_output_path(self, filename: str, suffix: str) -> Path:
        """Create a safe output path inside the configured figures folder."""
        if not filename.strip():
            raise ValueError("filename cannot be empty.")

        cleaned = (
            filename.strip()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )
        if not cleaned.endswith(suffix):
            cleaned = f"{cleaned}{suffix}"

        self.config.figure_output_dir.mkdir(parents=True, exist_ok=True)
        return self.config.figure_output_dir / cleaned

    @staticmethod
    def _validate_figure_mapping(figures: Mapping[str, object]) -> None:
        """Validate a dictionary of figure names to Plotly figures."""
        if not figures:
            raise ValueError("figures cannot be empty.")
        for name, figure in figures.items():
            if not str(name).strip():
                raise ValueError("Figure name cannot be empty.")
            Visualiser._require_write_html(figure)

    @staticmethod
    def _require_write_html(figure: object) -> None:
        """Ensure the supplied figure supports HTML export."""
        if not hasattr(figure, "write_html"):
            raise TypeError("Expected a Plotly figure with write_html().")

    @staticmethod
    def _require_write_image(figure: object) -> None:
        """Ensure the supplied figure supports image export."""
        if not hasattr(figure, "write_image"):
            raise TypeError("Expected a Plotly figure with write_image().")

    def _format_rate_figure(self, figure: object, legend_title: str) -> None:
        """Apply consistent formatting and fixed colours to rate-based figures."""
        figure.update_yaxes(tickformat=".0%", range=[0, 1])
        self._apply_academic_theme(figure)
        self._apply_trace_colours(figure)
        figure.update_layout(
            legend_title_text=legend_title,
            hovermode="x unified",
        )

    def _apply_academic_theme(self, figure: object) -> None:
        """Apply a consistent light Plotly theme to interface and exported figures."""
        if not hasattr(figure, "update_layout"):
            raise TypeError("Expected a Plotly figure with update_layout().")

        figure.update_layout(
            template="plotly_white",
            colorway=self.ACADEMIC_COLOURWAY,
            font={
                "family": "Arial, Helvetica, sans-serif",
                "size": 13,
                "color": "#1f2937",
            },
            title_font={
                "size": 18,
                "color": "#111827",
            },
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin={
                "l": 70,
                "r": 70,
                "t": 80,
                "b": 70,
            },
        )

        figure.update_xaxes(
            showgrid=True,
            gridcolor="#e5e7eb",
            zerolinecolor="#d1d5db",
            linecolor="#9ca3af",
            tickfont={"color": "#374151"},
            title_font={"color": "#374151"},
        )
        figure.update_yaxes(
            showgrid=True,
            gridcolor="#e5e7eb",
            zerolinecolor="#d1d5db",
            linecolor="#9ca3af",
            tickfont={"color": "#374151"},
            title_font={"color": "#374151"},
        )

    def _apply_trace_colours(self, figure: object) -> None:
        """Force stable line/marker colours so exported HTML is not black-only."""
        for index, trace in enumerate(getattr(figure, "data", [])):
            trace_name = str(getattr(trace, "name", "") or "")
            colour = (
                self.RATE_COLOUR_MAP.get(trace_name)
                or self.OPPONENT_COLOUR_MAP.get(trace_name)
                or self.ACADEMIC_COLOURWAY[index % len(self.ACADEMIC_COLOURWAY)]
            )

            if hasattr(trace, "line"):
                trace.line.color = colour
                if getattr(trace.line, "width", None) in (None, 0):
                    trace.line.width = 3
            if hasattr(trace, "marker"):
                trace.marker.color = colour
                trace.marker.line = {
                    "color": "white",
                    "width": 0.5,
                }

    @staticmethod
    def _set_single_trace_colour(figure: object, colour: str) -> None:
        """Set a fixed colour for single-series bar charts."""
        for trace in getattr(figure, "data", []):
            if hasattr(trace, "marker"):
                trace.marker.color = colour
                trace.marker.line = {"color": "white", "width": 0.5}
            if hasattr(trace, "line"):
                trace.line.color = colour

    def _ensure_plotly_available(self) -> None:
        """Ensure Plotly is installed before figures are generated."""
        if PLOTLY_IMPORT_ERROR is not None:
            raise VisualisationError(
                "Plotly is required for visualisation. Install it with: pip install plotly"
            ) from PLOTLY_IMPORT_ERROR


# ==========================================================
# Public convenience helpers
# ==========================================================


def build_textual_visual_summary(history: pd.DataFrame) -> str:
    """Return a textual summary of MENACE training history."""
    visualiser = Visualiser()
    return visualiser.training_text_summary(history)


def build_comparison_textual_summary(comparison_results: pd.DataFrame) -> str:
    """Return a textual summary of MENACE comparison results."""
    visualiser = Visualiser()
    return visualiser.comparison_text_summary(comparison_results)


def export_training_dashboard_html(
    history: pd.DataFrame,
    output_dir: Path | str = Path("results/figures"),
) -> Dict[str, Path]:
    """Convenience helper for exporting training figures without instantiating Visualiser."""
    visualiser = Visualiser(
        VisualisationConfig(figure_output_dir=Path(output_dir))
    )
    return visualiser.export_training_dashboard_html(history)


def export_comparison_dashboard_html(
    comparison_results: pd.DataFrame,
    output_dir: Path | str = Path("results/figures"),
) -> Dict[str, Path]:
    """Convenience helper for exporting comparison figures without instantiating Visualiser."""
    visualiser = Visualiser(
        VisualisationConfig(figure_output_dir=Path(output_dir))
    )
    return visualiser.export_comparison_dashboard_html(comparison_results)


__all__ = [
    "EducationalMoveVisual",
    "VisualisationConfig",
    "VisualSummary",
    "VisualisationError",
    "Visualiser",
    "build_textual_visual_summary",
    "build_comparison_textual_summary",
    "export_training_dashboard_html",
    "export_comparison_dashboard_html",
]
