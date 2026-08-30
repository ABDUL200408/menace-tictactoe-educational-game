"""
statistics.py

This file analyses the results produced by MENACE training and evaluation.
It converts raw experiment data into statistical summaries that support the
Streamlit interface.

The module computes performance measures such as win, loss, and draw rates,
moving averages, learning progress, and opponent comparisons. These results
are used to evaluate MENACE objectively and generate quantitative evidence.

This module is responsible only for statistical analysis. MENACE training is
implemented in trainer.py, while charts and figures are created in
visualisation.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd


REQUIRED_HISTORY_COLUMNS = {
    "game",
    "menace_wins",
    "draws",
    "win_rate",
}

OPTIONAL_OPPONENT_WIN_COLUMNS = {
    "opponent_wins",
    "random_wins",
}

REQUIRED_RATE_COLUMNS = {
    "win_rate",
    "loss_rate",
    "draw_rate",
}

REQUIRED_COMPARISON_COLUMNS = {
    "opponent_type",
    "opponent_name",
    "win_rate",
    "loss_rate",
    "draw_rate",
}


@dataclass(frozen=True)
class SummaryStatistics:
    """Stores the final performance summary of a single MENACE experiment."""

    total_games: int
    menace_wins: int
    opponent_wins: int
    draws: int
    win_rate: float
    loss_rate: float
    draw_rate: float
    final_matchboxes: Optional[int] = None
    final_total_beads: Optional[int] = None
    opponent_name: Optional[str] = None
    experiment_name: Optional[str] = None
    mode: Optional[str] = None

    def to_dict(self) -> Dict[str, int | float | str | None]:
        return {
            "total_games": self.total_games,
            "menace_wins": self.menace_wins,
            "opponent_wins": self.opponent_wins,
            "draws": self.draws,
            "win_rate": self.win_rate,
            "loss_rate": self.loss_rate,
            "draw_rate": self.draw_rate,
            "final_matchboxes": self.final_matchboxes,
            "final_total_beads": self.final_total_beads,
            "opponent_name": self.opponent_name,
            "experiment_name": self.experiment_name,
            "mode": self.mode,
        }


@dataclass(frozen=True)
class ComparisonStatistics:
    """Stores the win-rate comparison between two MENACE experiments."""

    baseline_label: str
    comparison_label: str
    baseline_win_rate: float
    comparison_win_rate: float
    absolute_improvement: float
    relative_improvement_percent: Optional[float]

    def to_dict(self) -> Dict[str, str | float | None]:
        return {
            "baseline_label": self.baseline_label,
            "comparison_label": self.comparison_label,
            "baseline_win_rate": self.baseline_win_rate,
            "comparison_win_rate": self.comparison_win_rate,
            "absolute_improvement": self.absolute_improvement,
            "relative_improvement_percent": self.relative_improvement_percent,
        }


@dataclass(frozen=True)
class OpponentComparisonSummary:
    """Summarises MENACE performance against one opponent type across repeated runs."""

    opponent_type: str
    runs: int
    mean_win_rate: float
    mean_loss_rate: float
    mean_draw_rate: float
    std_win_rate: float
    best_win_rate: float
    worst_win_rate: float

    def to_dict(self) -> Dict[str, str | int | float]:
        return {
            "opponent_type": self.opponent_type,
            "runs": self.runs,
            "mean_win_rate": self.mean_win_rate,
            "mean_loss_rate": self.mean_loss_rate,
            "mean_draw_rate": self.mean_draw_rate,
            "std_win_rate": self.std_win_rate,
            "best_win_rate": self.best_win_rate,
            "worst_win_rate": self.worst_win_rate,
        }


class StatisticsTracker:
    """
    Analyses MENACE training history and computes win rates, learning progress,
    stability measures, and reusable performance summaries.

    The tracker accepts experiment-history DataFrames and returns structured
    results for interface display, visualisation, reporting, and evaluation.
    """

    def __init__(self, history: pd.DataFrame) -> None:
        self.history = history.copy()
        self._validate_base_history()
        self._normalise_columns()
        self._validate_history()

    def summary(self) -> SummaryStatistics:
        """Compute final summary statistics."""
        final = self.history.iloc[-1]

        total_games = int(final["game"])
        menace_wins = int(final["menace_wins"])
        opponent_wins = int(final["opponent_wins"])
        draws = int(final["draws"])

        win_rate = float(final["win_rate"])
        loss_rate = float(final.get("loss_rate", opponent_wins / total_games))
        draw_rate = float(final.get("draw_rate", draws / total_games))

        final_matchboxes = (
            int(final["matchboxes"]) if "matchboxes" in self.history.columns else None
        )
        final_total_beads = (
            int(final["total_beads"]) if "total_beads" in self.history.columns else None
        )

        return SummaryStatistics(
            total_games=total_games,
            menace_wins=menace_wins,
            opponent_wins=opponent_wins,
            draws=draws,
            win_rate=win_rate,
            loss_rate=loss_rate,
            draw_rate=draw_rate,
            final_matchboxes=final_matchboxes,
            final_total_beads=final_total_beads,
            opponent_name=str(final["opponent_name"]) if "opponent_name" in self.history.columns else None,
            experiment_name=str(final["experiment_name"]) if "experiment_name" in self.history.columns else None,
            mode=str(final["mode"]) if "mode" in self.history.columns else None,
        )

    def add_moving_averages(self, window: int = 100) -> pd.DataFrame:
        """Add moving-average columns for smoother learning curves."""
        if window <= 0:
            raise ValueError("window must be greater than zero.")

        df = self.history.copy()
        actual_window = min(window, max(1, len(df)))

        for column in ["win_rate", "loss_rate", "draw_rate"]:
            if column in df.columns:
                df[f"{column}_ma"] = df[column].rolling(
                    window=actual_window,
                    min_periods=1,
                ).mean()

        return df

    def split_early_late(self, fraction: float = 0.25) -> pd.DataFrame:
        """Compare disjoint early and late game groups.

        History stores cumulative totals and cumulative rates.  Averaging those
        rates does not give the outcome rate inside a period.  Convert the
        cumulative counters to per-game outcomes first, then calculate each
        group's rate from the games that actually belong to that group.
        """
        if not 0 < fraction <= 0.5:
            raise ValueError("fraction must be greater than 0 and at most 0.5.")

        count = max(1, int(len(self.history) * fraction))
        counters = self.history[["menace_wins", "opponent_wins", "draws"]].copy()
        increments = counters.diff()
        increments.iloc[0] = counters.iloc[0]
        increments = increments.clip(lower=0)

        def phase_row(label: str, rows: pd.DataFrame) -> Dict[str, float | str]:
            wins = float(rows["menace_wins"].sum())
            losses = float(rows["opponent_wins"].sum())
            draws = float(rows["draws"].sum())
            games = int(wins + losses + draws)
            if games <= 0:
                raise ValueError(f"No game outcomes found in the {label} phase.")
            return {
                "phase": label,
                "games": games,
                "menace_wins": int(wins),
                "opponent_wins": int(losses),
                "draws": int(draws),
                "mean_win_rate": wins / games,
                "mean_loss_rate": losses / games,
                "mean_draw_rate": draws / games,
                "min_win_rate": wins / games,
                "max_win_rate": wins / games,
            }

        early = increments.head(count)
        late = increments.tail(count)

        return pd.DataFrame(
            [
                phase_row("early", early),
                phase_row("late", late),
            ]
        )

    def learning_gain(self, fraction: float = 0.25) -> ComparisonStatistics:
        """Estimate learning gain between early and late training."""
        phases = self.split_early_late(fraction=fraction)

        early_rate = float(
            phases.loc[phases["phase"] == "early", "mean_win_rate"].iloc[0]
        )
        late_rate = float(
            phases.loc[phases["phase"] == "late", "mean_win_rate"].iloc[0]
        )

        absolute = late_rate - early_rate
        relative = None if early_rate == 0 else (absolute / early_rate) * 100

        return ComparisonStatistics(
            baseline_label="early_training",
            comparison_label="late_training",
            baseline_win_rate=early_rate,
            comparison_win_rate=late_rate,
            absolute_improvement=absolute,
            relative_improvement_percent=relative,
        )

    def rate_stability(self) -> Dict[str, float]:
        """
        Estimate stability of win/loss/draw rates.

        Lower standard deviation means more stable observed behaviour.
        """
        return {
            "std_win_rate": float(self.history["win_rate"].std(ddof=0)),
            "std_loss_rate": float(self.history["loss_rate"].std(ddof=0)),
            "std_draw_rate": float(self.history["draw_rate"].std(ddof=0)),
        }

    def educational_interpretation(self) -> str:
        """Produce plain-English interpretation of training results."""
        summary = self.summary()
        gain = self.learning_gain()
        stability = self.rate_stability()

        direction = "improved" if gain.absolute_improvement > 0 else "did not clearly improve"

        return (
            f"MENACE played {summary.total_games} games. "
            f"It won {summary.menace_wins}, lost {summary.opponent_wins}, "
            f"and drew {summary.draws}. The final win rate was "
            f"{summary.win_rate:.1%}. Comparing early and late training, "
            f"MENACE {direction}; the estimated absolute win-rate change was "
            f"{gain.absolute_improvement:.1%}. The win-rate standard deviation "
            f"was {stability['std_win_rate']:.3f}, indicating how noisy the "
            "learning curve was. This evidence supports critical discussion of "
            "whether reinforcement changed MENACE's behaviour."
        )

    def export_summary_frame(self) -> pd.DataFrame:
        """Return one-row DataFrame summary."""
        return pd.DataFrame([self.summary().to_dict()])

    def export_analysis_frame(self) -> pd.DataFrame:
        """Return one-row DataFrame with summary, learning gain, and stability."""
        summary = self.summary().to_dict()
        gain = self.learning_gain().to_dict()
        stability = self.rate_stability()

        return pd.DataFrame([{**summary, **gain, **stability}])

    def _validate_base_history(self) -> None:
        """
        Validate fields needed before derived columns can be calculated.

        This ensures invalid input raises a clear ValueError rather than an
        internal pandas KeyError.
        """
        if self.history.empty:
            raise ValueError("History DataFrame cannot be empty.")

        missing = REQUIRED_HISTORY_COLUMNS.difference(self.history.columns)
        if missing:
            raise ValueError(
                "History DataFrame is missing required columns: "
                + ", ".join(sorted(missing))
            )

        if not OPTIONAL_OPPONENT_WIN_COLUMNS.intersection(self.history.columns):
            raise ValueError(
                "History DataFrame must contain opponent_wins or random_wins."
            )

    def _normalise_columns(self) -> None:
        """Normalise supported Trainer column aliases to a consistent structure."""
        if "opponent_wins" not in self.history.columns and "random_wins" in self.history.columns:
            self.history["opponent_wins"] = self.history["random_wins"]

        if "random_wins" not in self.history.columns and "opponent_wins" in self.history.columns:
            self.history["random_wins"] = self.history["opponent_wins"]

        if "loss_rate" not in self.history.columns:
            self.history["loss_rate"] = (
                self.history["opponent_wins"] / self.history["game"]
            )

        if "draw_rate" not in self.history.columns:
            self.history["draw_rate"] = self.history["draws"] / self.history["game"]

    def _validate_history(self) -> None:
        if self.history.empty:
            raise ValueError("History DataFrame cannot be empty.")

        missing = REQUIRED_HISTORY_COLUMNS.difference(self.history.columns)
        if missing:
            raise ValueError(
                "History DataFrame is missing required columns: "
                + ", ".join(sorted(missing))
            )

        if not OPTIONAL_OPPONENT_WIN_COLUMNS.intersection(self.history.columns):
            raise ValueError(
                "History DataFrame must contain opponent_wins or random_wins."
            )

        if (self.history["game"] <= 0).any():
            raise ValueError("All game numbers must be positive.")

        for column in REQUIRED_RATE_COLUMNS:
            if column in self.history.columns:
                invalid = (self.history[column] < 0) | (self.history[column] > 1)
                if invalid.any():
                    raise ValueError(f"{column} values must be between 0 and 1.")


class ExperimentComparator:
    """Compares repeated experiment summaries and ranks their performance."""

    def __init__(self, summaries: pd.DataFrame) -> None:
        self.summaries = summaries.copy()
        self._validate_summaries()

    def aggregate(self) -> pd.DataFrame:
        """Compute aggregate statistics over repeated experiments."""
        return pd.DataFrame(
            [
                {
                    "runs": len(self.summaries),
                    "mean_win_rate": float(self.summaries["win_rate"].mean()),
                    "std_win_rate": float(self.summaries["win_rate"].std(ddof=0)),
                    "mean_draw_rate": float(self.summaries["draw_rate"].mean()),
                    "std_draw_rate": float(self.summaries["draw_rate"].std(ddof=0)),
                    "mean_loss_rate": float(self.summaries["loss_rate"].mean()),
                    "std_loss_rate": float(self.summaries["loss_rate"].std(ddof=0)),
                }
            ]
        )

    def rank_by_win_rate(self) -> pd.DataFrame:
        """Rank experiments by final MENACE win rate."""
        return self.summaries.sort_values(
            by="win_rate",
            ascending=False,
        ).reset_index(drop=True)

    def _validate_summaries(self) -> None:
        if self.summaries.empty:
            raise ValueError("Summaries DataFrame cannot be empty.")

        missing = REQUIRED_RATE_COLUMNS.difference(self.summaries.columns)
        if missing:
            raise ValueError(
                "Summaries DataFrame is missing required columns: "
                + ", ".join(sorted(missing))
            )


class OpponentComparator:
    """Compares MENACE performance across different opponent types."""

    def __init__(self, comparison_results: pd.DataFrame) -> None:
        self.results = comparison_results.copy()
        self._validate_results()

    def evaluation_rows(self) -> pd.DataFrame:
        """Return only evaluation rows, excluding training rows."""
        if "phase" in self.results.columns:
            rows = self.results[self.results["phase"] == "evaluation"].copy()
        else:
            rows = self.results.copy()

        if rows.empty:
            raise ValueError("No evaluation rows available.")

        return rows

    def aggregate_by_opponent(self) -> pd.DataFrame:
        """Aggregate MENACE performance by opponent type."""
        rows = self.evaluation_rows()

        grouped = rows.groupby("opponent_type", as_index=False).agg(
            runs=("win_rate", "count"),
            mean_win_rate=("win_rate", "mean"),
            mean_loss_rate=("loss_rate", "mean"),
            mean_draw_rate=("draw_rate", "mean"),
            std_win_rate=("win_rate", lambda s: float(s.std(ddof=0))),
            best_win_rate=("win_rate", "max"),
            worst_win_rate=("win_rate", "min"),
        )

        return grouped.sort_values(
            by="mean_win_rate",
            ascending=False,
        ).reset_index(drop=True)

    def ranking(self) -> pd.DataFrame:
        """Rank opponent types from easiest to hardest for MENACE."""
        aggregate = self.aggregate_by_opponent()
        aggregate["rank_for_menace"] = range(1, len(aggregate) + 1)
        return aggregate

    def best_and_worst(self) -> Dict[str, str | float]:
        """Return best and worst opponent types for MENACE."""
        ranking = self.ranking()

        best = ranking.iloc[0]
        worst = ranking.iloc[-1]

        return {
            "best_opponent_type": str(best["opponent_type"]),
            "best_mean_win_rate": float(best["mean_win_rate"]),
            "worst_opponent_type": str(worst["opponent_type"]),
            "worst_mean_win_rate": float(worst["mean_win_rate"]),
        }

    def educational_interpretation(self) -> str:
        """Produce a plain-English interpretation of opponent comparison results."""
        ranking = self.ranking()
        best_worst = self.best_and_worst()

        ranking_text = ", ".join(
            f"{row.opponent_type} ({row.mean_win_rate:.1%})"
            for row in ranking.itertuples()
        )

        return (
            "MENACE was evaluated against multiple opponent types. "
            f"From easiest to hardest based on MENACE mean win rate, the ranking was: "
            f"{ranking_text}. MENACE performed best against "
            f"{best_worst['best_opponent_type']} with a mean win rate of "
            f"{best_worst['best_mean_win_rate']:.1%}, and worst against "
            f"{best_worst['worst_opponent_type']} with a mean win rate of "
            f"{best_worst['worst_mean_win_rate']:.1%}. This supports critical "
            "comparison between reinforcement learning, rule-based AI, and "
            "search-based AI."
        )

    def _validate_results(self) -> None:
        if self.results.empty:
            raise ValueError("Comparison results cannot be empty.")

        missing = REQUIRED_COMPARISON_COLUMNS.difference(self.results.columns)
        if missing:
            raise ValueError(
                "Comparison results missing required columns: "
                + ", ".join(sorted(missing))
            )


def compare_summaries(
    baseline: SummaryStatistics,
    comparison: SummaryStatistics,
    baseline_label: str = "baseline",
    comparison_label: str = "comparison",
) -> ComparisonStatistics:
    """Compare two experiment summaries."""
    absolute = comparison.win_rate - baseline.win_rate
    relative = None if baseline.win_rate == 0 else (absolute / baseline.win_rate) * 100

    return ComparisonStatistics(
        baseline_label=baseline_label,
        comparison_label=comparison_label,
        baseline_win_rate=baseline.win_rate,
        comparison_win_rate=comparison.win_rate,
        absolute_improvement=absolute,
        relative_improvement_percent=relative,
    )


def build_history_from_results(
    outcomes: Iterable[str],
    mode: str = "manual",
) -> pd.DataFrame:
    """Build a Trainer-compatible history DataFrame from MENACE outcomes."""
    menace_wins = 0
    opponent_wins = 0
    draws = 0
    rows: List[Dict[str, int | float | str]] = []

    for game_number, outcome in enumerate(outcomes, start=1):
        if outcome == "win":
            menace_wins += 1
        elif outcome == "loss":
            opponent_wins += 1
        elif outcome == "draw":
            draws += 1
        else:
            raise ValueError("Outcome must be one of: win, loss, draw.")

        rows.append(
            {
                "game": game_number,
                "menace_wins": menace_wins,
                "opponent_wins": opponent_wins,
                "random_wins": opponent_wins,
                "draws": draws,
                "win_rate": menace_wins / game_number,
                "loss_rate": opponent_wins / game_number,
                "draw_rate": draws / game_number,
                "mode": mode,
            }
        )

    if not rows:
        raise ValueError("At least one outcome is required.")

    return pd.DataFrame(rows)
