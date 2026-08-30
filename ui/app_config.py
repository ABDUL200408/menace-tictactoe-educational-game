"""Streamlit-facing configuration for the MENACE interface."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from src.config import DEFAULT_CONFIG, ProjectConfig

@dataclass(frozen=True)
class AppConfig:
    """
    Streamlit-facing view of the central project configuration.

    Values are sourced from src.config.DEFAULT_CONFIG so the interface does not
    duplicate paths, learning defaults, educational labels, or display settings.
    The flattened properties provide a consistent interface for UI components
    and tests.
    """

    project_config: ProjectConfig = DEFAULT_CONFIG

    @property
    def app_title(self) -> str:
        return self.project_config.app.app_title

    @property
    def page_title(self) -> str:
        return self.project_config.app.page_title

    @property
    def page_icon(self) -> str:
        return self.project_config.app.page_icon

    @property
    def layout(self) -> str:
        return self.project_config.app.layout

    @property
    def model_path(self) -> Path:
        return self.project_config.paths.default_model_file

    @property
    def training_log_path(self) -> Path:
        return self.project_config.paths.training_log_file

    @property
    def comparison_results_path(self) -> Path:
        return self.project_config.paths.comparison_results_file

    @property
    def experiment_summary_path(self) -> Path:
        return self.project_config.paths.experiment_summary_file

    @property
    def training_analysis_path(self) -> Path:
        return self.project_config.paths.training_analysis_file

    @property
    def comparison_analysis_path(self) -> Path:
        return self.project_config.paths.comparison_analysis_file

    @property
    def figure_output_dir(self) -> Path:
        return self.project_config.paths.figures_dir

    @property
    def default_training_games(self) -> int:
        return self.project_config.training.app_training_games

    @property
    def max_training_games(self) -> int:
        return self.project_config.training.max_app_training_games

    @property
    def default_comparison_training_games(self) -> int:
        return self.project_config.training.app_comparison_training_games

    @property
    def default_comparison_evaluation_games(self) -> int:
        return self.project_config.training.app_comparison_evaluation_games

    @property
    def max_comparison_evaluation_games(self) -> int:
        return self.project_config.training.max_app_comparison_evaluation_games

    @property
    def max_comparison_repetitions(self) -> int:
        return self.project_config.training.max_app_repetitions

    @property
    def random_seed(self) -> int | None:
        return self.project_config.menace.random_seed

    @property
    def reward_win_default(self) -> int:
        return self.project_config.menace.reward_win

    @property
    def reward_draw_default(self) -> int:
        return self.project_config.menace.reward_draw

    @property
    def penalty_loss_default(self) -> int:
        return self.project_config.menace.penalty_loss

    @property
    def use_symmetry(self) -> bool:
        return self.project_config.menace.use_symmetry

    @property
    def default_display_mode(self) -> str:
        return self.project_config.education.default_mode

    @property
    def simple_mode_name(self) -> str:
        return self.project_config.education.simple_mode_name

    @property
    def technical_mode_name(self) -> str:
        return self.project_config.education.technical_mode_name

    @property
    def square_labels(self) -> tuple[str, ...]:
        return self.project_config.education.square_labels

    @property
    def simple_page_labels(self) -> tuple[str, ...]:
        return self.project_config.education.simple_page_labels

    @property
    def technical_page_labels(self) -> tuple[str, ...]:
        return self.project_config.education.technical_page_labels

    @property
    def max_visible_beads_per_move(self) -> int:
        return self.project_config.visualisation.max_visible_beads_per_move

    @property
    def welcome_title(self) -> str:
        return self.project_config.education.welcome_title

    @property
    def welcome_text(self) -> str:
        return self.project_config.education.welcome_text

    @property
    def practice_explanation(self) -> str:
        return self.project_config.education.practice_explanation

    @property
    def symmetry_explanation(self) -> str:
        return self.project_config.education.symmetry_explanation

    @property
    def streamlit_training_experiment_name(self) -> str:
        return self.project_config.experiments.streamlit_training

    @property
    def streamlit_comparison_experiment_name(self) -> str:
        return self.project_config.experiments.streamlit_comparison


    def with_isolated_paths(self, participant_label: str) -> "AppConfig":
        """Return an app configuration with per-participant persistence paths."""
        import re

        clean_label = str(participant_label).strip().lower()
        if not re.fullmatch(r"user_[0-9]{2,}_[0-9a-f]{4}", clean_label):
            raise ValueError("participant_label must match user_01_a3f9 style.")

        paths = self.project_config.paths
        session_saved_models = paths.saved_models_dir / "sessions" / clean_label
        session_results = paths.results_dir / "sessions" / clean_label
        session_figures = session_results / "figures"

        isolated_paths = replace(
            paths,
            saved_models_dir=session_saved_models,
            results_dir=session_results,
            figures_dir=session_figures,
            default_model_file=session_saved_models / paths.default_model_file.name,
            training_log_file=session_results / paths.training_log_file.name,
            evaluation_log_file=session_results / paths.evaluation_log_file.name,
            comparison_results_file=session_results / paths.comparison_results_file.name,
            experiment_summary_file=session_results / paths.experiment_summary_file.name,
            training_analysis_file=session_results / paths.training_analysis_file.name,
            comparison_analysis_file=session_results / paths.comparison_analysis_file.name,
        )
        return AppConfig(replace(self.project_config, paths=isolated_paths))

    def validate(self) -> None:
        """Validate the complete central configuration before the UI starts."""
        self.project_config.validate()
