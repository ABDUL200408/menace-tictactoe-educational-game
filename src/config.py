"""
config.py

Central configuration for the MENACE MSc project.

This module keeps project metadata, paths, learning settings, experiment
defaults, visualisation options, and user-interface settings in one place.
Centralising these values improves consistency, maintainability, and
reproducibility across Streamlit, command-line experiments, and tests.

The educational settings define learner-facing language, page labels,
visual guidance, and optional technical detail used across the interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class ProjectInfo:
    """Stores stable metadata describing the MSc project."""

    project_code: str = "AIS_AL_project5"
    project_title: str = (
        "An online game to teach how machines are trained "
        "to win noughts and crosses (MENACE)"
    )
    package_name: str = "menace_tictactoe"
    version: str = "1.2.0"
    author_label: str = "MSc Artificial Intelligence Student"
    academic_use: str = "MSc dissertation and viva demonstration"

    def validate(self) -> None:
        """Validate required project metadata."""
        values = {
            "project_code": self.project_code,
            "project_title": self.project_title,
            "package_name": self.package_name,
            "version": self.version,
        }
        for name, value in values.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} cannot be empty.")


@dataclass(frozen=True)
class PathConfig:
    """Defines paths used for models, results, figures, and documentation."""

    root_dir: Path = Path(".")
    src_dir: Path = Path("src")
    tests_dir: Path = Path("tests")

    saved_models_dir: Path = Path("saved_models")
    results_dir: Path = Path("results")
    figures_dir: Path = Path("results/figures")
    docs_dir: Path = Path("docs")
    screenshots_dir: Path = Path("docs/screenshots")

    default_model_file: Path = Path("saved_models/menace_model.json")

    training_log_file: Path = Path("results/training_log.csv")
    evaluation_log_file: Path = Path("results/evaluation_log.csv")
    comparison_results_file: Path = Path("results/comparison_results.csv")
    experiment_summary_file: Path = Path("results/experiment_summary.csv")
    training_analysis_file: Path = Path("results/training_analysis.csv")
    comparison_analysis_file: Path = Path("results/comparison_analysis.csv")

    user_guide_file: Path = Path("docs/user_guide.md")
    evidence_checklist_file: Path = Path("docs/evidence_checklist.md")

    def ensure_directories(self) -> None:
        """Create project output directories when they do not already exist."""
        for directory in (
            self.saved_models_dir,
            self.results_dir,
            self.figures_dir,
            self.docs_dir,
            self.screenshots_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        """Validate important file extensions."""
        if self.default_model_file.suffix.lower() != ".json":
            raise ValueError("default_model_file must be a JSON file.")

        csv_paths = (
            self.training_log_file,
            self.evaluation_log_file,
            self.comparison_results_file,
            self.experiment_summary_file,
            self.training_analysis_file,
            self.comparison_analysis_file,
        )
        for csv_path in csv_paths:
            if csv_path.suffix.lower() != ".csv":
                raise ValueError(f"{csv_path} must be a CSV file.")


@dataclass(frozen=True)
class BoardConfig:
    """Stores the fixed 3x3 board settings used by the project."""

    board_size: int = 9
    rows: int = 3
    columns: int = 3
    empty_cell: str = " "
    valid_marks: Tuple[str, str] = ("X", "O")

    def validate(self) -> None:
        """Validate board dimensions and marks."""
        if self.board_size != self.rows * self.columns:
            raise ValueError("board_size must equal rows * columns.")
        if self.board_size != 9 or self.rows != 3 or self.columns != 3:
            raise ValueError("This MENACE project expects a 3x3 board.")
        if self.empty_cell in self.valid_marks:
            raise ValueError("empty_cell cannot be a player mark.")
        if set(self.valid_marks) != {"X", "O"}:
            raise ValueError("valid_marks must contain exactly X and O.")


@dataclass(frozen=True)
class MENACEConfig:
    """Defines the default MENACE learning and bead settings."""

    mark: str = "O"
    name: str = "MENACE"
    initial_beads: int = 3
    minimum_beads: int = 1
    reward_win: int = 3
    reward_draw: int = 1
    penalty_loss: int = -1
    use_symmetry: bool = True
    random_seed: int | None = 42

    def validate(self) -> None:
        """Validate MENACE learning settings."""
        if self.mark not in {"X", "O"}:
            raise ValueError("MENACE mark must be X or O.")
        if not self.name.strip():
            raise ValueError("MENACE name cannot be empty.")
        if self.initial_beads < self.minimum_beads:
            raise ValueError("initial_beads must be >= minimum_beads.")
        if self.minimum_beads < 1:
            raise ValueError("minimum_beads must be at least 1.")
        if self.reward_win < 0:
            raise ValueError("reward_win must be non-negative.")
        if self.reward_draw < 0:
            raise ValueError("reward_draw must be non-negative.")
        if self.penalty_loss > 0:
            raise ValueError("penalty_loss must be zero or negative.")


@dataclass(frozen=True)
class TrainingDefaults:
    """Provides default settings for training and evaluation experiments."""

    default_games: int = 1000
    app_training_games: int = 1000
    max_app_training_games: int = 10000
    simple_mode_max_training_games: int = 5000

    evaluation_games: int = 1000
    comparison_training_games: int = 5000
    comparison_evaluation_games: int = 1000
    app_comparison_training_games: int = 200
    app_comparison_evaluation_games: int = 20
    max_app_comparison_evaluation_games: int = 500

    repeated_experiment_runs: int = 3
    max_app_repetitions: int = 5
    log_interval: int = 1

    def validate(self) -> None:
        """Validate training and evaluation values."""
        positive_values = {
            "default_games": self.default_games,
            "app_training_games": self.app_training_games,
            "max_app_training_games": self.max_app_training_games,
            "simple_mode_max_training_games": self.simple_mode_max_training_games,
            "evaluation_games": self.evaluation_games,
            "comparison_training_games": self.comparison_training_games,
            "comparison_evaluation_games": self.comparison_evaluation_games,
            "app_comparison_training_games": self.app_comparison_training_games,
            "app_comparison_evaluation_games": self.app_comparison_evaluation_games,
            "max_app_comparison_evaluation_games": (
                self.max_app_comparison_evaluation_games
            ),
            "repeated_experiment_runs": self.repeated_experiment_runs,
            "max_app_repetitions": self.max_app_repetitions,
            "log_interval": self.log_interval,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero.")

        if self.simple_mode_max_training_games > self.max_app_training_games:
            raise ValueError(
                "simple_mode_max_training_games must be <= max_app_training_games."
            )
        if self.max_app_training_games < self.app_training_games:
            raise ValueError(
                "max_app_training_games must be >= app_training_games."
            )
        if (
            self.max_app_comparison_evaluation_games
            < self.app_comparison_evaluation_games
        ):
            raise ValueError(
                "max_app_comparison_evaluation_games must be >= "
                "app_comparison_evaluation_games."
            )


@dataclass(frozen=True)
class VisualisationDefaults:
    """Stores chart, figure, and educational bead-display settings."""

    moving_average_window: int = 100
    title_prefix: str = "MENACE Training"
    export_html: bool = True
    export_png: bool = False
    figure_width: int = 1000
    figure_height: int = 600

    # Bead circles provide a direct visual representation of MENACE move weights.
    max_visible_beads_per_move: int = 20
    bead_circle_symbol: str = "●"
    selected_move_symbol: str = "★"
    board_square_labels: Tuple[str, ...] = (
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
    )

    def validate(self) -> None:
        """Validate technical and educational visualisation settings."""
        if self.moving_average_window <= 0:
            raise ValueError("moving_average_window must be greater than zero.")
        if self.figure_width <= 0:
            raise ValueError("figure_width must be greater than zero.")
        if self.figure_height <= 0:
            raise ValueError("figure_height must be greater than zero.")
        if self.max_visible_beads_per_move <= 0:
            raise ValueError(
                "max_visible_beads_per_move must be greater than zero."
            )
        if len(self.board_square_labels) != 9:
            raise ValueError("board_square_labels must contain nine labels.")
        if len(set(self.board_square_labels)) != 9:
            raise ValueError("board_square_labels must be unique.")
        if any(not label.strip() for label in self.board_square_labels):
            raise ValueError("board_square_labels cannot contain empty values.")


@dataclass(frozen=True)
class EducationalDefaults:
    """Defines learner-facing language, guidance, interface labels, and board labels."""

    target_audience: str = "Learners aged approximately 14–16 and non-specialists"
    target_reading_age_min: int = 10
    target_reading_age_max: int = 12

    simple_mode_name: str = "Simple"
    technical_mode_name: str = "Technical"
    default_mode: str = "Simple"

    square_labels: Tuple[str, ...] = (
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
    )

    show_welcome_guide: bool = True
    show_step_by_step_guidance: bool = True
    show_move_toasts: bool = True
    show_plain_language_summary: bool = True
    show_technical_details_in_expander: bool = True

    welcome_title: str = "Welcome to MENACE"
    welcome_text: str = (
        "MENACE is like a child learning a new game. You choose a lettered "
        "square, MENACE chooses using coloured beads, and the bead groups "
        "change after each finished game."
    )
    practice_explanation: str = (
        "MENACE needs practice, just like a person learning a new game. "
        "It plays practice games by itself and remembers which choices helped."
    )
    symmetry_explanation: str = (
        "With use_symmetry enabled, MENACE treats boards that differ only by a "
        "rotation or reflection as the same pattern. It converts them to one "
        "canonical state and shares one matchbox, so learning from one orientation "
        "also improves every equivalent orientation."
    )

    simple_page_labels: Tuple[str, ...] = (
        "play",
        "train",
        "compare",
        "result and learning",
    )
    technical_page_labels: Tuple[str, ...] = (
        "play",
        "train",
        "compare",
        "result and learning",
    )

    def validate(self) -> None:
        """Validate educational interface settings."""
        if self.target_reading_age_min <= 0:
            raise ValueError("target_reading_age_min must be positive.")
        if self.target_reading_age_max < self.target_reading_age_min:
            raise ValueError(
                "target_reading_age_max must be >= target_reading_age_min."
            )
        valid_modes = {self.simple_mode_name, self.technical_mode_name}
        if self.default_mode not in valid_modes:
            raise ValueError(
                "default_mode must match simple_mode_name or technical_mode_name."
            )
        if len(self.square_labels) != 9:
            raise ValueError("square_labels must contain nine labels.")
        if len(set(self.square_labels)) != 9:
            raise ValueError("square_labels must be unique.")
        if len(self.simple_page_labels) != 4:
            raise ValueError("simple_page_labels must contain four page names.")
        if len(self.technical_page_labels) != 4:
            raise ValueError("technical_page_labels must contain four page names.")

    def square_label(self, move: int) -> str:
        """Return the learner-facing A–I label for an internal move index."""
        if not isinstance(move, int) or isinstance(move, bool):
            raise TypeError("move must be an integer.")
        if not 0 <= move < len(self.square_labels):
            raise ValueError("move must be between 0 and 8.")
        return self.square_labels[move]

    def internal_index(self, label: str) -> int:
        """Return the internal 0–8 index for a learner-facing square label."""
        if not isinstance(label, str) or not label.strip():
            raise ValueError("label cannot be empty.")
        normalised = label.strip().upper()
        try:
            return self.square_labels.index(normalised)
        except ValueError as exc:
            raise ValueError(
                f"Unknown square label {label!r}; expected A to I."
            ) from exc

@dataclass(frozen=True)
class AppDefaults:
    """Defines Streamlit layout, player marks, and display behaviour."""

    page_title: str = "MENACE: Learn by Playing"
    page_icon: str = "🎲"
    layout: str = "wide"
    app_title: str = "MENACE Learning Studio"

    default_human_mark: str = "X"
    default_menace_mark: str = "O"

    default_interface_mode: str = "Simple"
    show_internal_board_indexes_in_simple_mode: bool = False
    show_internal_board_indexes_in_technical_mode: bool = True
    show_matchbox_visualisation: bool = True
    show_reward_feedback: bool = True
    highlight_latest_menace_move: bool = True
    hide_dissertation_language_from_simple_mode: bool = True

    def validate(self) -> None:
        """Validate Streamlit application defaults."""
        if self.layout not in {"centered", "wide"}:
            raise ValueError("layout must be either centered or wide.")
        if self.default_human_mark == self.default_menace_mark:
            raise ValueError("Human and MENACE marks must be different.")
        if {self.default_human_mark, self.default_menace_mark} != {"X", "O"}:
            raise ValueError("Human and MENACE marks must be X and O.")
        if self.default_interface_mode not in {"Simple", "Technical"}:
            raise ValueError(
                "default_interface_mode must be Simple or Technical."
            )


@dataclass(frozen=True)
class ExperimentNames:
    """Provides consistent labels for training and evaluation experiments."""

    cli_training: str = "menace_cli_training"
    cli_evaluation: str = "menace_cli_evaluation"
    cli_comparison: str = "menace_cli_comparison"
    streamlit_training: str = "streamlit_menace_vs_random"
    streamlit_comparison: str = "streamlit_menace_ai_comparison"

    def validate(self) -> None:
        """Validate experiment labels."""
        for name, value in self.__dict__.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Experiment name {name!r} cannot be empty.")


@dataclass(frozen=True)
class SecurityConfig:
    """Defines practical limits for experiments and generated files."""

    max_training_games_cli: int = 1_000_000
    max_evaluation_games_cli: int = 1_000_000
    max_json_file_size_mb: int = 50
    max_csv_file_size_mb: int = 100

    def validate(self) -> None:
        """Validate practical safety limits."""
        positive_values = {
            "max_training_games_cli": self.max_training_games_cli,
            "max_evaluation_games_cli": self.max_evaluation_games_cli,
            "max_json_file_size_mb": self.max_json_file_size_mb,
            "max_csv_file_size_mb": self.max_csv_file_size_mb,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero.")


@dataclass(frozen=True)
class EvidenceConfig:
    """Lists expected output files, screenshots, and figures used as project evidence."""

    required_csv_outputs: Tuple[str, ...] = (
        "training_log.csv",
        "evaluation_log.csv",
        "comparison_results.csv",
        "experiment_summary.csv",
    )

    recommended_screenshots: Tuple[str, ...] = (
        "play_vs_menace_simple_mode.png",
        "play_vs_menace_technical_mode.png",
        "menace_decision_explanation.png",
        "educational_matchbox_beads.png",
        "training_dashboard.png",
        "learning_curve.png",
        "ai_comparison_results.png",
        "saved_model_json.png",
        "pytest_results.png",
    )

    recommended_figures: Tuple[str, ...] = (
        "learning_curve.html",
        "rolling_learning_curve.html",
        "outcome_rates.html",
        "early_late_comparison.html",
        "comparison_rates.html",
        "comparison_win_rate.html",
    )

    def validate(self) -> None:
        """Validate evidence output names."""
        if not self.required_csv_outputs:
            raise ValueError("At least one required CSV output must be listed.")
        for filename in self.required_csv_outputs:
            if not filename.endswith(".csv"):
                raise ValueError(f"{filename} must end with .csv.")


@dataclass(frozen=True)
class ProjectConfig:
    """Combines all project configuration sections into one shared object."""

    project: ProjectInfo = ProjectInfo()
    paths: PathConfig = PathConfig()
    board: BoardConfig = BoardConfig()
    menace: MENACEConfig = MENACEConfig()
    training: TrainingDefaults = TrainingDefaults()
    visualisation: VisualisationDefaults = VisualisationDefaults()
    education: EducationalDefaults = EducationalDefaults()
    app: AppDefaults = AppDefaults()
    experiments: ExperimentNames = ExperimentNames()
    security: SecurityConfig = SecurityConfig()
    evidence: EvidenceConfig = EvidenceConfig()

    def validate(self) -> None:
        """Validate all nested configuration sections."""
        self.project.validate()
        self.paths.validate()
        self.board.validate()
        self.menace.validate()
        self.training.validate()
        self.visualisation.validate()
        self.education.validate()
        self.app.validate()
        self.experiments.validate()
        self.security.validate()
        self.evidence.validate()

        if self.app.default_interface_mode != self.education.default_mode:
            raise ValueError(
                "AppDefaults.default_interface_mode and "
                "EducationalDefaults.default_mode must match."
            )
        if (
            self.visualisation.board_square_labels
            != self.education.square_labels
        ):
            raise ValueError(
                "Visualisation and education square labels must match."
            )

    def prepare_environment(self) -> None:
        """Validate configuration and create output folders."""
        self.validate()
        self.paths.ensure_directories()

    def evidence_paths(self) -> dict[str, Path]:
        """Return the key output paths used as dissertation evidence."""
        return {
            "model": self.paths.default_model_file,
            "training_log": self.paths.training_log_file,
            "evaluation_log": self.paths.evaluation_log_file,
            "comparison_results": self.paths.comparison_results_file,
            "experiment_summary": self.paths.experiment_summary_file,
            "training_analysis": self.paths.training_analysis_file,
            "comparison_analysis": self.paths.comparison_analysis_file,
            "figures_dir": self.paths.figures_dir,
            "screenshots_dir": self.paths.screenshots_dir,
        }


DEFAULT_CONFIG = ProjectConfig()
