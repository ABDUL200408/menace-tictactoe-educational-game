# MENACE Noughts and Crosses Educational AI Project

## Project Overview

**Project code:** AIS_AL_project5\
**Topic title:** *An online game to teach how machines are trained to
win noughts and crosses (MENACE)*

This project implements an interactive educational platform based on
**MENACE (Machine Educable Noughts And Crosses Engine)**. It
demonstrates reinforcement learning through matchboxes, coloured beads,
rewards, penalties, repeated practice, and observable changes in move
probabilities.

The Streamlit interface is organised into four learner-facing pages:

-   **Play** --- play against MENACE and inspect its decisions and
    learning.
-   **Train** --- give MENACE repeated practice games against a Random
    opponent.
-   **Compare** --- train a separate MENACE copy and evaluate it against
    Random, Heuristic, and Minimax opponents.
-   **Results & Learning** --- review measured learning evidence,
    charts, matchbox growth, and explanations of how MENACE learns.

The intended educational audience is novice learners aged approximately
**14--16**, while the generated evidence also supports technical
evaluation, reporting, and demonstration.

------------------------------------------------------------------------

## Key Features

-   Interactive Noughts and Crosses gameplay using learner-facing
    squares **A--I**.
-   MENACE matchboxes with weighted bead-based probabilistic move
    selection.
-   Reinforcement after completed games:
    -   win reward: **+3 beads**;
    -   draw reward: **+1 bead**;
    -   loss penalty: **−1 bead**, while preserving a minimum legal move
        weight.
-   Decision history so reinforcement is applied to the moves MENACE
    actually selected.
-   Symmetry reduction using rotations and reflections to map equivalent
    boards to one canonical state.
-   Lazy creation of matchboxes when a new canonical board state is
    encountered.
-   Human-vs-MENACE play with move history and decision explanations.
-   Shared MENACE learning memory across human play and practice
    training.
-   Practice training against **The Guesser (Random Player)**.
-   Evaluation against:
    -   **The Guesser (Random)**;
    -   **The Rule Follower (Heuristic)**;
    -   **The Careful Thinker (Minimax)**.
-   Comparison runs disable MENACE learning during evaluation.
-   Training statistics, learning curves, rolling averages,
    early-versus-late analysis, and matchbox-growth evidence.
-   JSON model persistence and CSV experimental evidence.
-   HTML and PNG figure export for report and viva evidence.
-   Light and dark interface themes.
-   Blue navigation, green primary actions, warning/destructive styling,
    visible keyboard focus indicators, and text labels that do not rely
    on colour alone.
-   **Start here** guidance and an interactive MENACE walkthrough for
    first-time users.
-   Automated Flesch Reading Ease and Flesch--Kincaid Grade Level
    auditing of learner-facing text.
-   Automated pytest suite covering game logic, MENACE learning,
    persistence, statistics, visualisation, themes, readability, and
    interface requirements.

------------------------------------------------------------------------

## Project Structure

``` text
menace_tictactoe_project/
├── .streamlit/
│   └── config.toml
├── app.py
├── main.py
├── README.md
├── requirements.txt
├── pytest.ini
├── src/
│   ├── board.py
│   ├── config.py
│   ├── game.py
│   ├── menace.py
│   ├── persistence.py
│   ├── players.py
│   ├── readability.py
│   ├── statistics.py
│   ├── trainer.py
│   └── visualisation.py
├── ui/
│   ├── __init__.py
│   ├── app_config.py
│   ├── common.py
│   ├── controller.py
│   ├── demos.py
│   ├── navigation.py
│   ├── persistence_controls.py
│   ├── session.py
│   ├── assets/
│   │   └── styles.css
│   └── pages/
│       ├── __init__.py
│       ├── play.py
│       ├── training.py
│       ├── comparison.py
│       ├── results.py
│       └── learning.py
├── tools/
│   └── readability_audit.py
├── tests/
│   ├── test_app_helpers.py
│   ├── test_board.py
│   ├── test_chart_theme_integration.py
│   ├── test_dual_theme.py
│   ├── test_game.py
│   ├── test_main.py
│   ├── test_menace.py
│   ├── test_persistence.py
│   ├── test_readability.py
│   ├── test_statistics.py
│   ├── test_trainer.py
│   ├── test_ui_regression.py
│   ├── test_ui_requirements.py
│   └── test_visualisation.py
├── results/
│   ├── training_log.csv
│   ├── training_analysis.csv
│   ├── comparison_results.csv
│   ├── comparison_analysis.csv
│   ├── readability_results.csv
│   └── figures/
├── saved_models/
│   └── menace_model.json
└── docs/
    └── screenshots/
```

Generated folders such as `.venv/`, `__pycache__/`, `.pytest_cache/`,
compiled Python files, and runtime backup files should not be committed
to the repository.

------------------------------------------------------------------------

## Architecture

The project separates the learner interface from the game and learning
logic.

### Core logic (`src/`)

-   **`board.py`** --- board representation, legal moves, win/draw
    detection, symmetry transformations, and canonicalisation.
-   **`game.py`** --- controls complete games, turns, move history, and
    results.
-   **`menace.py`** --- MENACE matchboxes, beads, weighted move
    selection, decision history, reinforcement, and model serialisation.
-   **`players.py`** --- Human, Random, FirstAvailable, Scripted,
    Heuristic, and Minimax players.
-   **`trainer.py`** --- repeated training, evaluation without learning,
    and comparative experiments.
-   **`statistics.py`** --- win/loss/draw summaries, rolling statistics,
    early-versus-late learning analysis, and opponent comparisons.
-   **`visualisation.py`** --- Plotly figures, matchbox/bead
    visualisations, and report evidence export.
-   **`persistence.py`** --- model and experiment persistence using JSON
    and CSV.
-   **`readability.py`** --- reproducible readability calculations.
-   **`config.py`** --- central project paths and default configuration.

### Interface (`ui/`)

`app.py` starts the Streamlit application. The `ui` package contains
navigation, session-state handling, persistence controls, shared
interface helpers, demonstrations, page renderers, and CSS styling.

### Command-line experiments (`main.py`)

`main.py` provides a reproducible command-line route for training,
evaluation, comparison, and result-file management without using the
Streamlit interface.

------------------------------------------------------------------------

## Installation

### 1. Clone or download the repository

Open a terminal in the project root directory.

### 2. Create a virtual environment

``` bash
python -m venv .venv
```

### 3. Activate the environment

**Windows PowerShell**

``` powershell
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt**

``` cmd
.venv\Scripts\activate.bat
```

**macOS / Linux**

``` bash
source .venv/bin/activate
```

### 4. Install dependencies

``` bash
python -m pip install -r requirements.txt
```

------------------------------------------------------------------------

## Run the Streamlit Application

From the project root:

``` bash
streamlit run app.py
```

Streamlit normally opens the application automatically. If necessary,
open the local address shown in the terminal, commonly:

``` text
http://localhost:8501
```

------------------------------------------------------------------------

## Using the Application

### Play

Play as **X** against MENACE as **O**. The page shows the live board,
move history, MENACE's selected matchbox and bead distribution, reward
or penalty after the result, shared learning memory, and
symmetry/canonical-state explanations.

**New Game** clears the current board only.\
**Continue previous learning** restores saved learning when available.\
**Start MENACE from the beginning** is a confirmed destructive reset of
the saved learning journey and generated evidence.

### Train

The Train page gives the active MENACE agent repeated practice games
against the Random opponent. It shows the latest practice batch
separately from human-play results and reports changes in practice
count, matchboxes, beads, wins, losses, and draws.

Heuristic and Minimax are not training opponents on this page; they are
reserved for evaluation on the Compare page.

### Compare

The Compare page creates a separate MENACE copy, trains it against
Random, then switches learning off and evaluates it against Random,
Heuristic, and Minimax. Each run records a seed because MENACE and
Random use stochastic choices, so repeated runs can produce different
results.

### Results & Learning

This page presents training evidence and educational explanations,
including learning curves, rolling performance, early-versus-late
comparison, outcome rates, matchbox growth, and the relationship between
matchboxes, beads, reinforcement, and future move probabilities.

------------------------------------------------------------------------

## Reproducible Command-Line Experiments

Show runner information:

``` bash
python main.py info
```

Train MENACE against Random:

``` bash
python main.py train --games 1000 --seed 42
```

Evaluate a saved MENACE model against Random without learning:

``` bash
python main.py evaluate --games 1000 --seed 42
```

Train and compare MENACE against Random, Heuristic, and Minimax:

``` bash
python main.py compare --trained-games 5000 --evaluation-games 1000 --repetitions 1 --seed 42
```

Delete generated CLI CSV result files:

``` bash
python main.py reset-results
```

Use `python main.py <command> --help` for the available options.

------------------------------------------------------------------------

## Evidence and Output Files

The project separates **persistent learning state** from **experimental
evidence**.

### Saved model

``` text
saved_models/menace_model.json
```

The JSON model stores MENACE's learned matchboxes, bead counts, and
configuration so learning can be continued later.

### CSV evidence

The main reproducible evidence files are:

``` text
results/training_log.csv
results/training_analysis.csv
results/comparison_results.csv
results/comparison_analysis.csv
results/readability_results.csv
```

Command-line evaluation may also generate:

``` text
results/evaluation_log.csv
results/experiment_summary.csv
```

### Figure evidence

Exported Plotly evidence is stored under:

``` text
results/figures/
```

HTML files preserve interactive charts. PNG export is supported through
Kaleido for report-ready figures.

The interface can also package practice or comparison evidence for
download. Exported evidence is for analysis, reporting, presentation,
and demonstration; MENACE does not read exported figures back into its
learning process.

------------------------------------------------------------------------

## Readability and Accessibility

The learner-facing interface is designed for self-guided use by novice
learners.

Implemented measures include:

-   visible keyboard focus indicators;
-   page navigation separated visually from primary actions;
-   red reserved for warning/destructive meaning;
-   labels and structure used alongside colour;
-   light and dark themes;
-   learner-facing A--I board labels rather than numeric board indexes;
-   a **Start here** guide and interactive demonstration;
-   plain-language explanations of matchboxes, beads, rewards,
    penalties, symmetry, and move probabilities.

The project also audits learner-facing text using **Flesch Reading
Ease** and **Flesch--Kincaid Grade Level**. The project quality target
is:

``` text
Flesch-Kincaid Grade Level <= 10.0
```

Run the audit from the project root:

``` bash
python tools/readability_audit.py
```

The results are written to:

``` text
results/readability_results.csv
```

Readability scores are treated as one quality indicator; they do not
prove comprehension or educational effectiveness.

------------------------------------------------------------------------

## Testing

Run the complete automated test suite:

``` bash
pytest
```

For verbose output:

``` bash
pytest -v
```

The suite contains 14 test files covering:

-   board rules and symmetry;
-   game flow;
-   MENACE learning and reinforcement;
-   training and evaluation;
-   model persistence;
-   statistics;
-   visualisation;
-   command-line experiment behaviour;
-   readability;
-   UI requirements and regression checks;
-   light/dark theme behaviour;
-   chart-theme integration.

To run an individual file:

``` bash
pytest tests/test_menace.py
```

------------------------------------------------------------------------

## MENACE Learning Model

MENACE does not calculate an optimal Tic-Tac-Toe strategy directly. For
each canonical board state, it stores a matchbox containing weighted
legal moves.

A move with more beads has a greater probability of being selected.
After a completed game, MENACE updates the bead counts for the moves
recorded in its decision history:

``` text
Win  -> +3 beads
Draw -> +1 bead
Loss -> -1 bead
```

The model uses symmetry so equivalent rotations and reflections can
share one canonical matchbox. This reduces duplicated states and makes
the learning process easier to inspect.

MENACE is intentionally simple and explainable. It can improve through
repeated experience but is not expected to match an optimal search
algorithm such as Minimax in every evaluation.

------------------------------------------------------------------------

## Reproducibility Notes

-   Training and command-line experiments support explicit random seeds.
-   Comparison runs record their seed.
-   Evaluation can be run with MENACE learning disabled.
-   JSON preserves the learned model.
-   CSV files preserve experimental evidence for later analysis.
-   Stochastic runs may differ when a different seed is used or when the
    interface starts a new randomly seeded comparison.

------------------------------------------------------------------------

## Technology

-   Python 3
-   Streamlit
-   pandas
-   NumPy
-   Plotly
-   Matplotlib
-   Kaleido
-   pytest
-   Pyphen

------------------------------------------------------------------------

## Project Context

This repository contains the software artefact for the project:

**An online game to teach how machines are trained to win noughts and
crosses (MENACE)**\
Manchester Metropolitan University
