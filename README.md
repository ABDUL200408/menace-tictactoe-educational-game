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
    -   win reward: **+3 beads**.
    -   draw reward: **+1 bead**.
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
│   └── sessions/
└── docs/
    └── screenshots/
```

------------------------------------------------------------------------

## Architecture

The project separates the core game and learning logic from the
Streamlit learner interface, persistence, evidence generation, and
automated testing.

### Root files

-   **`app.py`** --- main Streamlit entry point used to start the
    learner-facing application.
-   **`main.py`** --- command-line entry point for technical experiment
    and project utility operations.
-   **`requirements.txt`** --- lists the Python dependencies required by
    the project.
-   **`pytest.ini`** --- contains pytest configuration for the automated
    test suite.
-   **`README.md`** --- documents the project, installation,
    architecture, use, and testing.
-   **`.streamlit/config.toml`** --- contains Streamlit application
    configuration.

### Core logic (`src/`)

-   **`src/board.py`** --- board representation, legal moves, win/draw
    detection, symmetry transformations, and canonicalisation.
-   **`src/config.py`** --- central project paths, MENACE settings,
    training/evaluation defaults, visualisation defaults, and
    application configuration.
-   **`src/game.py`** --- complete game flow, turns, move records, board
    updates, and game results.
-   **`src/menace.py`** --- MENACE matchboxes, bead weights, weighted
    move selection, decision history, reinforcement, symmetry mappings,
    and model serialisation.
-   **`src/persistence.py`** --- saving and loading MENACE models and
    experiment data using JSON and CSV.
-   **`src/players.py`** --- Human, Random, FirstAvailable, Scripted,
    Heuristic, and Minimax player behaviours.
-   **`src/readability.py`** --- reproducible readability calculations
    for learner-facing text.
-   **`src/statistics.py`** --- win/loss/draw summaries, rolling
    statistics, early-versus-late analysis, and opponent comparisons.
-   **`src/trainer.py`** --- repeated MENACE training, evaluation
    without learning, and comparative experiments.
-   **`src/visualisation.py`** --- Plotly figures, matchbox/bead
    visualisations, and evidence export.

### Streamlit interface (`ui/`)

-   **`ui/__init__.py`** --- identifies the UI directory as a Python
    package.
-   **`ui/app_config.py`** --- adapts central project configuration for
    the Streamlit interface.
-   **`ui/common.py`** --- shared interface helpers and reusable UI
    functionality.
-   **`ui/controller.py`** --- coordinates application state,
    user/session identity, page rendering, MENACE state, and
    application-level interactions.
-   **`ui/demos.py`** --- interactive educational MENACE demonstration
    and walkthrough.
-   **`ui/navigation.py`** --- learner-facing page navigation and
    related behaviour.
-   **`ui/persistence_controls.py`** --- controls for continuing saved
    learning and starting MENACE again from the beginning.
-   **`ui/session.py`** --- Streamlit session-state keys and
    session-related state management.
-   **`ui/assets/styles.css`** --- interface styling for layout, themes,
    navigation, buttons, focus states, cards, and other visual
    components.

### Learner-facing pages (`ui/pages/`)

-   **`ui/pages/__init__.py`** --- identifies the pages directory as a
    Python package.
-   **`ui/pages/play.py`** --- Human-vs-MENACE gameplay, move history,
    decision explanations, bead information, reinforcement feedback, and
    learning persistence after completed games.
-   **`ui/pages/training.py`** --- repeated MENACE practice against
    Random and presentation of training outcomes and learning progress.
-   **`ui/pages/comparison.py`** --- separate-copy MENACE training
    against Random followed by evaluation, with learning disabled,
    against Random, Heuristic, and Minimax.
-   **`ui/pages/results.py`** --- measured training evidence,
    statistics, charts, exported evidence, and learning results.
-   **`ui/pages/learning.py`** --- learner-facing explanations and
    supporting educational content about MENACE learning.

### Tools (`tools/`)

-   **`tools/readability_audit.py`** --- runs the automated readability
    audit and writes readability evidence.

### Automated tests (`tests/`)

-   **`tests/test_app_helpers.py`** --- shared application/helper
    behaviour.
-   **`tests/test_board.py`** --- board rules, legal moves, outcomes,
    symmetry, and canonicalisation.
-   **`tests/test_chart_theme_integration.py`** --- chart rendering and
    theme integration.
-   **`tests/test_dual_theme.py`** --- light and dark theme behaviour.
-   **`tests/test_game.py`** --- game flow, turns, move history, and
    results.
-   **`tests/test_main.py`** --- command-line entry-point and
    experiment-runner behaviour.
-   **`tests/test_menace.py`** --- MENACE matchboxes, choices,
    decisions, reinforcement, symmetry, and model behaviour.
-   **`tests/test_persistence.py`** --- saving, loading, resetting, and
    persistence behaviour.
-   **`tests/test_readability.py`** --- readability calculations and
    requirements.
-   **`tests/test_statistics.py`** --- statistical summaries and
    learning-analysis calculations.
-   **`tests/test_trainer.py`** --- training, evaluation, and comparison
    workflows.
-   **`tests/test_ui_regression.py`** --- important interface regression
    checks.
-   **`tests/test_ui_requirements.py`** --- specified learner-facing UI
    requirements.
-   **`tests/test_visualisation.py`** --- visualisation and
    evidence-generation behaviour.

### Generated data and evidence

-   **`results/`** --- generated training, comparison, readability, and
    figure evidence.
-   **`results/figures/`** --- general exported figures where
    applicable.
-   **`results/sessions/`** --- runtime user/session-specific generated
    evidence where applicable.
-   **`saved_models/`** --- saved MENACE learning models.
-   **`saved_models/sessions/`** --- runtime user/session-specific
    MENACE models where applicable.
-   **`docs/screenshots/`** --- project screenshots and supporting
    documentation evidence.

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

```bash
pytest
```

For verbose output:

```bash
pytest -v
```

The suite contains **14 test files**. Together, they cover the main game logic, MENACE learning behaviour, training and evaluation workflows, persistence, statistics, visualisation, readability, command-line behaviour, interface requirements, regression checks, and theme integration.

The 14 test files are:

- **`tests/test_app_helpers.py`** — tests shared application and helper behaviour.
- **`tests/test_board.py`** — tests board rules, legal moves, wins, draws, symmetry transformations, and canonicalisation.
- **`tests/test_chart_theme_integration.py`** — tests integration between chart rendering and the application theme.
- **`tests/test_dual_theme.py`** — tests light and dark theme behaviour.
- **`tests/test_game.py`** — tests game flow, turns, move history, and game results.
- **`tests/test_main.py`** — tests command-line entry-point and experiment-runner behaviour.
- **`tests/test_menace.py`** — tests MENACE matchboxes, bead-based move selection, decision history, reinforcement, symmetry handling, and model behaviour.
- **`tests/test_persistence.py`** — tests saving, loading, resetting, and persistence-related behaviour.
- **`tests/test_readability.py`** — tests readability calculations and readability requirements.
- **`tests/test_statistics.py`** — tests statistical summaries, rates, rolling measures, and learning-analysis calculations.
- **`tests/test_trainer.py`** — tests training, evaluation, and comparison workflows.
- **`tests/test_ui_regression.py`** — checks important interface behaviour against regressions.
- **`tests/test_ui_requirements.py`** — verifies specified learner-facing UI requirements.
- **`tests/test_visualisation.py`** — tests visualisation and evidence-generation behaviour.

To run an individual test file:

```bash
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
