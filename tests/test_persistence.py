"""
test_persistence.py

Unit tests for src.persistence.

These tests verify that the MENACE persistence layer correctly saves and loads:

- trained MENACE models
- matchboxes and bead counts
- model metadata
- experiment CSV artefacts
- JSON evidence artefacts

This is important for the MSc project because saved models and exported
results support reproducibility, dissertation evidence, and viva demonstration.

Run:
    pytest tests/test_persistence.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.board import Board
from src.menace import MENACEPlayer
from src.persistence import ArtefactPersistence, ModelPersistence, PersistenceError


def make_trained_menace() -> MENACEPlayer:
    """Create a small trained MENACE model for persistence tests."""
    menace = MENACEPlayer(mark="O", seed=42, use_symmetry=True)
    board = Board.from_string("X        ")

    menace.choose_move(board)
    menace.learn_from_result("win")

    return menace


def test_model_persistence_save_creates_json_file(tmp_path: Path) -> None:
    menace = make_trained_menace()
    path = tmp_path / "menace_model.json"

    persistence = ModelPersistence(path)
    persistence.save(menace)

    assert path.exists()
    assert path.suffix == ".json"


def test_model_persistence_saved_file_contains_metadata_and_model(tmp_path: Path) -> None:
    menace = make_trained_menace()
    path = tmp_path / "menace_model.json"

    ModelPersistence(path).save(menace)

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert "metadata" in payload
    assert "model" in payload
    assert payload["metadata"]["model_type"] == "MENACE"
    assert payload["model"]["mark"] == "O"
    assert payload["model"]["games_played"] == 1
    assert isinstance(payload["model"]["matchboxes"], dict)


def test_model_persistence_round_trip_preserves_learning_state(tmp_path: Path) -> None:
    menace = make_trained_menace()
    path = tmp_path / "menace_model.json"

    persistence = ModelPersistence(path)
    persistence.save(menace)

    restored = persistence.load()

    assert restored.mark == menace.mark
    assert restored.name == menace.name
    assert restored.use_symmetry == menace.use_symmetry
    assert restored.games_played == menace.games_played
    assert restored.wins == menace.wins
    assert restored.losses == menace.losses
    assert restored.draws == menace.draws
    assert restored.matchboxes.keys() == menace.matchboxes.keys()
    assert restored.total_beads() == menace.total_beads()


def test_model_persistence_exists_reports_file_presence(tmp_path: Path) -> None:
    menace = make_trained_menace()
    path = tmp_path / "menace_model.json"

    persistence = ModelPersistence(path)

    assert persistence.exists() is False

    persistence.save(menace)

    assert persistence.exists() is True


def test_model_persistence_delete_removes_model_file(tmp_path: Path) -> None:
    menace = make_trained_menace()
    path = tmp_path / "menace_model.json"

    persistence = ModelPersistence(path)
    persistence.save(menace)

    assert path.exists()

    persistence.delete()

    assert not path.exists()


def test_model_persistence_backup_created_when_overwriting(tmp_path: Path) -> None:
    first = make_trained_menace()
    second = make_trained_menace()
    path = tmp_path / "menace_model.json"

    persistence = ModelPersistence(path)
    persistence.save(first, create_backup=False)
    persistence.save(second, create_backup=True)

    assert path.exists()
    assert persistence.backup_path.exists()


def test_model_persistence_delete_can_remove_backup(tmp_path: Path) -> None:
    menace = make_trained_menace()
    path = tmp_path / "menace_model.json"

    persistence = ModelPersistence(path)
    persistence.save(menace, create_backup=False)
    persistence.save(menace, create_backup=True)

    assert persistence.backup_path.exists()

    persistence.delete(delete_backup=True)

    assert not path.exists()
    assert not persistence.backup_path.exists()


def test_inspect_metadata_returns_model_metadata(tmp_path: Path) -> None:
    menace = make_trained_menace()
    path = tmp_path / "menace_model.json"

    persistence = ModelPersistence(path)
    persistence.save(
        menace,
        extra_metadata={
            "experiment": "unit_test",
            "seed": 42,
        },
    )

    metadata = persistence.inspect_metadata()

    assert metadata["model_type"] == "MENACE"
    assert metadata["project_code"] == "AIS_AL_project5"
    assert metadata["games_played"] == 1
    assert metadata["wins"] == 1
    assert metadata["matchboxes"] >= 1
    assert metadata["extra"]["experiment"] == "unit_test"
    assert metadata["extra"]["seed"] == 42


def test_inspect_model_summary_returns_report_ready_summary(tmp_path: Path) -> None:
    menace = make_trained_menace()
    path = tmp_path / "menace_model.json"

    persistence = ModelPersistence(path)
    persistence.save(menace)

    summary = persistence.inspect_model_summary()

    assert summary["path"] == str(path)
    assert summary["model_type"] == "MENACE"
    assert summary["mark"] == "O"
    assert summary["use_symmetry"] is True
    assert summary["games_played"] == 1
    assert summary["wins"] == 1
    assert summary["matchboxes"] >= 1


def test_loading_missing_model_raises_persistence_error(tmp_path: Path) -> None:
    path = tmp_path / "missing_model.json"

    with pytest.raises(PersistenceError):
        ModelPersistence(path).load()


def test_inspecting_missing_model_raises_persistence_error(tmp_path: Path) -> None:
    path = tmp_path / "missing_model.json"

    with pytest.raises(PersistenceError):
        ModelPersistence(path).inspect_metadata()


def test_loading_corrupt_json_raises_persistence_error(tmp_path: Path) -> None:
    path = tmp_path / "corrupt_model.json"
    path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(PersistenceError):
        ModelPersistence(path).load()


def test_loading_json_without_metadata_raises_persistence_error(tmp_path: Path) -> None:
    path = tmp_path / "invalid_model.json"
    path.write_text(
        json.dumps(
            {
                "model": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PersistenceError):
        ModelPersistence(path).load()


def test_loading_json_without_model_raises_persistence_error(tmp_path: Path) -> None:
    path = tmp_path / "invalid_model.json"
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "model_type": "MENACE",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PersistenceError):
        ModelPersistence(path).load()


def test_loading_non_menace_model_raises_persistence_error(tmp_path: Path) -> None:
    path = tmp_path / "invalid_model.json"
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "model_type": "OtherModel",
                },
                "model": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PersistenceError):
        ModelPersistence(path).load()


def test_loading_model_missing_required_fields_raises_persistence_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid_model.json"
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "model_type": "MENACE",
                },
                "model": {
                    "mark": "O",
                    "matchboxes": {},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PersistenceError):
        ModelPersistence(path).load()


def test_artefact_persistence_save_dataframe_csv_creates_file(tmp_path: Path) -> None:
    dataframe = pd.DataFrame(
        [
            {
                "game": 1,
                "win_rate": 0.5,
                "loss_rate": 0.25,
                "draw_rate": 0.25,
            }
        ]
    )
    path = tmp_path / "results" / "training_log.csv"

    ArtefactPersistence.save_dataframe_csv(dataframe, path)

    assert path.exists()

    loaded = pd.read_csv(path)
    assert loaded.iloc[0]["game"] == 1
    assert loaded.iloc[0]["win_rate"] == 0.5


def test_artefact_persistence_rejects_empty_dataframe_csv(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"

    with pytest.raises(PersistenceError):
        ArtefactPersistence.save_dataframe_csv(pd.DataFrame(), path)


def test_artefact_persistence_load_dataframe_csv_reads_file(tmp_path: Path) -> None:
    path = tmp_path / "comparison_results.csv"
    original = pd.DataFrame(
        [
            {
                "opponent_type": "Random",
                "win_rate": 0.6,
            }
        ]
    )
    original.to_csv(path, index=False)

    loaded = ArtefactPersistence.load_dataframe_csv(path)

    assert not loaded.empty
    assert loaded.iloc[0]["opponent_type"] == "Random"
    assert loaded.iloc[0]["win_rate"] == 0.6


def test_artefact_persistence_load_missing_csv_raises_error(tmp_path: Path) -> None:
    path = tmp_path / "missing.csv"

    with pytest.raises(PersistenceError):
        ArtefactPersistence.load_dataframe_csv(path)


def test_artefact_persistence_save_json_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "experiment_config.json"
    data = {
        "project": "AIS_AL_project5",
        "seed": 42,
        "training_games": 1000,
    }

    ArtefactPersistence.save_json(data, path)

    assert path.exists()

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["project"] == "AIS_AL_project5"
    assert loaded["seed"] == 42
    assert loaded["training_games"] == 1000


def test_artefact_persistence_load_json_reads_file(tmp_path: Path) -> None:
    path = tmp_path / "experiment_config.json"
    path.write_text(
        json.dumps(
            {
                "project": "AIS_AL_project5",
                "seed": 42,
            }
        ),
        encoding="utf-8",
    )

    loaded = ArtefactPersistence.load_json(path)

    assert loaded["project"] == "AIS_AL_project5"
    assert loaded["seed"] == 42


def test_artefact_persistence_load_missing_json_raises_error(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"

    with pytest.raises(PersistenceError):
        ArtefactPersistence.load_json(path)


def test_artefact_persistence_load_corrupt_json_raises_error(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.json"
    path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(PersistenceError):
        ArtefactPersistence.load_json(path)


def test_model_persistence_rejects_unsupported_schema_version(
    tmp_path: Path,
) -> None:
    """A structurally incompatible saved model must not be loaded silently."""
    path = tmp_path / "menace_model.json"
    persistence = ModelPersistence(path)
    persistence.save(MENACEPlayer(mark="O", seed=42))

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metadata"]["schema_version"] = "1.0"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PersistenceError) as error:
        persistence.load()

    assert isinstance(error.value.__cause__, ValueError)
    assert "Unsupported model schema" in str(error.value.__cause__)
