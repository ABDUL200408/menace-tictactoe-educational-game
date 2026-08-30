"""
persistence.py

This file is responsible for saving and loading the MENACE model and project artefacts.
It is independent from the learning algorithm and only manages persistence, allowing
trained models, matchboxes, bead counts, and experiment results to be stored and
reloaded for reproducibility, evaluation, and dissertation evidence.

Why JSON instead of pickle?

- JSON is human-readable.
- JSON is easy to inspect during the viva.
- Matchboxes and bead counts can be verified manually.
- JSON improves reproducibility and supports academic evidence.
- JSON is safer than pickle because it does not execute Python objects when loaded.
"""


from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.menace import MENACEPlayer


class PersistenceError(RuntimeError):
    """Custom exception for clear persistence-related error messages."""


@dataclass(frozen=True)
class ModelMetadata:
    """Stores saved-model metadata for traceability and reproducibility."""

    project_code: str = "AIS_AL_project5"
    project_title: str = (
        "An online game to teach how machines are trained "
        "to win noughts and crosses (MENACE)"
    )
    model_type: str = "MENACE"
    schema_version: str = "2.0"
    saved_at: str = ""
    notes: str = "Saved MENACE model with matchboxes and bead counts."

    def to_dict(self) -> Dict[str, str]:
        """Return serialisable metadata dictionary."""
        return {
            "project_code": self.project_code,
            "project_title": self.project_title,
            "model_type": self.model_type,
            "schema_version": self.schema_version,
            "saved_at": self.saved_at or datetime.now().isoformat(timespec="seconds"),
            "notes": self.notes,
        }


class ModelPersistence:
    """Manages saving, loading, validating, and inspecting MENACE JSON models."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def save(
        self,
        menace: MENACEPlayer,
        *,
        create_backup: bool = True,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save MENACEPlayer to JSON using atomic write.

        Args:
            menace:
                MENACE model to save.
            create_backup:
                If True, create a .bak copy before overwrite.
            extra_metadata:
                Optional extra metadata for experiment traceability.
        """
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)

            if create_backup and self.path.exists():
                self._create_backup()

            model_dict = menace.to_dict()

            payload = {
                "metadata": self._build_metadata(menace, extra_metadata),
                "model": model_dict,
            }

            self._validate_payload(payload)

            temp_path = self.path.with_suffix(self.path.suffix + ".tmp")

            with temp_path.open("w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, ensure_ascii=False)

            temp_path.replace(self.path)

        except Exception as exc:
            raise PersistenceError(
                f"Failed to save MENACE model to {self.path}"
            ) from exc

    def load(self) -> MENACEPlayer:
        """
        Load MENACEPlayer from JSON.

        Returns:
            Loaded MENACEPlayer.
        """
        try:
            payload = self._read_json(self.path)
            self._validate_payload(payload)
            return MENACEPlayer.from_dict(payload["model"])

        except Exception as exc:
            raise PersistenceError(
                f"Failed to load MENACE model from {self.path}"
            ) from exc

    def exists(self) -> bool:
        """Return True if model file exists."""
        return self.path.exists()

    def delete(self, *, delete_backup: bool = False) -> None:
        """
        Delete saved model file.

        Args:
            delete_backup:
                If True, also delete .bak file.
        """
        try:
            if self.path.exists():
                self.path.unlink()

            backup = self.backup_path
            if delete_backup and backup.exists():
                backup.unlink()

        except Exception as exc:
            raise PersistenceError(
                f"Failed to delete model file {self.path}"
            ) from exc

    def inspect_metadata(self) -> Dict[str, Any]:
        """Read saved model metadata without constructing a MENACEPlayer."""
        try:
            payload = self._read_json(self.path)
            self._validate_payload(payload)
            return dict(payload["metadata"])

        except Exception as exc:
            raise PersistenceError(
                f"Failed to inspect metadata from {self.path}"
            ) from exc

    def inspect_model_summary(self) -> Dict[str, Any]:
        """Return a lightweight model summary for interface display and reporting."""
        try:
            payload = self._read_json(self.path)
            self._validate_payload(payload)

            model = payload["model"]
            matchboxes = model.get("matchboxes", {})

            return {
                "path": str(self.path),
                "model_type": payload["metadata"].get("model_type"),
                "schema_version": payload["metadata"].get("schema_version"),
                "saved_at": payload["metadata"].get("saved_at"),
                "mark": model.get("mark"),
                "use_symmetry": model.get("use_symmetry"),
                "games_played": model.get("games_played", 0),
                "wins": model.get("wins", 0),
                "losses": model.get("losses", 0),
                "draws": model.get("draws", 0),
                "matchboxes": len(matchboxes) if isinstance(matchboxes, dict) else 0,
                "reward_win": model.get("reward_win"),
                "reward_draw": model.get("reward_draw"),
                "penalty_loss": model.get("penalty_loss"),
            }

        except Exception as exc:
            raise PersistenceError(
                f"Failed to inspect model summary from {self.path}"
            ) from exc

    @property
    def backup_path(self) -> Path:
        """Return backup path."""
        return self.path.with_suffix(self.path.suffix + ".bak")

    def _create_backup(self) -> None:
        """Create backup copy of existing model."""
        shutil.copy2(self.path, self.backup_path)

    def _build_metadata(
        self,
        menace: MENACEPlayer,
        extra_metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build metadata dictionary with model summary."""
        metadata: Dict[str, Any] = ModelMetadata().to_dict()

        summary = menace.training_summary()

        metadata.update(
            {
                "games_played": summary.get("games_played"),
                "wins": summary.get("wins"),
                "losses": summary.get("losses"),
                "draws": summary.get("draws"),
                "matchboxes": summary.get("matchboxes"),
                "total_beads": summary.get("total_beads"),
                "use_symmetry": summary.get("use_symmetry"),
                "reward_win": menace.reward_win,
                "reward_draw": menace.reward_draw,
                "penalty_loss": menace.penalty_loss,
            }
        )

        if extra_metadata:
            metadata["extra"] = extra_metadata

        return metadata

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        """Read JSON object from disk."""
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if not isinstance(payload, dict):
            raise ValueError("JSON root must be an object.")

        return payload

    @staticmethod
    def _validate_payload(payload: Dict[str, Any]) -> None:
        """
        Validate saved JSON structure.

        Raises:
            ValueError:
                If required fields are missing.
        """
        if not isinstance(payload, dict):
            raise ValueError("Model file must contain a JSON object.")

        if "metadata" not in payload:
            raise ValueError("Model file is missing metadata.")

        if "model" not in payload:
            raise ValueError("Model file is missing model data.")

        metadata = payload["metadata"]
        model = payload["model"]

        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary.")

        if not isinstance(model, dict):
            raise ValueError("model must be a dictionary.")

        if metadata.get("model_type") != "MENACE":
            raise ValueError("Saved model is not a MENACE model.")

        expected_schema = ModelMetadata().schema_version
        actual_schema = metadata.get("schema_version")
        if actual_schema != expected_schema:
            raise ValueError(
                f"Unsupported model schema {actual_schema!r}; "
                f"expected {expected_schema!r}."
            )

        required_model_fields = {
            "mark",
            "name",
            "initial_beads",
            "minimum_beads",
            "reward_win",
            "reward_draw",
            "penalty_loss",
            "use_symmetry",
            "games_played",
            "wins",
            "losses",
            "draws",
            "matchboxes",
        }

        missing = required_model_fields.difference(model.keys())
        if missing:
            raise ValueError(
                "Saved MENACE model is missing required fields: "
                + ", ".join(sorted(missing))
            )

        if not isinstance(model["matchboxes"], dict):
            raise ValueError("model.matchboxes must be a dictionary.")


class ArtefactPersistence:
    """Saves and loads CSV and JSON artefacts used for analysis and project evidence."""
    @staticmethod
    def save_dataframe_csv(
        dataframe: pd.DataFrame,
        path: Path | str,
        *,
        index: bool = False,
    ) -> None:
        """Save DataFrame to CSV."""
        output_path = Path(path)

        try:
            if dataframe.empty:
                raise ValueError("Cannot save empty DataFrame.")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            dataframe.to_csv(output_path, index=index)

        except Exception as exc:
            raise PersistenceError(
                f"Failed to save DataFrame CSV to {output_path}"
            ) from exc

    @staticmethod
    def load_dataframe_csv(path: Path | str) -> pd.DataFrame:
        """Load DataFrame from CSV."""
        input_path = Path(path)

        try:
            if not input_path.exists():
                raise FileNotFoundError(f"CSV file not found: {input_path}")

            return pd.read_csv(input_path)

        except Exception as exc:
            raise PersistenceError(
                f"Failed to load DataFrame CSV from {input_path}"
            ) from exc

    @staticmethod
    def save_json(data: Dict[str, Any], path: Path | str) -> None:
        """Save dictionary to JSON."""
        output_path = Path(path)

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with output_path.open("w", encoding="utf-8") as file:
                json.dump(data, file, indent=2, ensure_ascii=False)

        except Exception as exc:
            raise PersistenceError(
                f"Failed to save JSON artefact to {output_path}"
            ) from exc

    @staticmethod
    def load_json(path: Path | str) -> Dict[str, Any]:
        """Load dictionary from JSON."""
        input_path = Path(path)

        try:
            if not input_path.exists():
                raise FileNotFoundError(f"JSON file not found: {input_path}")

            with input_path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            if not isinstance(data, dict):
                raise ValueError("JSON artefact must contain an object.")

            return data

        except Exception as exc:
            raise PersistenceError(
                f"Failed to load JSON artefact from {input_path}"
            ) from exc
