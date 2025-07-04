# utils/model_io.py
"""Light‑weight I/O helpers for pre‑trained models.

This module standardises where **CNN** (PyTorch) and **Deep Forest**
(joblib‑serialised) models are saved and how they are loaded back.

Directory layout (created on‑demand) ::

    pretrained_models/
    ├── CNNs/
    │   └── <name>.pt
    └── DeepForest/
        └── <name>.pkl

Functions
---------
* :func:`save_cnn` / :func:`load_cnn`
* :func:`save_deepforest` / :func:`load_deepforest`
* :func:`list_models` – quick inventory helper
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Type
import logging
import torch

# Optional heavy deps are imported lazily to avoid torch overhead
import importlib

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base directories (resolved relative to project root)
# ---------------------------------------------------------------------------

PRETRAIN_DIR = Path("pretrained_models").resolve()
CNN_DIR = PRETRAIN_DIR / "CNNs"
DF_DIR = PRETRAIN_DIR / "DeepForest"


# ---------------------------------------------------------------------------
# Helper: ensure directory exists
# ---------------------------------------------------------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CNN helpers (PyTorch state_dict)
# ---------------------------------------------------------------------------

def save_cnn(model: "torch.nn.Module", name: str) -> Path:  # type: ignore[name‑defined]
    """Save **state_dict** of *model* under ``CNN_DIR``.

    Parameters
    ----------
    model : torch.nn.Module
        The CNN model to be saved.
    name : str
        File name without extension (``.pt`` will be appended).

    Returns
    -------
    Path
        Full path to the saved file.
    """
    torch = importlib.import_module("torch")  # lazy import
    _ensure_dir(CNN_DIR)
    file_path = CNN_DIR / f"{name}.pt"
    torch.save(model.state_dict(), file_path)
    LOGGER.info("[model_io] CNN saved → %s", file_path.relative_to(Path.cwd()))
    return file_path


def load_cnn(name: str, model_cls: Type["torch.nn.Module"], *, map_location: str | None = None):  # type: ignore[name‑defined]
    """Instantiate *model_cls* and load its weights from disk.

    ``model_cls`` should be the exact class used when saving.

    Parameters
    ----------
    name : str
        Base file name (without extension).
    model_cls : type[torch.nn.Module]
        Class of the CNN to instantiate.
    map_location : str | None, optional
        Same as ``torch.load(map_location=...)``.

    Returns
    -------
    torch.nn.Module
        Model with weights loaded and set to ``eval()`` mode.
    """
    torch = importlib.import_module("torch")  # lazy import
    file_path = CNN_DIR / f"{name}.pt"
    if not file_path.exists():
        raise FileNotFoundError(f"CNN model not found: {file_path}")

    model = model_cls()
    state = torch.load(file_path, map_location=map_location or "cpu")
    model.load_state_dict(state)
    model.eval()
    LOGGER.info("[model_io] CNN loaded ← %s", file_path.relative_to(Path.cwd()))
    return model


# ---------------------------------------------------------------------------
# Deep Forest helpers (joblib)
# ---------------------------------------------------------------------------

def save_deepforest(q_net: Any, name: str) -> Path:
    """Serialise a *DeepForestQNetwork* (or any sklearn‑compatible object)."""
    import joblib  # heavy but pure‑py

    _ensure_dir(DF_DIR)
    file_path = DF_DIR / f"{name}.pkl"
    joblib.dump(q_net, file_path)
    LOGGER.info("[model_io] DeepForest saved → %s", file_path.relative_to(Path.cwd()))
    return file_path


def load_deepforest(name: str):
    """Load a previously saved DeepForest network object."""
    import joblib

    file_path = DF_DIR / f"{name}.pkl"
    if not file_path.exists():
        raise FileNotFoundError(f"DeepForest model not found: {file_path}")

    model = joblib.load(file_path)
    LOGGER.info("[model_io] DeepForest loaded ← %s", file_path.relative_to(Path.cwd()))
    return model


# ---------------------------------------------------------------------------
# Inventory utility
# ---------------------------------------------------------------------------

def list_models(kind: str = "all") -> dict[str, list[str]]:
    """Return available model names.

    Parameters
    ----------
    kind : {"cnn", "deepforest", "all"}
        Which category to list.

    Returns
    -------
    dict[str, list[str]]
        Mapping of category → list of names (without extensions).
    """
    kinds = {"cnn", "deepforest", "all"}
    if kind not in kinds:
        raise ValueError(f"kind must be one of {kinds}")

    result: dict[str, list[str]] = {}
    if kind in {"cnn", "all"}:
        result["cnn"] = [p.stem for p in CNN_DIR.glob("*.pt")]
    if kind in {"deepforest", "all"}:
        result["deepforest"] = [p.stem for p in DF_DIR.glob("*.pkl")]
    return result


__all__ = [
    "save_cnn",
    "load_cnn",
    "save_deepforest",
    "load_deepforest",
    "list_models",
]
