# utils/data.py
"""Dataset + helper loader for CNN heuristic training.

This module converts the project’s maze dataset into PyTorch tensors
suitable for training **MazeFeatureExtractor**. Target generation is
based on *normalised Manhattan‑distance to goal* (option *"dist"*).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from utils.maze_io import load_sample  # type: ignore
except ImportError as e:  # unit‑test fallback
    raise RuntimeError("utils.maze_io not found; ensure project path is set") from e

# ---------------------------------------------------------------------------
# Target map helpers
# ---------------------------------------------------------------------------

def _manhattan_distance_map(h: int, w: int, goal: Tuple[int, int]) -> np.ndarray:
    """Return *H×W* array with normalised Manhattan distance to *goal*.

    All values are mapped to [0, 1] where **1.0 = goal cell**, 0.0 = farthest.
    Walls are **left untouched** – caller can overwrite after masking.
    """
    gr, gc = goal
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    dist = np.abs(rows - gr) + np.abs(cols - gc)
    norm = dist.max() or 1  # avoid div‑0 on 1×1 maze
    score = 1.0 - dist / norm
    return score.astype(np.float32)


def make_target(img: np.ndarray, meta: dict, *, kind: str = "dist") -> np.ndarray:
    """Create target map according to *kind*.

    Parameters
    ----------
    img : ndarray (H×W) – binary maze image (0 path, 1 wall)
    meta : dict – metadata JSON returned by *load_sample*
    kind : {"dist"}
    """
    h, w = img.shape
    goal = tuple(meta["goal"]) if "goal" in meta else tuple(meta["exit"])

    # 1. distance‑based target
    if kind == "dist":
        tgt = _manhattan_distance_map(h, w, goal)
        tgt[img == 1] = 0.0  # walls = 0 (no reward)
        return tgt[None]  # add channel dim (1, H, W)

    raise ValueError(f"Unsupported target kind: {kind}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MazeCNNDataset(Dataset):
    """Return (input, target) pairs for CNN heuristic training."""

    def __init__(
        self,
        subset: str = "train",
        *,
        target_kind: str = "dist",
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.subset = subset
        self.target_kind = target_kind
        self.transform = transform

        img_dir: Path = Path("datasets") / subset / "img"
        self.sample_ids: List[str] = sorted(p.stem for p in img_dir.glob("*.png"))
        if not self.sample_ids:
            raise FileNotFoundError(f"No PNG files found in {img_dir}")

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401 (simple‑return OK)
        return len(self.sample_ids)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        sample_id = self.sample_ids[idx]
        img, meta, _ = load_sample(sample_id, subset=self.subset)

        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W) 0.0–1.0
        inp = torch.from_numpy(arr[None])  # (1, H, W)
        tgt = torch.from_numpy(make_target(arr, meta, kind=self.target_kind))

        if self.transform is not None:
            inp = self.transform(inp)
        return inp, tgt

# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def get_maze_cnn_loader(
    subset: str = "train",
    *,
    target_kind: str = "dist",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Shortcut to obtain DataLoader ready for training."""
    ds = MazeCNNDataset(subset=subset, target_kind=target_kind)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


__all__ = [
    "MazeCNNDataset",
    "get_maze_cnn_loader",
    "make_target",
]
