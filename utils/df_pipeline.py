# utils/data.py
"""Dataset + helper loader for CNN heuristic training.

This module converts the project’s maze dataset into PyTorch tensors
suitable for training **MazeFeatureExtractor**. Target generation is
based on *normalised Manhattan-distance to goal* (option *"dist"*).

Supports optional resizing so that DataLoader can batch tensors of
equal spatial dimensions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Tuple, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from utils.maze_io import load_sample  # type: ignore
except ImportError as e:
    raise RuntimeError("utils.maze_io not found; ensure project path is set") from e

# ---------------------------------------------------------------------------
# Target map helpers
# ---------------------------------------------------------------------------

def _manhattan_distance_map(h: int, w: int, goal: Tuple[int, int]) -> np.ndarray:
    """Return *H×W* array with normalized Manhattan distance to *goal*.

    Values in [0,1]: 1.0 at goal cell, 0.0 at farthest path cell.
    Walls are left as-is (mask later).
    """
    gr, gc = goal
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    dist = np.abs(rows - gr) + np.abs(cols - gc)
    norm = dist.max() or 1  # avoid div-zero
    score = 1.0 - dist / norm
    return score.astype(np.float32)


def make_target(img: np.ndarray, meta: dict, *, kind: str = "dist") -> np.ndarray:
    """Create target map according to *kind*.

    Parameters
    ----------
    img : ndarray (H×W) – binary maze image (0 path, 1 wall)
    meta : dict – metadata JSON from load_sample
    kind : {"dist"}

    Returns
    -------
    tgt : ndarray (1, H, W)
    """
    h, w = img.shape
    goal = tuple(meta.get("goal", meta.get("exit")))

    if kind == "dist":
        tgt = _manhattan_distance_map(h, w, goal)
        tgt[img == 1] = 0.0
        return tgt[None]

    raise ValueError(f"Unsupported target kind: {kind}")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MazeCNNDataset(Dataset):
    """Return (input, target) pairs for CNN heuristic training, with optional resize."""

    def __init__(
        self,
        subset: str = "train",
        *,
        target_kind: str = "dist",
        resize: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.subset = subset
        self.target_kind = target_kind
        self.resize = resize
        self.transform = transform

        img_dir: Path = Path("datasets") / subset / "img"
        self.sample_ids: List[str] = sorted(p.stem for p in img_dir.glob("*.png"))
        if not self.sample_ids:
            raise FileNotFoundError(f"No PNG files found in {img_dir}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        sample_id = self.sample_ids[idx]
        img, meta, _ = load_sample(sample_id, subset=self.subset)

        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W)
        inp = torch.from_numpy(arr[None])               # (1, H, W)
        tgt = torch.from_numpy(make_target(arr, meta, kind=self.target_kind))  # (1, H, W)

        # Optional resize to common spatial dims
        if self.resize:
            size = self.resize
            inp = F.interpolate(inp.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)
            tgt = F.interpolate(tgt.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)

        if self.transform:
            inp = self.transform(inp)
        return inp, tgt

# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def get_maze_cnn_loader(
    subset: str = "train",
    *,
    target_kind: str = "dist",
    resize: Optional[Tuple[int, int]] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Get DataLoader for MazeCNNDataset with optional resize."""
    ds = MazeCNNDataset(
        subset=subset,
        target_kind=target_kind,
        resize=resize,
        transform=None,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

__all__ = [
    "MazeCNNDataset",
    "get_maze_cnn_loader",
    "make_target",
]
