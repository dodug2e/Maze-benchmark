#!/usr/bin/env python3
"""
Training Utility for Pre‑trained Models
======================================

This script now uses **DataLoader**‑based loops for the CNN heuristic
and offline‑buffer training for the Deep Forest Q‑network.

Run Examples
------------
CNN (distance map target):
    python scripts/train_models.py cnn \
        --name cnn_dist_v1 \
        --subset train \
        --epochs 5 \
        --batch-size 16 \
        --lr 1e-3

Deep Forest (offline buffer):
    python scripts/train_models.py deepforest \
        --name df_maze_small \
        --buffer-file buffers/df_buffer_train.pkl
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

# ---------------------------------------------------------------------------
# Project imports (lazy‑safe)
# ---------------------------------------------------------------------------
try:
    from utils.maze_io import load_sample  # noqa: F401 (still used by DF)
    from utils.model_io import save_cnn, save_deepforest
    from algorithms.aco_cnn import MazeFeatureExtractor
    from algorithms.dqn_deepforest import DeepForestQNetwork, Experience
    from utils.data import get_maze_cnn_loader
    from utils.df_pipeline import build_buffer, load_buffer
except ImportError as e:  # pragma: no cover
    raise RuntimeError("Project modules not importable – check PYTHONPATH") from e


# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("train_models")

# ---------------------------------------------------------------------------
# CNN TRAINING --------------------------------------------------------------
# ---------------------------------------------------------------------------

def train_cnn(args: argparse.Namespace):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    LOGGER.info("🚀 CNN training started – %s", vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MazeFeatureExtractor().to(device)
    model.train()

    # Loss: BCEWithLogits ⇒ 마지막 Sigmoid 제거 권장.
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # DataLoader
    loader = get_maze_cnn_loader(
        subset=args.subset,
        batch_size=args.batch_size,
        target_kind="dist",
        shuffle=True,
    )

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for x, y in loader:  # (B,1,H,W)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        epoch_loss = running_loss / len(loader.dataset)
        LOGGER.info("Epoch %3d | loss %.5f", epoch, epoch_loss)

    save_cnn(model, args.name)
    LOGGER.info("✅ CNN saved as %s.pt", args.name)


# ---------------------------------------------------------------------------
# DEEP FOREST TRAINING ------------------------------------------------------
# ---------------------------------------------------------------------------

def train_deepforest(args: argparse.Namespace):
    LOGGER.info("🌲 Deep Forest training – %s", vars(args))

    # 1. Experience buffer --------------------------------------------------
    if args.buffer_file:
        buffer = load_buffer(args.buffer_file)
    else:
        buffer = build_buffer(
            subset=args.subset,
            n_samples=args.n_samples,
            policy="bfs",
        )

    # 2. Train network ------------------------------------------------------
    q_net = DeepForestQNetwork()
    q_net.fit(buffer)

    # 3. Save model ---------------------------------------------------------
    save_deepforest(q_net, args.name)
    LOGGER.info("✅ Deep Forest saved as %s.pkl", args.name)


# ---------------------------------------------------------------------------
# CLI ----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("train_models.py", description="Train CNN / DeepForest models")
    sub = p.add_subparsers(dest="mode", required=True)

    # CNN ---------------------------------------------------------------
    p_cnn = sub.add_parser("cnn", help="Train CNN heuristic model")
    p_cnn.add_argument("--name", required=True)
    p_cnn.add_argument("--subset", default="train")
    p_cnn.add_argument("--epochs", type=int, default=5)
    p_cnn.add_argument("--batch-size", type=int, default=32)
    p_cnn.add_argument("--lr", type=float, default=1e-3)
    p_cnn.set_defaults(func=train_cnn)

    # DeepForest --------------------------------------------------------
    p_df = sub.add_parser("deepforest", help="Train Deep Forest Q-network")
    p_df.add_argument("--name", required=True)
    p_df.add_argument("--subset", default="train")
    p_df.add_argument("--n-samples", type=int, default=10000, help="buffer size if building on the fly")
    p_df.add_argument("--buffer-file", help="pre-built pickle buffer")
    p_df.set_defaults(func=train_deepforest)

    return p


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    args.func(args)  # type: ignore[attr-defined]


if __name__ == "__main__":
    main(sys.argv[1:])
