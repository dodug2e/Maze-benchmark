# utils/df_pipeline.py
"""Experience‑collection helpers for *DeepForest* pre‑training.

The goal is to build an **offline buffer** of (state, action, reward, …)
that can be fed into :pymeth:`DeepForestQNetwork.fit` before on‑policy
fine‑tuning.  This reduces early random exploration cost.

Workflow
========

Example – collect 10k transitions with a simple BFS oracle::

    from utils.df_pipeline import build_buffer, save_buffer
    exps = build_buffer(subset="train", n_samples=10000, policy="bfs")
    save_buffer(exps, "buffer_train.pkl")

Then, in ``train_models.py`` → load buffer → ``q_net.fit(buffer)``.
"""
from __future__ import annotations

import random
import pickle
from pathlib import Path
from typing import List, Tuple, Literal

import numpy as np
from tqdm import tqdm

try:
    from algorithms.dqn_deepforest import Experience, MazeEnvironment  # type: ignore
    from utils.maze_io import load_sample  # type: ignore
except ImportError as e:  # pragma: no cover
    raise RuntimeError("Project modules missing – ensure PYTHONPATH is set") from e

# ---------------------------------------------------------------------------
# Simple policies for data collection
# ---------------------------------------------------------------------------

Policy = Literal["random", "bfs"]  # extendable


def _random_policy(env: MazeEnvironment):
    return random.randrange(env.action_size)


def _bfs_policy(env: MazeEnvironment):
    """Single‑step action towards BFS shortest path (if reachable)."""
    from collections import deque

    rows, cols = env.rows, env.cols
    start = env.current_pos
    goal = env.goal
    maze = env.maze

    q = deque([start])
    parent = {start: None}
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        for dx, dy in env.actions:
            nr, nc = r + dx, c + dy
            if (0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] == 0 and (nr, nc) not in parent):
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))
    if goal not in parent:
        return _random_policy(env)

    # reconstruct first step
    step = goal
    while parent[step] != start:
        step = parent[step]
    dr, dc = step[0] - start[0], step[1] - start[1]
    return env.actions.index((dr, dc))


_POLICY_LOOKUP = {
    "random": _random_policy,
    "bfs": _bfs_policy,
}


# ---------------------------------------------------------------------------
# Buffer builder
# ---------------------------------------------------------------------------

def build_buffer(
    *,
    subset: str = "train",
    n_samples: int = 5000,
    policy: Policy = "random",
    max_steps_ep: int = 300,
    seed: int = 42,
) -> List[Experience]:
    """Collect *n_samples* transitions and return as list[Experience]."""
    random.seed(seed)
    np.random.seed(seed)

    buffer: List[Experience] = []
    all_ids = [p.stem for p in (Path("datasets") / subset / "img").glob("*.png")]
    random.shuffle(all_ids)
    pid = tqdm(total=n_samples, desc="Collect")

    for sid in all_ids:
        if len(buffer) >= n_samples:
            break
        img, meta, _ = load_sample(sid, subset=subset)
        maze = (np.asarray(img) < 128).astype(np.uint8)
        start = tuple(meta.get("entrance", meta.get("start")))
        goal = tuple(meta.get("exit", meta.get("goal")))
        env = MazeEnvironment(maze, start, goal)

        state = env.reset()
        steps = 0
        while steps < max_steps_ep and len(buffer) < n_samples:
            a = _POLICY_LOOKUP[policy](env)
            next_state, reward, done = env.step(a)
            buffer.append(Experience(state, a, reward, next_state, done))
            state = next_state
            steps += 1
            if done:
                state = env.reset()
                steps = 0
        pid.update(len(buffer) - pid.n)

    pid.close()
    return buffer[:n_samples]


# ---------------------------------------------------------------------------
# Saving / loading utility
# ---------------------------------------------------------------------------

def save_buffer(buffer: List[Experience], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(buffer, f)
    print(f"[df_pipeline] buffer saved → {path}")


def load_buffer(path: str | Path) -> List[Experience]:
    with open(path, "rb") as f:
        buf: List[Experience] = pickle.load(f)
    print(f"[df_pipeline] buffer loaded ← {path}")
    return buf


__all__ = [
    "build_buffer",
    "save_buffer",
    "load_buffer",
]
