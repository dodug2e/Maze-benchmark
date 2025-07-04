# utils/df_pipeline.py

import random, pickle
from pathlib import Path
from typing import List, Literal
import numpy as np
from tqdm import tqdm

from algorithms.dqn_deepforest import Experience, MazeEnvironment
from utils.maze_io import load_sample

Policy = Literal["random", "bfs"]

def _random_policy(env: MazeEnvironment):
    return random.randrange(env.action_size)

def _bfs_policy(env: MazeEnvironment):
    from collections import deque
    rows, cols = env.rows, env.cols
    start, goal = env.current_pos, env.goal
    maze = env.maze
    q = deque([start])
    parent = {start: None}
    while q:
        r, c = q.popleft()
        if (r, c) == goal: break
        for dx, dy in env.actions:
            nr, nc = r+dx, c+dy
            if (0 <= nr < rows and 0 <= nc < cols and maze[nr,nc]==0 and (nr,nc) not in parent):
                parent[(nr,nc)] = (r,c)
                q.append((nr,nc))
    if goal not in parent:
        return _random_policy(env)
    step = goal
    while parent[step] != start:
        step = parent[step]
    dr, dc = step[0]-start[0], step[1]-start[1]
    return env.actions.index((dr, dc))

_POLICY = {"random": _random_policy, "bfs": _bfs_policy}

def build_buffer(*, subset: str="train", n_samples: int=5000, policy: Policy="random", max_steps_ep: int=300, seed: int=42) -> List[Experience]:
    random.seed(seed); np.random.seed(seed)
    buffer: List[Experience] = []
    ids = [p.stem for p in (Path("datasets")/subset/"img").glob("*.png")]
    tqdm_bar = tqdm(total=n_samples, desc="Collect")
    for sid in ids:
        if len(buffer)>=n_samples: break
        img, meta, _ = load_sample(sid, subset=subset)
        maze = (np.asarray(img)<128).astype(np.uint8)
        start = tuple(meta.get("entrance", meta.get("start")))
        goal  = tuple(meta.get("exit",     meta.get("goal")))
        env = MazeEnvironment(maze, start, goal)
        state = env.reset(); steps=0
        while steps<max_steps_ep and len(buffer)<n_samples:
            a = _POLICY[policy](env)
            nxt, r, done = env.step(a)
            buffer.append(Experience(state, a, r, nxt, done))
            state = nxt; steps+=1
            if done:
                state = env.reset(); steps=0
        tqdm_bar.update(len(buffer)-tqdm_bar.n)
    tqdm_bar.close()
    return buffer[:n_samples]

def save_buffer(buffer: List[Experience], path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(buffer, f)
    print(f"[df_pipeline] saved → {path}")

def load_buffer(path: str) -> List[Experience]:
    path = Path(path)
    with open(path, "rb") as f:
        buf = pickle.load(f)
    print(f"[df_pipeline] loaded ← {path}")
    return buf
