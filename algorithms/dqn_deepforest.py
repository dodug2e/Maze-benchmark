from __future__ import annotations

"""
DQN + Deep Forest 알고리즘 (refactored)
- Skip empty‑action training rows (no dummy 0‑targets)
- Single StandardScaler fit (TODO: incremental if needed)
- Reward 중복 패널티 제거
- Random seed helper for reproducibility
"""

import copy
import random
import time
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Seed helper ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


seed_everything(42)

# ---------------------------------------------------------------------------
# Experience tuple & result dataclass
# ---------------------------------------------------------------------------

Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done"]
)


@dataclass
class DQNDeepForestResult:
    algorithm: str = "DQN+DeepForest"
    maze_id: str = ""
    maze_size: Tuple[int, int] = (0, 0)
    execution_time: float = 0.0
    power_consumption: float = 0.0
    vram_usage: float = 0.0
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    solution_found: bool = False
    solution_length: int = 0
    total_steps: int = 0
    max_steps: int = 0
    failure_reason: str = ""
    path: List[Tuple[int, int]] | None = None
    training_episodes: int = 0
    training_time: float = 0.0
    final_epsilon: float = 0.0
    average_reward: float = 0.0
    convergence_episode: int = 0
    forest_training_time: float = 0.0
    feature_importance: Dict[str, List[float]] | None = None


# ---------------------------------------------------------------------------
# Deep‑Forest‑based Q‑network
# ---------------------------------------------------------------------------

class DeepForestQNetwork:
    """Q‑network backed by a cascade of Random/Extra Trees (gcForest‑style)."""

    NUM_FEATURES: int = 19  # update this constant if extract_features changes

    def __init__(
        self,
        *,
        action_size: int = 4,
        n_estimators: int = 100,
        n_layers: int = 3,
        window_size: int = 5,
    ) -> None:
        self.action_size = action_size
        self.n_estimators = n_estimators
        self.n_layers = n_layers
        self.window_size = window_size

        # forests[layer][action] -> {"rf": RF, "et": ET}
        self.q_forests: list[list[dict[str, Any]]] = []
        for layer in range(n_layers):
            layer_forests: list[dict[str, Any]] = []
            for action in range(action_size):
                forests = {
                    "rf": RandomForestRegressor(
                        n_estimators=n_estimators,
                        random_state=42 + action + layer,
                        n_jobs=-1,
                        warm_start=False,
                        max_depth=15,
                        min_samples_leaf=2,
                    ),
                    "et": ExtraTreesRegressor(
                        n_estimators=n_estimators,
                        random_state=142 + action + layer,
                        n_jobs=-1,
                        warm_start=False,
                        max_depth=15,
                        min_samples_leaf=2,
                    ),
                }
                layer_forests.append(forests)
            self.q_forests.append(layer_forests)

        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance_: dict[str, Any] = {}

    # ---------------------------------------------------------------------
    # Feature extraction (19‑D vector)
    # ---------------------------------------------------------------------

    def extract_features(self, state: np.ndarray, pos: Tuple[int, int]) -> np.ndarray:
        maze = state[0]
        goal_map = state[2]
        rows, cols = maze.shape
        r, c = pos
        half = self.window_size // 2

        # 1. local window around agent (padding with walls)
        pad_maze = np.pad(maze, half, constant_values=1)
        window = pad_maze[r : r + self.window_size, c : c + self.window_size]

        feats: list[float] = []
        feats.append(np.mean(window))  # wall density

        center = self.window_size // 2
        dirs = [window[:center, center], window[center + 1 :, center], window[center, :center], window[center, center + 1 :]]
        feats.extend([np.mean(d == 1) if d.size else 0.0 for d in dirs])

        # 2. goal‑distance & direction
        g_pos = np.argwhere(goal_map == 1)
        if len(g_pos):
            gr, gc = g_pos[0]
            man = abs(r - gr) + abs(c - gc)
            euc = np.hypot(r - gr, c - gc)
            feats.extend([man / (rows + cols), euc / np.hypot(rows, cols)])
            dr, dc = gr - r, gc - c
            feats.extend([float(dr > 0), float(dr < 0), float(dc > 0), float(dc < 0)])
        else:
            feats.extend([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

        # 3. action validity (N,S,W,E)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dx, c + dy
            feats.append(float(0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] == 0))

        # 4. connectivity / freedom metrics
        feats.append(np.mean(window == 0))
        neighbors = sum(
            1
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if 0 <= r + dx < rows and 0 <= c + dy < cols and maze[r + dx, c + dy] == 0
        )
        feats.append(neighbors / 4.0)

        # 5. line patterns in window
        h_lines = np.sum(np.all(window == 1, axis=1))
        v_lines = np.sum(np.all(window == 1, axis=0))
        feats.extend([v_lines / self.window_size, h_lines / self.window_size])

        assert len(feats) == self.NUM_FEATURES, "Feature length mismatch"
        return np.asarray(feats, dtype=np.float32)

    # ---------------------------------------------------------------------
    # Fitting / predicting
    # ---------------------------------------------------------------------

    def _prepare_training_data(
        self,
        experiences: List[Experience],
        *,
        target_network: "DeepForestQNetwork" | None = None,
        gamma: float = 0.99,
    ) -> Tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        feats_per_action: dict[int, list[np.ndarray]] = {a: [] for a in range(self.action_size)}
        targets_per_action: dict[int, list[float]] = {a: [] for a in range(self.action_size)}

        for s, a, r, s_next, done in experiences:
            pos = tuple(map(int, np.argwhere(s[1] == 1)[0]))
            feature_vec = self.extract_features(s, pos)

            if done:
                q_target = r
            else:
                tn = target_network or self
                next_pos_idx = np.argwhere(s_next[1] == 1)
                if len(next_pos_idx):
                    next_pos = tuple(map(int, next_pos_idx[0]))
                    q_next = tn.predict(s_next, next_pos)
                    q_target = r + gamma * float(np.max(q_next))
                else:
                    q_target = r

            feats_per_action[a].append(feature_vec)
            targets_per_action[a].append(q_target)

        feats_dict: dict[int, np.ndarray] = {}
        tgts_dict: dict[int, np.ndarray] = {}
        for a in range(self.action_size):
            if feats_per_action[a]:  # 학습 데이터가 있을 때만 포함
                feats_dict[a] = np.vstack(feats_per_action[a])
                tgts_dict[a] = np.asarray(targets_per_action[a])
        return feats_dict, tgts_dict

    def fit(
        self,
        experiences: List[Experience],
        *,
        target_network: "DeepForestQNetwork" | None = None,
        gamma: float = 0.99,
    ) -> None:
        if not experiences:
            return

        X_dict, y_dict = self._prepare_training_data(
            experiences, target_network=target_network, gamma=gamma
        )
        if not X_dict:  # no data at all
            return

        # Fit scaler (single batch fit for now)
        if not self.is_trained:
            all_feats = np.vstack(list(X_dict.values()))
            self.scaler.fit(all_feats)

        # Train layer by layer, per available action
        for a, X in X_dict.items():
            y = y_dict[a]
            X = self.scaler.transform(X)
            layer_input = X
            for layer_idx, forests in enumerate(self.q_forests):
                preds = []
                fdict = forests[a]
                for tree_name, tree in fdict.items():
                    tree.fit(layer_input, y)
                    preds.append(tree.predict(layer_input))
                if layer_idx < self.n_layers - 1:
                    layer_input = np.column_stack([layer_input, *preds])
            # feature importance 저장 (마지막 layer 기준)
            fi = {}
            for n, t in self.q_forests[-1][a].items():
                if hasattr(t, "feature_importances_"):
                    fi[n] = t.feature_importances_.tolist()
            self.feature_importance_[f"action_{a}"] = fi

        self.is_trained = True

    def predict(self, state: np.ndarray, pos: Tuple[int, int]) -> np.ndarray:
        if not self.is_trained:
            return np.random.rand(self.action_size)

        feat = self.extract_features(state, pos).reshape(1, -1)
        feat = self.scaler.transform(feat)
        q_vals: list[float] = []
        for a in range(self.action_size):
            layer_in = feat.copy()
            for layer_idx in range(self.n_layers):
                forests = self.q_forests[layer_idx][a]
                preds = [tree.predict(layer_in) for tree in forests.values()]
                if layer_idx < self.n_layers - 1:
                    layer_in = np.column_stack([layer_in, *preds])
            q_vals.append(float(np.mean(preds)))  # 마지막 layer 평균
        return np.asarray(q_vals, dtype=np.float32)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Experience(*args))

    def sample(self, batch_size: int):
        size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, size)

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Maze environment (refactored reward)
# ---------------------------------------------------------------------------

class MazeEnvironment:
    def __init__(self, maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.current_pos = start
        self.rows, self.cols = maze.shape
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.action_size = 4
        self.max_steps = self.rows * self.cols * 2

    def reset(self):
        self.current_pos = self.start
        self.visited: set[Tuple[int, int]] = {self.start}
        self.steps = 0
        return self._state()

    def _state(self):
        s = np.zeros((3, self.rows, self.cols), dtype=np.float32)
        s[0] = self.maze
        s[1, self.current_pos[0], self.current_pos[1]] = 1.0
        s[2, self.goal[0], self.goal[1]] = 1.0
        return s

    def _valid(self, pos):
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.maze[r, c] == 0

    def step(self, action: int):
        self.steps += 1
        dx, dy = self.actions[action]
        new_pos = (self.current_pos[0] + dx, self.current_pos[1] + dy)

        if self._valid(new_pos):
            # move
            self.current_pos = new_pos
            self.visited.add(new_pos)
            reward = self._reward(new_pos)
        else:
            reward = -10.0  # invalid move penalty (single)

        done = self.current_pos == self.goal or self.steps >= self.max_steps
        return self._state(), reward, done

    def _reward(self, new_pos):
        if new_pos == self.goal:
            return 100.0
        if new_pos in self.visited:
            return -2.0
        cur_dist = abs(self.current_pos[0] - self.goal[0]) + abs(self.current_pos[1] - self.goal[1])
        new_dist = abs(new_pos[0] - self.goal[0]) + abs(new_pos[1] - self.goal[1])
        return 1.0 if new_dist < cur_dist else -0.1


# ---------------------------------------------------------------------------
# Agent wrapping network & buffer
# ---------------------------------------------------------------------------

class DQNDeepForestAgent:
    def __init__(
        self,
        *,
        action_size: int = 4,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        buffer_size: int = 10000,
        update_interval: int = 100,
        n_estimators: int = 50,
        n_layers: int = 2,
    ):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.action_size = action_size

        self.q_net = DeepForestQNetwork(
            action_size=action_size,
            n_estimators=n_estimators,
            n_layers=n_layers,
        )
        self.target_net = copy.deepcopy(self.q_net)
        self.buffer = ReplayBuffer(buffer_size)
        self.learn_step = 0

    # ------------------------------------------------------------------
    # interaction helpers
    # ------------------------------------------------------------------

    def choose_action(self, state, training: bool):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        pos_idx = np.argwhere(state[1] == 1)
        if len(pos_idx):
            pos = tuple(map(int, pos_idx[0]))
            return int(np.argmax(self.q_net.predict(state, pos)))
        return random.randrange(self.action_size)

    def store(self, *args):
        self.buffer.push(*args)

    # ------------------------------------------------------------------
    # learning
    # ------------------------------------------------------------------

    def replay(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = self.buffer.sample(self.batch_size)
        self.q_net.fit(batch, target_network=self.target_net)
        self.learn_step += 1
        if self.learn_step % self.update_interval == 0:
            # shallow clone to reduce RAM
            self.target_net = copy.deepcopy(self.q_net)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# ---------------------------------------------------------------------------
# Solver orchestrating training / testing
# ---------------------------------------------------------------------------

class DQNDeepForestSolver:
    def __init__(
        self,
        *,
        episodes: int = 1000,
        max_steps_ep: int = 500,
        **agent_kwargs,
    ):
        self.episodes = episodes
        self.max_steps_ep = max_steps_ep
        self.agent = DQNDeepForestAgent(**agent_kwargs)

    def train(self, env: MazeEnvironment):
        ep_rewards: list[float] = []
        success_eps: list[int] = []
        total_step = 0
        for ep in range(self.episodes):
            s = env.reset()
            ep_r = 0.0
            for _ in range(self.max_steps_ep):
                a = self.agent.choose_action(s, training=True)
                s_next, r, done = env.step(a)
                self.agent.store(s, a, r, s_next, done)
                self.agent.replay()
                s = s_next
                ep_r += r
                total_step += 1
                if done:
                    if env.current_pos == env.goal:
                        success_eps.append(ep)
                    break
            ep_rewards.append(ep_r)
            self.agent.decay_epsilon()
            if ep % 100 == 0:
                avg_r = np.mean(ep_rewards[-100:])
                succ = sum(r > 50 for r in ep_rewards[-100:]) / min(100, len(ep_rewards))
                print(
                    f"Ep {ep:5d} | avgR {avg_r:7.2f} | succ {succ:.2%} | eps {self.agent.epsilon:.3f}"
                )
        return ep_rewards, success_eps

    def test(self, env: MazeEnvironment):
        s = env.reset()
        path = [env.current_pos]
        for step in range(self.max_steps_ep):
            a = self.agent.choose_action(s, training=False)
            s, _, done = env.step(a)
            path.append(env.current_pos)
            if done:
                break
        return env.current_pos == env.goal, path, step + 1

    # --------------------------------------------------------------
    # Public interface: solve one maze & produce result dataclass
    # --------------------------------------------------------------

    def solve(
        self,
        maze: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        *,
        maze_id: str | None = None,
    ) -> DQNDeepForestResult:
        t0 = time.time()
        res = DQNDeepForestResult(maze_id=maze_id or "", maze_size=maze.shape)
        if maze[start] == 1:
            res.failure_reason = "시작점이 벽입니다"
            res.execution_time = time.time() - t0
            return res
        if maze[goal] == 1:
            res.failure_reason = "목표점이 벽입니다"
            res.execution_time = time.time() - t0
            return res

        env = MazeEnvironment(maze, start, goal)
        t_train = time.time()
        ep_rewards, success_eps = self.train(env)
        res.training_time = time.time() - t_train
        res.training_episodes = self.episodes
        res.final_epsilon = self.agent.epsilon
        res.average_reward = float(np.mean(ep_rewards[-100:])) if ep_rewards else 0.0
        res.feature_importance = self.agent.q_net.feature_importance_
        if success_eps:
            res.convergence_episode = success_eps[0]

        success, path, steps = self.test(env)
        res.solution_found = success
        res.path = path if success else None
        res.solution_length = len(path) if success else 0
        res.total_steps = steps
        res.execution_time = time.time() - t0
        if not success:
            res.failure_reason = "훈련 완료 후에도 해결책을 찾지 못했습니다"
        return res


# ---------------------------------------------------------------------------
# Utility helpers (load image/metadata)
# ---------------------------------------------------------------------------

def load_maze_from_image(p: str) -> np.ndarray:
    img = Image.open(p).convert("L")
    maze = (np.asarray(img) < 128).astype(np.uint8)
    return maze


def load_metadata(p: str) -> Dict[str, Any]:
    import json

    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Simple smoke‑test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_maze = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    start, goal = (0, 0), (6, 6)
    print("Running quick DQN+DF test (200 episodes)…")
    solver = DQNDeepForestSolver(episodes=200, max_steps_ep=150)
    res = solver.solve(test_maze, start, goal)
    print("Solved:", res.solution_found)
    print("Len:", res.solution_length, "| AvgR:", f"{res.average_reward:.2f}")
    if res.feature_importance:
        for act, imp in res.feature_importance.items():
            for typ, vals in imp.items():
                top3 = np.argsort(vals)[-3:][::-1]
                top_vals = [f"{vals[i]:.3f}" for i in top3]
                print(f"{act}/{typ} top feat idx: {top3} | val {top_vals}")
