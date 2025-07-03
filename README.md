# 🌀 Maze‑Benchmark

> **Lab‑Seminar Edition (v2.0‑Lite)** – *Ready for Isaac Sim extension*

---

## 1 🎯 Purpose

Build an **extensible yet lightweight** benchmark framework for maze‑solving algorithms (ACO variants, DQN, PPO, A\*, …) that will pass the upcoming undergraduate lab seminar *and* scale later to Isaac Sim.

---

## 2 📁 Directory Layout

```
maze-benchmark/
├── algorithms/          # ACO, DQN, PPO … (one file per algo)
├── configs/             # YAML – hyper‑parameters
├── datasets/            # ✓  see Section 3
├── scripts/             # train.py · profile.py · visualize.py
├── utils/               # maze_io.py · profiler.py · seed.py
├── tests/               # pytest test_*.py
├── docs/                # MkDocs source
├── maze_runner.py       # CLI entry‑point
├── pyproject.toml       # poetry build
├── Dockerfile           # CUDA + CPU compatible
└── .github/workflows/ci.yml  # lint + tests + docker build
```

---

## 3 📦 Dataset Structure

The dataset is **pre‑split** into `train (10 000)`, `valid (8 000)` and `test (2 000)` samples.

```
datasets/
├── train/
│   ├── img/            # 000001.png …
│   ├── metadata/       # 000001.json …
│   ├── arrays/         # 000001.npy  …
│   └── dataset_info.json
├── valid/
│   └── …               # identical sub‑folders
└── test/
    └── …
```

| sub‑folder          | content                                             | note                                 |
| ------------------- | --------------------------------------------------- | ------------------------------------ |
| `img/`              | 2‑D maze raster as **PNG**                          | 1 file = 1 sample                    |
| `metadata/`         | per‑maze **JSON** (start/goal coords, name, etc.)   | *1:1 mapping to img*                 |
| `arrays/`           | **NumPy .npy** tensor (optional pre‑processed form) | shape & dtype in `dataset_info.json` |
| `dataset_info.json` | { `num_samples`, `version`, `sha256` }              | used by CI integrity test            |

\### 3.1 Loading API

```python
from utils.maze_io import load_sample
img, meta, arr = load_sample("000001", subset="train")
```

`utils/maze_io.py` hides the folder logic and returns PIL Image, JSON‑dict and optionally the NumPy array.

\### 3.2 Integrity Check
Run once after cloning or in CI:

```bash
python scripts/inspect_dataset.py
```

*Verifies*: PNG = JSON = NPY count, checksum matches `dataset_info.json`.

---

## 4 ⚡ Quick Start

```bash
# Install
poetry install

# Train (single epoch dry‑run)
python scripts/train.py algo=PPO subset=train --dry-run

# Profile VRAM
python scripts/profile.py algo=ACO metric=vram subset=test
```

---

## 5 ✅ Done Definition

| axis                | criterion                                     |
| ------------------- | --------------------------------------------- |
| **Functionality**   | CLI one‑liner trains & tests all 5 algorithms |
| **Performance**     | RTX 3060 (6 GB) ±10 % FPS & VRAM vs baseline  |
| **Reproducibility** | `seed=42`, Dockerfile provided                |
| **Readability**     | `pylint ≥ 8.5/10`, each module ≤ 400 LOC      |
| **Extensibility**   | New algorithm = add 1 file in `algorithms/`   |

---

## 6 🧪 Lab‑Seminar Checklist

1. `poetry install` → deps OK
2. `pytest -q` → all tests green
3. `python scripts/train.py algo=DQN --dry-run` → 1 epoch OK
4. `scripts/visualize.py` → graphs saved to `docs/`
5. Slides & README mention **Isaac Sim capable**

---

## 7 🚀 Isaac Sim Extension (future‑proof)

*Replace* the Docker base image with NVIDIA Isaac and plug the simulator feed into `utils/maze_io.py` adapter hooks. No core changes required.

---

Happy benchmarking! 🎉
