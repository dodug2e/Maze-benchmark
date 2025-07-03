# 🌀 Maze Benchmark System Prompt – Lab-Seminar Edition (v2.0-Lite)

> **⚙️ NOTE 이 프로젝트는 향후 Isaac Sim과도 손쉽게 연동 가능하도록 설계되었습니다.**

---

## 1 🎯 목표

* **학부 랩세미나 통과용** 미로-탐색 벤치마크 프레임워크 구축
* 지원 알고리즘: **ACO, ACO + CNN/DeepForest, DQN, PPO, A\***
* 단일 CLI로 학습·평가·프로파일링 실행

---

## 2 📂 디렉터리 스켈레톤

```
maze-benchmark/
├── algorithms/          # ACO, DQN, PPO 등 (패러다임별 하위 모듈)
├── configs/             # *.yaml – Hydra/OMEGACONF
├── datasets/            # .npy 미로 데이터
├── scripts/             # train.py · profile.py · visualize.py
├── utils/               # 순수 함수·프로파일러
├── tests/               # pytest 단위·통합 테스트
├── docs/                # MkDocs (논문화용)
└── README.md            # ← 이 프롬프트 삽입
```

---

## 3 🛠️ Done Definition

| 축       | 기준                                          |
| ------- | ------------------------------------------- |
| **기능성** | CLI 한 줄로 5 알고리즘 학습·추론                       |
| **성능**  | RTX 3060 (6 GB) 기준, **FPS·VRAM 변동 ≤ ±10 %** |
| **재현성** | `seed=42`, Dockerfile 제공                    |
| **가독성** | `pylint ≥ 8.5/10`, 모듈 길이 ≤ 400 라인           |
| **확장성** | 새 알고리즘 추가 시 기존 코드 수정 ≤ 1 파일                 |

---

## 4 💨 빠른 시작

```bash
# 학습
python scripts/train.py algo=PPO maze=data/maze01.npy

# VRAM 프로파일링
python scripts/profile.py algo=ACO metric=vram
```

---

## 5 🔑 핵심 설계 포인트

1. **플러그인 구조** – `algorithms/__init__.py`에서 자동 탐색
2. **외부 YAML 설정** – `configs/*.yaml`로 하이퍼파라미터 관리
3. **경량 의존성** – 수정 없이 Colab·로컬·서버 실행 가능
4. **테스트 우선** – `pytest -q` 올그린 시만 커밋 병합

---

## 6 🧪 랩세미나 체크리스트

1. `poetry install` → 의존성 정상 설치
2. `pytest -q` → 모든 테스트 통과
3. `python scripts/train.py algo=DQN --dry-run` → 1 epoch OK
4. 결과 그래프 (`scripts/visualize.py`) 저장 확인
5. README + 발표 슬라이드에 **Isaac Sim 확장 가능** 문구 포함

---

### ✅ 이제 이 프롬프트를 `README.md` 최상단에 붙여두고 작업을 시작하세요!
