# 🚀 빠른 시작 가이드

CNN과 Deep Forest를 학습시키고 ACO와 결합하여 미로 벤치마크를 실행하는 방법입니다.

## 📋 사전 준비

### 1. 환경 설정
```bash
# 필요한 패키지 설치
pip install torch torchvision numpy scikit-learn pillow matplotlib pandas joblib psutil

# NVIDIA GPU 모니터링 (선택사항)
pip install pynvml
```

### 2. 디렉터리 구조 확인
```
maze-benchmark/
├── algorithms/           # CNN, Deep Forest, 하이브리드 ACO
├── configs/             # 설정 파일
├── datasets/            # 미로 데이터 (train/valid/test)
├── scripts/             # 학습 및 실행 스크립트
├── utils/               # 유틸리티 (maze_io.py, profiler.py)
└── output/              # 결과 저장 (자동 생성)
```

## 🏃‍♂️ 1단계: 빠른 테스트

가장 간단한 방법으로 모든 것이 작동하는지 확인:

```bash
# 빠른 테스트 (적은 데이터, 간단한 설정)
python scripts/run_benchmark.py --quick-test

# GPU 상태만 확인
python scripts/run_benchmark.py --gpu-check
```

## 🤖 2단계: ML 모델 학습

### CNN과 Deep Forest 학습
```bash
# 전체 ML 모델 학습
python scripts/train_ml_models.py

# 특정 모델만 학습
python scripts/train_ml_models.py --models cnn
python scripts/train_ml_models.py --models deep_forest

# 빠른 테스트
python scripts/train_ml_models.py --dry-run
```

### 설정 파일 사용
`configs/ml_training.json` 파일을 수정하여 하이퍼파라미터 조정:

```json
{
  "cnn": {
    "epochs": 10,
    "batch_size": 8,
    "learning_rate": 0.001
  },
  "deep_forest": {
    "n_estimators": 100,
    "n_layers": 3,
    "max_depth": 15
  }
}
```

## 🎯 3단계: 하이브리드 벤치마크

### 전체 벤치마크 실행
```bash
# 전체 파이프라인 (ML 학습 + 하이브리드 평가)
python scripts/run_benchmark.py

# 기존 모델로 평가만
python scripts/run_benchmark.py --eval-only

# 강제 재학습
python scripts/run_benchmark.py --force-retrain
```

### 단계별 실행
```bash
# 1. ML 모델만 학습
python scripts/run_benchmark.py --ml-only

# 2. 평가만 실행 (기존 모델 사용)
python scripts/run_benchmark.py --eval-only
```

## 📊 4단계: 결과 확인

### 결과 파일 위치
```
output/
├── models/                    # 학습된 모델
│   ├── best_cnn_model.pth
│   └── deep_forest_model.joblib
├── results/                   # 평가 결과
│   ├── benchmark_report.md    # 📋 최종 리포트
│   ├── hybrid_evaluation.json
│   └── full_benchmark_results.json
└── logs/                      # 로그 파일
```

### 주요 결과 파일
- **`benchmark_report.md`**: 사람이 읽기 쉬운 최종 리포트
- **`hybrid_evaluation.json`**: 상세한 성능 데이터
- **`training_results.json`**: ML 모델 학습 결과

## ⚙️ 고급 사용법

### RTX 3060 최적화 설정
```bash
# 메모리 절약 모드
python scripts/run_benchmark.py --quick-test --test-mazes 10

# 특정 설정으로 실행
python scripts/run_benchmark.py --config configs/rtx3060_optimized.json
```

### 커스텀 설정 파일
```json
{
  "data_limits": {
    "max_train_samples": 1000,
    "max_valid_samples": 200,
    "max_test_samples": 50
  },
  "cnn": {
    "batch_size": 4,
    "epochs": 5
  },
  "evaluation": {
    "test_mazes": 20
  }
}
```

### 실시간 모니터링
```bash
# 별도 터미널에서 VRAM 모니터링
watch -n 1 nvidia-smi

# 성능 프로파일링과 함께 실행
python scripts/run_benchmark.py --verbose
```

## 🐛 문제 해결

### 일반적인 오류

1. **CUDA 메모리 부족**
   ```bash
   # 배치 크기 줄이기
   python scripts/run_benchmark.py --quick-test
   ```

2. **데이터셋을 찾을 수 없음**
   ```bash
   # 데이터셋 경로 확인
   ls datasets/train/img/
   ```

3. **모델 파일을 찾을 수 없음**
   ```bash
   # ML 모델 먼저 학습
   python scripts/run_benchmark.py --ml-only
   ```

### 성능 튜닝

RTX 3060 (6GB VRAM)에 최적화된 설정:

```json
{
  "cnn": {
    "batch_size": 8,
    "input_size": 200
  },
  "deep_forest": {
    "n_estimators": 50,
    "n_layers": 2
  },
  "aco_hybrid": {
    "n_ants": 30,
    "n_iterations": 50
  }
}
```

## 📈 예상 결과

### 벤치마크 완료 시 출력 예시
```
🎯 벤치마크 완료 요약
================================================================================

🏆 알고리즘 순위:
  🥇 1위: ACO+CNN
      성공률: 95.2%
      평균 경로: 127.3

  🥈 2위: ACO+DeepForest
      성공률: 93.8%
      평균 경로: 129.1

  🥉 3위: ACO
      성공률: 89.4%
      평균 경로: 135.7

✅ 최고 성공률: 95.2% (ACO+CNN)
⚡ 가장 빠른 실행: 1.234초 (ACO)

💻 최종 VRAM 사용률: 23.4%

📊 상세 결과는 output/results/ 디렉터리를 확인하세요!
================================================================================
```

## 🔗 다음 단계

1. **Isaac Sim 연동**: 실제 시뮬레이션 환경에서 테스트
2. **더 많은 알고리즘**: DQN, PPO 추가
3. **하이퍼파라미터 최적화**: Optuna 등을 이용한 자동 튜닝
4. **성능 비교**: 다른 하드웨어에서의 벤치마크

## 💡 팁

- 처음에는 `--quick-test`로 시작하세요
- VRAM 사용량을 주시하면서 배치 크기를 조정하세요
- 학습된 모델은 재사용되므로 한 번만 학습하면 됩니다
- 상세한 로그가 필요하면 `--verbose` 옵션을 사용하세요

---

**문제가 있으면 로그 파일(`training.log`, `benchmark.log`)을 확인하세요!** 🔍