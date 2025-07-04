#!/usr/bin/env python3
"""
통합 학습 스크립트
Usage: python scripts/train.py algo=DQN subset=train --dry-run
"""

import argparse
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.maze_io import get_loader, get_dataset_stats
from utils.profiler import get_profiler, profile_execution
from algorithms import get_algorithm
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingConfig:
    """학습 설정"""
    def __init__(self, algo: str, subset: str = "train", dry_run: bool = False,
                 max_samples: int = None, seed: int = 42):
        self.algo = algo
        self.subset = subset
        self.dry_run = dry_run
        self.max_samples = max_samples or (10 if dry_run else None)
        self.seed = seed
        
        # 알고리즘별 기본 설정
        self.algo_configs = {
            'ACO': {'max_iterations': 100, 'num_ants': 30},
            'ACO_CNN': {'max_iterations': 100, 'num_ants': 30, 'cnn_epochs': 50},
            'ACO_DeepForest': {'max_iterations': 100, 'num_ants': 30, 'forest_estimators': 100},
            'DQN': {'episodes': 1000, 'epsilon': 0.1, 'learning_rate': 0.001},
            'DQN_DeepForest': {'episodes': 1000, 'epsilon': 0.1, 'forest_estimators': 100},
            'PPO': {'episodes': 1000, 'learning_rate': 0.0003, 'clip_ratio': 0.2},
            'A_STAR': {'heuristic': 'manhattan'}
        }
        
        if dry_run:
            # 드라이런 시 파라미터 축소
            for config in self.algo_configs.values():
                if 'max_iterations' in config:
                    config['max_iterations'] = 10
                if 'episodes' in config:
                    config['episodes'] = 10
                if 'cnn_epochs' in config:
                    config['cnn_epochs'] = 5

def load_training_data(config: TrainingConfig):
    """학습 데이터 로드"""
    logger.info(f"Loading {config.subset} dataset...")
    
    # 데이터 로더 초기화
    loader = get_loader()
    
    # 샘플 ID 목록 가져오기
    sample_ids = loader.get_sample_ids(config.subset)
    
    if config.max_samples:
        sample_ids = sample_ids[:config.max_samples]
    
    logger.info(f"Found {len(sample_ids)} samples in {config.subset} subset")
    
    # 배치 로드
    samples = loader.batch_load_samples(sample_ids, config.subset)
    
    logger.info(f"Successfully loaded {len(samples)} samples")
    return samples

def train_algorithm(algo_name: str, samples: list, config: TrainingConfig):
    """알고리즘 학습 실행"""
    logger.info(f"Starting {algo_name} training...")
    
    # 알고리즘 인스턴스 생성
    algorithm = get_algorithm(algo_name)
    if algorithm is None:
        raise ValueError(f"Algorithm '{algo_name}' not found")
    
    # 알고리즘 설정
    algo_config = config.algo_configs.get(algo_name, {})
    algorithm.configure(algo_config)
    
    # 성능 프로파일링과 함께 학습 실행
    profiler = get_profiler()
    
    results = {
        'algorithm': algo_name,
        'total_samples': len(samples),
        'successful_runs': 0,
        'failed_runs': 0,
        'average_solution_length': 0,
        'average_execution_time': 0,
        'performance_metrics': {}
    }
    
    successful_solutions = []
    execution_times = []
    
    with profile_execution(f"{algo_name} training on {len(samples)} samples"):
        for i, sample in enumerate(samples):
            try:
                logger.info(f"Processing sample {i+1}/{len(samples)}: {sample['id']}")
                
                # 미로 데이터 준비
                maze_array = sample['array']
                if maze_array is None:
                    # 이미지에서 배열 생성
                    maze_array = get_loader().convert_image_to_array(sample['image'])
                
                metadata = sample['metadata']
                
                # 알고리즘 실행
                start_time = time.time()
                result = algorithm.solve(maze_array, metadata)
                execution_time = time.time() - start_time
                
                execution_times.append(execution_time)
                
                if result.get('success', False):
                    results['successful_runs'] += 1
                    solution_length = result.get('solution_length', 0)
                    successful_solutions.append(solution_length)
                    logger.info(f"  ✓ Success: {solution_length} steps in {execution_time:.2f}s")
                else:
                    results['failed_runs'] += 1
                    logger.info(f"  ✗ Failed in {execution_time:.2f}s")
                    
            except Exception as e:
                logger.error(f"Error processing sample {sample['id']}: {e}")
                results['failed_runs'] += 1
                continue
    
    # 결과 계산
    if successful_solutions:
        results['average_solution_length'] = sum(successful_solutions) / len(successful_solutions)
    
    if execution_times:
        results['average_execution_time'] = sum(execution_times) / len(execution_times)
    
    # 성능 메트릭 수집
    results['performance_metrics'] = profiler.get_summary_stats()
    
    logger.info(f"Training completed for {algo_name}")
    logger.info(f"  Success rate: {results['successful_runs']}/{len(samples)} ({results['successful_runs']/len(samples)*100:.1f}%)")
    logger.info(f"  Average solution length: {results['average_solution_length']:.1f}")
    logger.info(f"  Average execution time: {results['average_execution_time']:.2f}s")
    
    return results

def save_results(results: Dict[Any, Any], output_path: str):
    """결과 저장"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Train maze-solving algorithms')
    parser.add_argument('algo', choices=['ACO', 'ACO_CNN', 'ACO_DeepForest', 
                                        'DQN', 'DQN_DeepForest', 'PPO', 'A_STAR'],
                       help='Algorithm to train')
    parser.add_argument('--subset', choices=['train', 'valid', 'test'], 
                       default='train', help='Dataset subset')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Run with limited samples for testing')
    parser.add_argument('--max-samples', type=int, 
                       help='Maximum number of samples to process')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    parser.add_argument('--output-dir', default='results', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # 시드 설정
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 설정 생성
    config = TrainingConfig(
        algo=args.algo,
        subset=args.subset,
        dry_run=args.dry_run,
        max_samples=args.max_samples,
        seed=args.seed
    )
    
    try:
        # 데이터 로드
        samples = load_training_data(config)
        
        # 알고리즘 학습
        results = train_algorithm(args.algo, samples, config)
        
        # 결과 저장
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"{args.output_dir}/{args.algo}_{args.subset}_{timestamp}.json"
        save_results(results, output_path)
        
        # RTX 3060 한계 체크
        profiler = get_profiler()
        limits_check = profiler.check_rtx3060_limits()
        if limits_check['warnings']:
            logger.warning("RTX 3060 limits check:")
            for warning in limits_check['warnings']:
                logger.warning(f"  - {warning}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()