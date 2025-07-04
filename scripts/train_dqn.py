#!/usr/bin/env python3
"""
DQN 학습 및 벤치마크 실행 스크립트
RTX 3060 최적화 설정 포함
"""

import argparse
import json
import time
from pathlib import Path
import logging
import sys
import os

# 프로젝트 루트 디렉터리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from algorithms.dqn_benchmark_wrapper import DQNBenchmarkWrapper
from utils.maze_io import get_dataset_stats, get_loader
from utils.profiler import get_profiler
import numpy as np

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dqn_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# RTX 3060 최적화 기본 설정
RTX3060_OPTIMIZED_CONFIG = {
    'training_episodes': 2000,
    'max_steps': 1000,
    'learning_rate': 1e-3,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'memory_size': 8000,    # 6GB VRAM 고려하여 축소
    'batch_size': 16,       # RTX 3060에 최적화
    'target_update': 1000,
    'save_interval': 500,
    'device': 'auto'
}

# 미로 크기별 최적화 설정
SIZE_OPTIMIZED_CONFIGS = {
    'small': {   # 50x50 이하
        'batch_size': 32,
        'memory_size': 10000,
        'hidden_sizes': [256, 128, 64]
    },
    'medium': {  # 50x50 ~ 100x100
        'batch_size': 16,
        'memory_size': 8000,
        'hidden_sizes': [512, 256, 128]
    },
    'large': {   # 100x100 이상
        'batch_size': 8,
        'memory_size': 5000,
        'hidden_sizes': [512, 256]
    }
}

def get_size_category(maze_size: tuple) -> str:
    """미로 크기에 따른 카테고리 결정"""
    max_dim = max(maze_size)
    if max_dim <= 50:
        return 'small'
    elif max_dim <= 100:
        return 'medium'
    else:
        return 'large'

def load_and_check_dataset(subset: str = "train") -> dict:
    """데이터셋 로드 및 상태 확인"""
    try:
        stats = get_dataset_stats(subset)
        logger.info(f"{subset} 데이터셋 통계:")
        logger.info(f"  총 샘플 수: {stats['total_samples']}")
        logger.info(f"  크기 분포: {stats['sizes']}")
        return stats
    except Exception as e:
        logger.error(f"데이터셋 로드 실패: {e}")
        raise

def run_single_maze_training(args):
    """단일 미로 DQN 학습"""
    logger.info(f"단일 미로 DQN 학습 시작: {args.maze_id}")
    
    # 데이터셋 로더 초기화
    loader = get_loader()
    
    try:
        # 미로 데이터 로드
        maze_array, metadata = loader.load_maze_as_array(args.maze_id, args.subset)
        start = tuple(metadata.get('entrance', (0, 0)))
        goal = tuple(metadata.get('exit', (maze_array.shape[0]-1, maze_array.shape[1]-1)))
        
        logger.info(f"미로 정보: 크기={maze_array.shape}, 시작={start}, 목표={goal}")
        
        # 크기별 최적화 설정 적용
        size_category = get_size_category(maze_array.shape)
        config = RTX3060_OPTIMIZED_CONFIG.copy()
        config.update(SIZE_OPTIMIZED_CONFIGS[size_category])
        
        if args.episodes:
            config['training_episodes'] = args.episodes
        if args.max_steps:
            config['max_steps'] = args.max_steps
        if args.learning_rate:
            config['learning_rate'] = args.learning_rate
        
        logger.info(f"사용 설정 ({size_category}): {config}")
        
        # DQN 벤치마크 래퍼 생성
        wrapper = DQNBenchmarkWrapper(
            model_save_dir=args.model_dir,
            **config
        )
        
        # 벤치마크 실행
        result = wrapper.run_benchmark(
            maze_id=args.maze_id,
            subset=args.subset,
            force_retrain=args.force_retrain
        )
        
        # 결과 저장
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            result_data = {
                'maze_id': result.maze_id,
                'algorithm': result.algorithm,
                'success': result.solution_found,
                'execution_time': result.execution_time,
                'solution_length': result.solution_length,
                'total_steps': result.total_steps,
                'performance': {
                    'vram_usage': result.vram_usage,
                    'gpu_utilization': result.gpu_utilization,
                    'cpu_utilization': result.cpu_utilization,
                    'power_consumption': result.power_consumption
                },
                'training_stats': wrapper.get_training_stats(),
                'config': config
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"결과 저장: {output_path}")
        
        # 결과 출력
        print(f"\n=== DQN 학습 및 실행 결과 ===")
        print(f"미로 ID: {result.maze_id}")
        print(f"미로 크기: {result.maze_size}")
        print(f"해결 성공: {'✅' if result.solution_found else '❌'}")
        print(f"실행 시간: {result.execution_time:.2f}초")
        print(f"해결 경로 길이: {result.solution_length}")
        print(f"총 스텝: {result.total_steps}/{result.max_steps}")
        print(f"VRAM 사용량: {result.vram_usage:.1f}MB")
        print(f"GPU 사용률: {result.gpu_utilization:.1f}%")
        
        if not result.solution_found:
            print(f"실패 원인: {result.failure_reason}")
        
        # 학습 통계
        training_stats = wrapper.get_training_stats()
        print(f"\n=== 학습 통계 ===")
        print(f"학습 에피소드: {training_stats.get('total_episodes', 0)}")
        print(f"최종 성공률: {training_stats.get('final_success_rate', 0):.2f}")
        print(f"평균 보상: {training_stats.get('average_reward', 0):.2f}")
        
        convergence = training_stats.get('convergence_episode', -1)
        if convergence > 0:
            print(f"수렴 에피소드: {convergence}")
        else:
            print("수렴점을 찾지 못했습니다.")
        
        # 리소스 정리
        wrapper.cleanup()
        
        return result
        
    except Exception as e:
        logger.error(f"DQN 학습 실행 중 오류: {e}")
        raise

def run_batch_training(args):
    """배치 DQN 학습"""
    logger.info(f"배치 DQN 학습 시작: {args.batch_size}개 미로")
    
    # 데이터셋 통계 로드
    stats = load_and_check_dataset(args.subset)
    sample_ids = stats['sample_ids']
    
    # 배치 크기만큼 샘플 선택
    if args.batch_size > len(sample_ids):
        logger.warning(f"배치 크기가 전체 샘플 수보다 큽니다. {len(sample_ids)}개로 조정합니다.")
        args.batch_size = len(sample_ids)
    
    selected_ids = sample_ids[:args.batch_size]
    logger.info(f"선택된 미로 ID들: {selected_ids}")
    
    # 기본 설정
    config = RTX3060_OPTIMIZED_CONFIG.copy()
    if args.episodes:
        config['training_episodes'] = args.episodes
    if args.max_steps:
        config['max_steps'] = args.max_steps
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    
    # DQN 벤치마크 래퍼 생성
    wrapper = DQNBenchmarkWrapper(
        model_save_dir=args.model_dir,
        **config
    )
    
    # 배치 실행
    results = wrapper.batch_benchmark(
        maze_ids=selected_ids,
        subset=args.subset,
        force_retrain=args.force_retrain
    )
    
    # 결과 분석
    successful = [r for r in results if r.solution_found]
    success_rate = len(successful) / len(results) if results else 0
    
    avg_execution_time = np.mean([r.execution_time for r in results]) if results else 0
    avg_vram = np.mean([r.vram_usage for r in results]) if results else 0
    avg_solution_length = np.mean([r.solution_length for r in successful]) if successful else 0
    
    # 결과 출력
    print(f"\n=== 배치 DQN 학습 결과 ===")
    print(f"총 미로 수: {len(results)}")
    print(f"성공 미로 수: {len(successful)}")
    print(f"성공률: {success_rate:.2f}")
    print(f"평균 실행 시간: {avg_execution_time:.2f}초")
    print(f"평균 VRAM 사용량: {avg_vram:.1f}MB")
    print(f"평균 해결 경로 길이: {avg_solution_length:.1f}")
    
    # 결과 저장
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        batch_result = {
            'summary': {
                'total_mazes': len(results),
                'successful_mazes': len(successful),
                'success_rate': success_rate,
                'avg_execution_time': avg_execution_time,
                'avg_vram_usage': avg_vram,
                'avg_solution_length': avg_solution_length
            },
            'individual_results': [
                {
                    'maze_id': r.maze_id,
                    'success': r.solution_found,
                    'execution_time': r.execution_time,
                    'solution_length': r.solution_length,
                    'vram_usage': r.vram_usage,
                    'failure_reason': r.failure_reason
                }
                for r in results
            ],
            'config': config
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(batch_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"배치 결과 저장: {output_path}")
    
    # 리소스 정리
    wrapper.cleanup()
    
    return results

def main():
    parser = argparse.ArgumentParser(description='DQN 미로 해결 학습 및 벤치마크')
    
    # 기본 인수
    parser.add_argument('--maze-id', type=str, help='단일 미로 ID (예: 000001)')
    parser.add_argument('--batch-size', type=int, help='배치 학습할 미로 개수')
    parser.add_argument('--subset', type=str, default='train', choices=['train', 'valid', 'test'],
                       help='데이터셋 분할')
    
    # 학습 설정
    parser.add_argument('--episodes', type=int, help='학습 에피소드 수')
    parser.add_argument('--max-steps', type=int, help='에피소드당 최대 스텝 수')
    parser.add_argument('--learning-rate', type=float, help='학습률')
    parser.add_argument('--force-retrain', action='store_true', help='강제 재학습')
    
    # 출력 설정
    parser.add_argument('--output', type=str, help='결과 저장 경로 (JSON)')
    parser.add_argument('--model-dir', type=str, default='models/dqn', help='모델 저장 디렉터리')
    
    # 시스템 설정
    parser.add_argument('--dry-run', action='store_true', help='드라이 런 (설정만 확인)')
    parser.add_argument('--check-dataset', action='store_true', help='데이터셋만 확인')
    
    args = parser.parse_args()
    
    # 데이터셋 확인
    if args.check_dataset:
        load_and_check_dataset(args.subset)
        return
    
    # 드라이 런
    if args.dry_run:
        logger.info("드라이 런 모드: 설정만 확인합니다.")
        logger.info(f"RTX 3060 최적화 설정: {RTX3060_OPTIMIZED_CONFIG}")
        logger.info(f"크기별 설정: {SIZE_OPTIMIZED_CONFIGS}")
        return
    
    # RTX 3060 한계 체크
    profiler = get_profiler()
    limits_check = profiler.check_rtx3060_limits()
    
    logger.info("RTX 3060 상태 체크:")
    logger.info(f"  VRAM 사용률: {limits_check['vram_utilization_percent']:.1f}%")
    
    if limits_check['warnings']:
        logger.warning("RTX 3060 경고:")
        for warning in limits_check['warnings']:
            logger.warning(f"  - {warning}")
    
    # 실행 모드 결정
    if args.maze_id:
        # 단일 미로 학습
        run_single_maze_training(args)
    elif args.batch_size:
        # 배치 학습
        run_batch_training(args)
    else:
        parser.error("--maze-id 또는 --batch-size 중 하나를 지정해야 합니다.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        sys.exit(1)