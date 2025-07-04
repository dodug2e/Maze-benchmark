#!/usr/bin/env python3
"""
성능 프로파일링 스크립트
Usage: python scripts/profile.py algo=ACO metric=vram subset=test
"""

import argparse
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.maze_io import get_loader
from utils.profiler import get_profiler, profile_execution
from algorithms import get_algorithm
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProfilingConfig:
    """프로파일링 설정"""
    def __init__(self, algo: str, metric: str = "vram", subset: str = "test", 
                 duration: int = 60, num_samples: int = 10):
        self.algo = algo
        self.metric = metric
        self.subset = subset
        self.duration = duration
        self.num_samples = num_samples

def profile_algorithm_performance(config: ProfilingConfig) -> Dict[str, Any]:
    """알고리즘 성능 프로파일링"""
    logger.info(f"Starting performance profiling for {config.algo}")
    logger.info(f"Metric: {config.metric}, Subset: {config.subset}, Duration: {config.duration}s")
    
    # 데이터 로더 및 알고리즘 준비
    loader = get_loader()
    algorithm = get_algorithm(config.algo)
    
    if algorithm is None:
        raise ValueError(f"Algorithm '{config.algo}' not found")
    
    # 테스트 샘플 로드
    sample_ids = loader.get_sample_ids(config.subset)[:config.num_samples]
    samples = loader.batch_load_samples(sample_ids, config.subset)
    
    logger.info(f"Loaded {len(samples)} samples for profiling")
    
    # 프로파일러 초기화
    profiler = get_profiler()
    
    # 성능 측정 결과 저장
    results = {
        'algorithm': config.algo,
        'metric': config.metric,
        'subset': config.subset,
        'duration': config.duration,
        'num_samples': len(samples),
        'measurements': [],
        'summary': {},
        'rtx3060_limits': {}
    }
    
    # 실제 프로파일링 실행
    with profile_execution(f"{config.algo} profiling"):
        start_time = time.time()
        
        while time.time() - start_time < config.duration:
            for sample in samples:
                try:
                    # 미로 데이터 준비
                    maze_array = sample['array']
                    if maze_array is None:
                        maze_array = loader.convert_image_to_array(sample['image'])
                    
                    metadata = sample['metadata']
                    
                    # 실행 전 메트릭 수집
                    pre_metrics = profiler.get_current_metrics()
                    
                    # 알고리즘 실행
                    exec_start = time.time()
                    result = algorithm.solve(maze_array, metadata)
                    exec_time = time.time() - exec_start
                    
                    # 실행 후 메트릭 수집
                    post_metrics = profiler.get_current_metrics()
                    
                    # 측정값 기록
                    measurement = {
                        'sample_id': sample['id'],
                        'execution_time': exec_time,
                        'success': result.get('success', False),
                        'pre_metrics': pre_metrics.to_dict(),
                        'post_metrics': post_metrics.to_dict(),
                        'vram_delta': post_metrics.vram_used_mb - pre_metrics.vram_used_mb,
                        'gpu_utilization': post_metrics.gpu_percent,
                        'cpu_utilization': post_metrics.cpu_percent,
                        'power_consumption': post_metrics.power_watts
                    }
                    
                    results['measurements'].append(measurement)
                    
                    # 설정된 시간 초과 체크
                    if time.time() - start_time >= config.duration:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error profiling sample {sample['id']}: {e}")
                    continue
            
            # 설정된 시간 초과 체크
            if time.time() - start_time >= config.duration:
                break
    
    # 요약 통계 계산
    if results['measurements']:
        measurements = results['measurements']
        
        # 메트릭별 통계
        execution_times = [m['execution_time'] for m in measurements]
        vram_deltas = [m['vram_delta'] for m in measurements]
        gpu_utils = [m['gpu_utilization'] for m in measurements]
        cpu_utils = [m['cpu_utilization'] for m in measurements]
        power_consumption = [m['power_consumption'] for m in measurements if m['power_consumption'] > 0]
        
        def calculate_stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {'min': 0, 'max': 0, 'mean': 0, 'median': 0}
            
            values_sorted = sorted(values)
            return {
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
                'median': values_sorted[len(values_sorted) // 2]
            }
        
        results['summary'] = {
            'total_measurements': len(measurements),
            'success_rate': sum(1 for m in measurements if m['success']) / len(measurements),
            'execution_time': calculate_stats(execution_times),
            'vram_delta_mb': calculate_stats(vram_deltas),
            'gpu_utilization': calculate_stats(gpu_utils),
            'cpu_utilization': calculate_stats(cpu_utils),
            'power_consumption': calculate_stats(power_consumption)
        }
    
    # RTX 3060 한계 체크
    results['rtx3060_limits'] = profiler.check_rtx3060_limits()
    
    # 전체 프로파일링 통계
    results['profiling_stats'] = profiler.get_summary_stats()
    
    return results

def print_profiling_results(results: Dict[str, Any]):
    """프로파일링 결과 출력"""
    print(f"\n{'='*60}")
    print(f"프로파일링 결과: {results.get('algorithm', 'N/A')}")
    print(f"{'='*60}")

    summary = results.get('summary', {})
    print(f"측정 횟수: {summary.get('total_measurements', 0)}")
    print(f"성공률: {summary.get('success_rate', 0):.1%}")

    print(f"\n실행 시간 (초):")
    exec_stats = summary.get('execution_time', {})
    print(f"  평균: {exec_stats.get('mean', 0):.3f}s")
    print(f"  최소: {exec_stats.get('min', 0):.3f}s")
    print(f"  최대: {exec_stats.get('max', 0):.3f}s")
    print(f"  중간값: {exec_stats.get('median', 0):.3f}s")

    print(f"\nVRAM 사용량 변화 (MB):")
    vram_stats = summary.get('vram_delta_mb', {})
    print(f"  평균: {vram_stats.get('mean', 0):.1f}MB")
    print(f"  최소: {vram_stats.get('min', 0):.1f}MB")
    print(f"  최대: {vram_stats.get('max', 0):.1f}MB")

    print(f"\nGPU 사용률 (%):")
    gpu_stats = summary.get('gpu_utilization', {})
    print(f"  평균: {gpu_stats.get('mean', 0):.1f}%")
    print(f"  최대: {gpu_stats.get('max', 0):.1f}%")

    print(f"\nCPU 사용률 (%):")
    cpu_stats = summary.get('cpu_utilization', {})
    print(f"  평균: {cpu_stats.get('mean', 0):.1f}%")
    print(f"  최대: {cpu_stats.get('max', 0):.1f}%")

    power_stats = summary.get('power_consumption', {})
    if power_stats.get('mean', 0) > 0:
        print(f"\n전력 소비 (W):")
        print(f"  평균: {power_stats.get('mean', 0):.1f}W")
        print(f"  최대: {power_stats.get('max', 0):.1f}W")

    limits = results.get('rtx3060_limits', {})
    warnings = limits.get('warnings', [])
    if warnings:
        print(f"\n⚠️  RTX 3060 한계 경고:")
        for warning in warnings:
            print(f"  - {warning}")

    print(f"\nVRAM 사용률: {limits.get('vram_utilization_percent', 0):.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Profile maze-solving algorithms')
    parser.add_argument('algo', choices=['ACO', 'ACO_CNN', 'ACO_DeepForest', 
                                        'DQN', 'DQN_DeepForest', 'PPO', 'A_STAR'],
                       help='Algorithm to profile')
    parser.add_argument('--metric', choices=['vram', 'gpu', 'cpu', 'power'], 
                       default='vram', help='Primary metric to profile')
    parser.add_argument('--subset', choices=['train', 'valid', 'test'], 
                       default='test', help='Dataset subset')
    parser.add_argument('--duration', type=int, default=60, 
                       help='Profiling duration in seconds')
    parser.add_argument('--num-samples', type=int, default=10, 
                       help='Number of samples to cycle through')
    parser.add_argument('--output-dir', default='profiling_results', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # 설정 생성
    config = ProfilingConfig(
        algo=args.algo,
        metric=args.metric,
        subset=args.subset,
        duration=args.duration,
        num_samples=args.num_samples
    )
    
    try:
        # 프로파일링 실행
        results = profile_algorithm_performance(config)
        
        # 결과 출력
        print_profiling_results(results)
        
        # 결과 저장
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"{args.output_dir}/{args.algo}_{args.metric}_{timestamp}.json"
        
        os.makedirs(args.output_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Profiling results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Profiling failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()