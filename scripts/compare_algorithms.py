#!/usr/bin/env python3
"""
다중 알고리즘 성능 비교 스크립트
ACO, ACO+CNN, ACO+DeepForest, DQN 통합 벤치마크
"""

import argparse
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import sys
from typing import List, Dict
import algorithms.ppo_benchmark_wrapper as PPOBenchmarkWrapper

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from algorithms.dqn_benchmark_wrapper import DQNBenchmarkWrapper
from utils.maze_io import get_loader
from utils.profiler import get_profiler
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlgorithmComparator:
    """알고리즘 성능 비교 클래스"""
    
    def __init__(self, output_dir: str = "results/comparison"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 비교 결과 저장용
        self.results = []
        
        # 알고리즘별 래퍼 초기화
        self.algorithm_wrappers = {
            'DQN': DQNBenchmarkWrapper(
                training_episodes=1000,  # 비교용 단축
                max_steps=1000,
                model_save_dir="models/comparison"
            ),
            'PPO': PPOBenchmarkWrapper(
                total_timesteps=100000,  # 비교용 단축
                max_episode_steps=1000,
                model_save_dir="models/comparison"
            )
        }
        
    def run_single_comparison(self, 
                            algorithms: List[str], 
                            maze_id: str, 
                            subset: str = "test") -> Dict:
        """단일 미로에 대한 알고리즘 비교"""
        
        logger.info(f"미로 {maze_id} 알고리즘 비교 시작: {algorithms}")
        
        maze_results = {
            'maze_id': maze_id,
            'algorithms': {}
        }
        
        for algo in algorithms:
            logger.info(f"{algo} 실행 중...")
            
            try:
                if algo == 'DQN':
                    # DQN 실행
                    result = self.algorithm_wrappers['DQN'].run_benchmark(
                        maze_id=maze_id,
                        subset=subset
                    )
                    
                    algo_result = {
                        'algorithm': result.algorithm,
                        'success': result.solution_found,
                        'execution_time': result.execution_time,
                        'solution_length': result.solution_length,
                        'total_steps': result.total_steps,
                        'vram_usage': result.vram_usage,
                        'gpu_utilization': result.gpu_utilization,
                        'cpu_utilization': result.cpu_utilization,
                        'power_consumption': result.power_consumption,
                        'failure_reason': result.failure_reason
                    }
                    
                elif algo == 'PPO':
                    # PPO 실행
                    result = self.algorithm_wrappers['PPO'].run_benchmark(
                        maze_id=maze_id,
                        subset=subset
                    )
                    
                    algo_result = {
                        'algorithm': result.algorithm,
                        'success': result.solution_found,
                        'execution_time': result.execution_time,
                        'solution_length': result.solution_length,
                        'total_steps': result.total_steps,
                        'vram_usage': result.vram_usage,
                        'gpu_utilization': result.gpu_utilization,
                        'cpu_utilization': result.cpu_utilization,
                        'power_consumption': result.power_consumption,
                        'failure_reason': result.failure_reason,
                        'training_stats': result.additional_metrics.get('training_stats', {})
                    }
                    
                elif algo in ['ACO', 'ACO_CNN', 'ACO_DeepForest']:
                    # 기존 ACO 알고리즘들 실행 (구현된 것으로 가정)
                    algo_result = self._run_aco_algorithm(algo, maze_id, subset)
                    
                else:
                    logger.warning(f"알 수 없는 알고리즘: {algo}")
                    continue
                    
                maze_results['algorithms'][algo] = algo_result
                logger.info(f"{algo} 완료: {'성공' if algo_result['success'] else '실패'}")
                
            except Exception as e:
                logger.error(f"{algo} 실행 실패: {e}")
                maze_results['algorithms'][algo] = {
                    'algorithm': algo,
                    'success': False,
                    'error': str(e)
                }
        
        return maze_results
    
    def _run_aco_algorithm(self, algo: str, maze_id: str, subset: str) -> Dict:
        """ACO 알고리즘 실행 (기존 구현 연동)"""
        # 기존 ACO 구현이 있다고 가정하고 모의 결과 반환
        # 실제로는 기존 구현된 ACO 클래스들을 호출해야 함
        
        logger.warning(f"{algo} 모의 실행 - 실제 구현으로 교체 필요")
        
        # 모의 결과 (실제 구현 시 제거)
        import random
        success = random.choice([True, False])
        
        return {
            'algorithm': algo,
            'success': success,
            'execution_time': random.uniform(1.0, 10.0),
            'solution_length': random.randint(20, 100) if success else 0,
            'total_steps': random.randint(50, 500),
            'vram_usage': random.uniform(100, 1000),
            'gpu_utilization': random.uniform(10, 80),
            'cpu_utilization': random.uniform(20, 90),
            'power_consumption': random.uniform(50, 150),
            'failure_reason': None if success else 'timeout'
        }
    
    def run_batch_comparison(self, 
                           algorithms: List[str], 
                           maze_ids: List[str], 
                           subset: str = "test") -> List[Dict]:
        """배치 알고리즘 비교"""
        
        logger.info(f"배치 비교 시작: {len(maze_ids)}개 미로, {len(algorithms)}개 알고리즘")
        
        all_results = []
        
        for i, maze_id in enumerate(maze_ids):
            logger.info(f"진행률: {i+1}/{len(maze_ids)} - 미로 {maze_id}")
            
            try:
                maze_result = self.run_single_comparison(algorithms, maze_id, subset)
                all_results.append(maze_result)
                
                # 중간 결과 저장
                if (i + 1) % 5 == 0:
                    self._save_intermediate_results(all_results, f"batch_comparison_partial_{i+1}.json")
                    
            except Exception as e:
                logger.error(f"미로 {maze_id} 비교 실패: {e}")
                continue
        
        logger.info("배치 비교 완료")
        return all_results
    
    def _save_intermediate_results(self, results: List[Dict], filename: str):
        """중간 결과 저장"""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"중간 결과 저장: {filepath}")
    
    def analyze_results(self, results: List[Dict]) -> Dict:
        """결과 분석 및 통계 생성"""
        
        logger.info("결과 분석 시작...")
        
        # 알고리즘별 성능 집계
        algo_stats = {}
        
        all_algorithms = set()
        for result in results:
            all_algorithms.update(result['algorithms'].keys())
        
        for algo in all_algorithms:
            algo_results = []
            for result in results:
                if algo in result['algorithms']:
                    algo_results.append(result['algorithms'][algo])
            
            if algo_results:
                successful = [r for r in algo_results if r.get('success', False)]
                
                algo_stats[algo] = {
                    'total_mazes': len(algo_results),
                    'successful_mazes': len(successful),
                    'success_rate': len(successful) / len(algo_results),
                    'avg_execution_time': np.mean([r.get('execution_time', 0) for r in algo_results]),
                    'avg_solution_length': np.mean([r.get('solution_length', 0) for r in successful]) if successful else 0,
                    'avg_vram_usage': np.mean([r.get('vram_usage', 0) for r in algo_results]),
                    'avg_gpu_utilization': np.mean([r.get('gpu_utilization', 0) for r in algo_results]),
                    'avg_power_consumption': np.mean([r.get('power_consumption', 0) for r in algo_results])
                }
        
        analysis = {
            'summary': {
                'total_mazes': len(results),
                'algorithms_tested': list(all_algorithms),
                'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'algorithm_stats': algo_stats,
            'detailed_results': results
        }
        
        return analysis
    
    def generate_report(self, analysis: Dict, save_plots: bool = True) -> str:
        """분석 보고서 생성"""
        
        logger.info("보고서 생성 중...")
        
        # JSON 결과 저장
        analysis_file = self.output_dir / "algorithm_comparison_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        # CSV 요약 저장
        algo_stats = analysis['algorithm_stats']
        df = pd.DataFrame.from_dict(algo_stats, orient='index')
        csv_file = self.output_dir / "algorithm_comparison_summary.csv"
        df.to_csv(csv_file, encoding='utf-8-sig')
        
        # 시각화 생성
        if save_plots:
            self._create_comparison_plots(algo_stats)
        
        # 텍스트 보고서 생성
        report_file = self.output_dir / "algorithm_comparison_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("🌀 알고리즘 성능 비교 보고서\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"분석 일시: {analysis['summary']['analysis_date']}\n")
            f.write(f"테스트 미로 수: {analysis['summary']['total_mazes']}\n")
            f.write(f"비교 알고리즘: {', '.join(analysis['summary']['algorithms_tested'])}\n\n")
            
            f.write("📊 성능 요약\n")
            f.write("-" * 30 + "\n")
            
            for algo, stats in algo_stats.items():
                f.write(f"\n🔸 {algo}\n")
                f.write(f"  성공률: {stats['success_rate']:.2%}\n")
                f.write(f"  평균 실행시간: {stats['avg_execution_time']:.2f}초\n")
                f.write(f"  평균 경로길이: {stats['avg_solution_length']:.1f}\n")
                f.write(f"  평균 VRAM: {stats['avg_vram_usage']:.1f}MB\n")
                f.write(f"  평균 GPU 사용률: {stats['avg_gpu_utilization']:.1f}%\n")
                f.write(f"  평균 전력소비: {stats['avg_power_consumption']:.1f}W\n")
            
            # 순위 매기기
            f.write(f"\n🏆 성능 순위\n")
            f.write("-" * 30 + "\n")
            
            # 성공률 기준 순위
            success_ranking = sorted(algo_stats.items(), 
                                   key=lambda x: x[1]['success_rate'], 
                                   reverse=True)
            
            f.write("성공률 순위:\n")
            for i, (algo, stats) in enumerate(success_ranking, 1):
                f.write(f"  {i}. {algo}: {stats['success_rate']:.2%}\n")
            
            # 실행시간 기준 순위 (성공한 케이스만)
            time_ranking = sorted([(algo, stats) for algo, stats in algo_stats.items() 
                                 if stats['success_rate'] > 0], 
                                key=lambda x: x[1]['avg_execution_time'])
            
            f.write("\n실행속도 순위 (빠른 순):\n")
            for i, (algo, stats) in enumerate(time_ranking, 1):
                f.write(f"  {i}. {algo}: {stats['avg_execution_time']:.2f}초\n")
        
        logger.info(f"보고서 생성 완료: {report_file}")
        return str(report_file)
    
    def _create_comparison_plots(self, algo_stats: Dict):
        """비교 시각화 생성"""
        
        # 성능 메트릭 플롯
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        algorithms = list(algo_stats.keys())
        
        # 성공률
        success_rates = [algo_stats[algo]['success_rate'] for algo in algorithms]
        ax1.bar(algorithms, success_rates, color='skyblue', alpha=0.7)
        ax1.set_title('알고리즘별 성공률')
        ax1.set_ylabel('성공률')
        ax1.set_ylim(0, 1)
        
        # 실행시간
        exec_times = [algo_stats[algo]['avg_execution_time'] for algo in algorithms]
        ax2.bar(algorithms, exec_times, color='lightcoral', alpha=0.7)
        ax2.set_title('알고리즘별 평균 실행시간')
        ax2.set_ylabel('실행시간 (초)')
        
        # VRAM 사용량
        vram_usage = [algo_stats[algo]['avg_vram_usage'] for algo in algorithms]
        ax3.bar(algorithms, vram_usage, color='lightgreen', alpha=0.7)
        ax3.set_title('알고리즘별 평균 VRAM 사용량')
        ax3.set_ylabel('VRAM (MB)')
        
        # 전력 소비
        power_consumption = [algo_stats[algo]['avg_power_consumption'] for algo in algorithms]
        ax4.bar(algorithms, power_consumption, color='gold', alpha=0.7)
        ax4.set_title('알고리즘별 평균 전력 소비')
        ax4.set_ylabel('전력 (W)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "algorithm_comparison_plots.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("비교 플롯 저장 완료")
    
    def cleanup(self):
        """리소스 정리"""
        if 'DQN' in self.algorithm_wrappers:
            self.algorithm_wrappers['DQN'].cleanup()
        logger.info("알고리즘 비교기 정리 완료")

def main():
    parser = argparse.ArgumentParser(description='다중 알고리즘 성능 비교')
    
    parser.add_argument('--algorithms', type=str, required=True,
                       help='비교할 알고리즘 (쉼표로 구분, 예: ACO,DQN,ACO_CNN)')
    parser.add_argument('--maze-ids', type=str, required=True,
                       help='테스트할 미로 ID들 (쉼표로 구분, 예: 000001,000002,000003)')
    parser.add_argument('--subset', type=str, default='test',
                       choices=['train', 'valid', 'test'], help='데이터셋 분할')
    parser.add_argument('--output-dir', type=str, default='results/comparison',
                       help='결과 저장 디렉터리')
    parser.add_argument('--no-plots', action='store_true', help='플롯 생성 안 함')
    
    args = parser.parse_args()
    
    # 인수 파싱
    algorithms = [algo.strip() for algo in args.algorithms.split(',')]
    maze_ids = [maze_id.strip() for maze_id in args.maze_ids.split(',')]
    
    logger.info(f"알고리즘 비교 시작:")
    logger.info(f"  알고리즘: {algorithms}")
    logger.info(f"  미로 ID: {maze_ids}")
    logger.info(f"  데이터셋: {args.subset}")
    
    # 비교 실행
    comparator = AlgorithmComparator(args.output_dir)
    
    try:
        # 배치 비교 실행
        results = comparator.run_batch_comparison(algorithms, maze_ids, args.subset)
        
        # 결과 분석
        analysis = comparator.analyze_results(results)
        
        # 보고서 생성
        report_file = comparator.generate_report(analysis, not args.no_plots)
        
        print(f"\n🎉 비교 완료!")
        print(f"📁 결과 디렉터리: {args.output_dir}")
        print(f"📊 보고서: {report_file}")
        print(f"📈 요약 CSV: {args.output_dir}/algorithm_comparison_summary.csv")
        
        if not args.no_plots:
            print(f"📊 시각화: {args.output_dir}/algorithm_comparison_plots.png")
        
    except Exception as e:
        logger.error(f"비교 실행 실패: {e}")
        raise
    finally:
        comparator.cleanup()

if __name__ == "__main__":
    main()