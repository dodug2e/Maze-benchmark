#!/usr/bin/env python3
"""
미로 벤치마크 메인 실행 스크립트
CNN, Deep Forest, 하이브리드 ACO 통합 실행
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 모듈 import
from utils.maze_io import get_loader
from utils.profiler import get_profiler, profile_execution
from scripts.train_ml_models import MLModelTrainer
from algorithms.hybrid_aco import HybridSolver

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MazeBenchmark:
    """미로 알고리즘 벤치마크 시스템"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.maze_loader = get_loader(self.config.get('dataset_path', 'datasets'))
        self.profiler = get_profiler()
        
        # 결과 디렉터리 설정
        self.output_dir = Path(self.config.get('output_dir', 'output'))
        self.output_dir.mkdir(exist_ok=True)
        
        self.models_dir = self.output_dir / 'models'
        self.results_dir = self.output_dir / 'results'
        self.logs_dir = self.output_dir / 'logs'
        
        for directory in [self.models_dir, self.results_dir, self.logs_dir]:
            directory.mkdir(exist_ok=True)
        
        # 결과 저장
        self.benchmark_results = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"설정 파일을 찾을 수 없음: {config_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"설정 파일 파싱 오류: {e}")
            sys.exit(1)
    
    def check_system_requirements(self) -> bool:
        """시스템 요구사항 확인"""
        logger.info("시스템 요구사항 확인 중...")
        
        # GPU 상태 확인
        gpu_status = self.profiler.check_rtx3060_limits()
        logger.info(f"초기 VRAM 사용률: {gpu_status['vram_utilization_percent']:.1f}%")
        
        hardware_limits = self.config.get('hardware', {})
        gpu_limit = hardware_limits.get('gpu_memory_limit_mb', 6144)
        
        if gpu_status['current_metrics']['vram_used_mb'] > gpu_limit * 0.8:
            logger.warning(f"VRAM 사용량이 높습니다: {gpu_status['current_metrics']['vram_used_mb']:.1f}MB")
        
        # GPU 경고 출력
        if gpu_status['warnings']:
            logger.warning("GPU 상태 경고:")
            for warning in gpu_status['warnings']:
                logger.warning(f"  - {warning}")
        
        # 데이터셋 확인
        try:
            train_samples = len(self.maze_loader.get_sample_ids('train'))
            valid_samples = len(self.maze_loader.get_sample_ids('valid'))
            test_samples = len(self.maze_loader.get_sample_ids('test'))
            
            logger.info(f"데이터셋 크기: train={train_samples}, valid={valid_samples}, test={test_samples}")
            
            if train_samples == 0:
                logger.error("훈련 데이터가 없습니다!")
                return False
                
        except Exception as e:
            logger.error(f"데이터셋 로드 실패: {e}")
            return False
        
        return True
    
    def train_ml_models(self) -> Dict:
        """ML 모델들 학습"""
        logger.info("\n" + "="*60)
        logger.info("ML 모델 학습 단계")
        logger.info("="*60)
        
        # 이미 학습된 모델이 있는지 확인
        cnn_model_path = self.models_dir / 'best_cnn_model.pth'
        df_model_path = self.models_dir / 'deep_forest_model.joblib'
        
        force_retrain = self.config.get('force_retrain', False)
        
        if (cnn_model_path.exists() and df_model_path.exists() and not force_retrain):
            logger.info("기존 학습된 모델 발견, 학습 단계를 건너뜁니다.")
            logger.info("재학습을 원하면 'force_retrain': true 설정하세요.")
            
            return {
                'cnn': {'model_path': str(cnn_model_path), 'skipped': True},
                'deep_forest': {'model_path': str(df_model_path), 'skipped': True}
            }
        
        # ML 모델 학습 실행
        trainer = MLModelTrainer(self.config)
        
        with profile_execution("ML 모델 학습"):
            results = trainer.run_training()
            trainer.generate_report(results)
        
        self.benchmark_results['ml_training'] = results
        return results
    
    def run_hybrid_evaluation(self, ml_results: Dict) -> Dict:
        """하이브리드 알고리즘 평가"""
        logger.info("\n" + "="*60)
        logger.info("하이브리드 알고리즘 평가 단계")
        logger.info("="*60)
        
        # 테스트할 미로 선택
        test_ids = self.maze_loader.get_sample_ids('test')
        eval_config = self.config.get('evaluation', {})
        max_test_mazes = eval_config.get('test_mazes', 50)
        
        if len(test_ids) > max_test_mazes:
            test_ids = test_ids[:max_test_mazes]
        
        logger.info(f"평가할 미로 개수: {len(test_ids)}")
        
        # 하이브리드 알고리즘 결과 저장
        hybrid_results = {
            'total_mazes': len(test_ids),
            'algorithms': {},
            'comparison': {},
            'individual_results': []
        }
        
        # ACO 설정
        aco_config = self.config.get('aco_hybrid', {})
        
        success_counts = {'aco': 0, 'aco_cnn': 0, 'aco_df': 0}
        total_times = {'aco': 0, 'aco_cnn': 0, 'aco_df': 0}
        total_path_lengths = {'aco': [], 'aco_cnn': [], 'aco_df': []}
        
        # 각 미로에 대해 평가
        for i, sample_id in enumerate(test_ids):
            logger.info(f"\n미로 {i+1}/{len(test_ids)} 평가 중... (ID: {sample_id})")
            
            try:
                # 미로 데이터 로드
                img, metadata, array = self.maze_loader.load_sample(sample_id, 'test')
                
                if array is not None:
                    maze_array = array
                else:
                    maze_array = self.maze_loader.convert_image_to_array(img)
                
                # 시작점과 끝점
                start_pos = tuple(metadata.get('entrance', (1, 1)))
                goal_pos = tuple(metadata.get('exit', (maze_array.shape[0]-2, maze_array.shape[1]-2)))
                
                # 하이브리드 솔버 생성
                solver = HybridSolver(maze_array, start_pos, goal_pos)
                
                # 모델 로드
                cnn_path = None
                df_path = None
                
                if 'cnn' in ml_results and 'model_paths' in ml_results['cnn']:
                    cnn_path = ml_results['cnn']['model_paths'].get('best')
                elif 'cnn' in ml_results and 'model_path' in ml_results['cnn']:
                    cnn_path = ml_results['cnn']['model_path']
                
                if 'deep_forest' in ml_results and 'model_path' in ml_results['deep_forest']:
                    df_path = ml_results['deep_forest']['model_path']
                
                if cnn_path and os.path.exists(cnn_path):
                    solver.load_cnn_model(cnn_path)
                
                if df_path and os.path.exists(df_path):
                    solver.load_deep_forest_model(df_path)
                
                # 모든 알고리즘으로 해결
                maze_results = solver.solve_all(**aco_config)
                
                # 결과 집계
                maze_result = {
                    'maze_id': sample_id,
                    'maze_size': maze_array.shape,
                    'start_pos': start_pos,
                    'goal_pos': goal_pos,
                    'algorithms': maze_results
                }
                
                for algo_name, result in maze_results.items():
                    if 'error' not in result:
                        if result['success']:
                            success_counts[algo_name] += 1
                            total_path_lengths[algo_name].append(result['path_length'])
                        
                        total_times[algo_name] += result['execution_time']
                
                hybrid_results['individual_results'].append(maze_result)
                
                # 진행률 출력
                if (i + 1) % 10 == 0:
                    current_success_rates = {
                        name: count / (i + 1) for name, count in success_counts.items()
                    }
                    logger.info(f"진행률: {i+1}/{len(test_ids)}, 성공률: {current_success_rates}")
                
            except Exception as e:
                logger.error(f"미로 {sample_id} 평가 실패: {e}")
                continue
        
        # 최종 결과 계산
        for algo_name in success_counts.keys():
            avg_time = total_times[algo_name] / len(test_ids)
            success_rate = success_counts[algo_name] / len(test_ids)
            avg_path_length = (sum(total_path_lengths[algo_name]) / len(total_path_lengths[algo_name]) 
                             if total_path_lengths[algo_name] else 0)
            
            hybrid_results['algorithms'][algo_name] = {
                'success_count': success_counts[algo_name],
                'success_rate': success_rate,
                'avg_execution_time': avg_time,
                'avg_path_length': avg_path_length,
                'total_successful_paths': len(total_path_lengths[algo_name])
            }
        
        # 알고리즘 비교
        hybrid_results['comparison'] = self._compare_algorithms(hybrid_results['algorithms'])
        
        # 결과 저장
        results_file = self.results_dir / 'hybrid_evaluation.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(hybrid_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"하이브리드 평가 완료 - 결과 저장: {results_file}")
        
        return hybrid_results
    
    def _compare_algorithms(self, algorithm_results: Dict) -> Dict:
        """알고리즘 성능 비교"""
        comparison = {
            'best_success_rate': {'algorithm': None, 'value': 0},
            'best_avg_path_length': {'algorithm': None, 'value': float('inf')},
            'fastest_execution': {'algorithm': None, 'value': float('inf')},
            'ranking': []
        }
        
        for algo_name, results in algorithm_results.items():
            # 최고 성공률
            if results['success_rate'] > comparison['best_success_rate']['value']:
                comparison['best_success_rate'] = {
                    'algorithm': algo_name,
                    'value': results['success_rate']
                }
            
            # 최단 평균 경로
            if (results['avg_path_length'] > 0 and 
                results['avg_path_length'] < comparison['best_avg_path_length']['value']):
                comparison['best_avg_path_length'] = {
                    'algorithm': algo_name,
                    'value': results['avg_path_length']
                }
            
            # 가장 빠른 실행
            if results['avg_execution_time'] < comparison['fastest_execution']['value']:
                comparison['fastest_execution'] = {
                    'algorithm': algo_name,
                    'value': results['avg_execution_time']
                }
        
        # 전체 랭킹 (성공률 우선, 그 다음 경로 길이)
        ranking_data = []
        for algo_name, results in algorithm_results.items():
            score = results['success_rate'] * 1000 - results['avg_path_length']
            ranking_data.append((algo_name, score, results))
        
        ranking_data.sort(key=lambda x: x[1], reverse=True)
        comparison['ranking'] = [
            {
                'rank': i + 1,
                'algorithm': algo_name,
                'score': score,
                'success_rate': results['success_rate'],
                'avg_path_length': results['avg_path_length']
            }
            for i, (algo_name, score, results) in enumerate(ranking_data)
        ]
        
        return comparison
    
    def generate_final_report(self, ml_results: Dict, hybrid_results: Dict):
        """최종 벤치마크 리포트 생성"""
        logger.info("최종 벤치마크 리포트 생성 중...")
        
        report_path = self.results_dir / 'benchmark_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 미로 알고리즘 벤치마크 결과\n\n")
            f.write(f"**생성일시**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**하드웨어**: RTX 3060 (6GB VRAM)\n\n")
            
            # ML 모델 학습 결과
            f.write("## 1. ML 모델 학습 결과\n\n")
            
            if 'cnn' in ml_results and 'error' not in ml_results['cnn']:
                cnn = ml_results['cnn']
                if not cnn.get('skipped', False):
                    f.write("### CNN 모델\n")
                    f.write(f"- **최고 검증 정확도**: {cnn.get('best_val_accuracy', 0):.4f}\n")
                    f.write(f"- **학습 시간**: {cnn.get('training_time', 0):.2f}초\n")
                    f.write(f"- **모델 파라미터**: {cnn.get('model_info', {}).get('total_parameters', 0):,}\n")
                    f.write(f"- **피크 VRAM**: {cnn.get('performance', {}).get('vram_used_mb', {}).get('peak', 0):.1f}MB\n\n")
                else:
                    f.write("### CNN 모델\n- **상태**: 기존 모델 사용 (학습 건너뜀)\n\n")
            
            if 'deep_forest' in ml_results and 'error' not in ml_results['deep_forest']:
                df = ml_results['deep_forest']
                if not df.get('skipped', False):
                    f.write("### Deep Forest 모델\n")
                    f.write(f"- **테스트 정확도**: {df.get('test_accuracy', 0):.4f}\n")
                    f.write(f"- **학습 시간**: {df.get('training_time', 0):.2f}초\n")
                    f.write(f"- **최적 레이어 수**: {df.get('best_layer_count', 0)}\n")
                    f.write(f"- **총 트리 수**: {df.get('model_info', {}).get('total_trees', 'N/A')}\n\n")
                else:
                    f.write("### Deep Forest 모델\n- **상태**: 기존 모델 사용 (학습 건너뜀)\n\n")
            
            # 하이브리드 알고리즘 결과
            f.write("## 2. 하이브리드 알고리즘 성능 비교\n\n")
            
            f.write(f"**평가 미로 수**: {hybrid_results.get('total_mazes', 0)}개\n\n")
            
            # 성능 테이블
            f.write("### 알고리즘별 성능\n\n")
            f.write("| 알고리즘 | 성공률 | 평균 경로 길이 | 평균 실행 시간 |\n")
            f.write("|----------|--------|--------------|---------------|\n")
            
            algo_names = {'aco': 'ACO', 'aco_cnn': 'ACO+CNN', 'aco_df': 'ACO+DeepForest'}
            
            for algo_key, algo_display in algo_names.items():
                if algo_key in hybrid_results.get('algorithms', {}):
                    results = hybrid_results['algorithms'][algo_key]
                    f.write(f"| {algo_display} | {results['success_rate']:.3f} | "
                           f"{results['avg_path_length']:.1f} | "
                           f"{results['avg_execution_time']:.3f}초 |\n")
            
            f.write("\n")
            
            # 최고 성능
            f.write("### 최고 성능\n\n")
            comparison = hybrid_results.get('comparison', {})
            
            if 'best_success_rate' in comparison:
                best_sr = comparison['best_success_rate']
                algo_name = algo_names.get(best_sr['algorithm'], best_sr['algorithm'])
                f.write(f"- **최고 성공률**: {algo_name} ({best_sr['value']:.3f})\n")
            
            if 'best_avg_path_length' in comparison:
                best_pl = comparison['best_avg_path_length']
                algo_name = algo_names.get(best_pl['algorithm'], best_pl['algorithm'])
                f.write(f"- **최단 평균 경로**: {algo_name} ({best_pl['value']:.1f})\n")
            
            if 'fastest_execution' in comparison:
                fastest = comparison['fastest_execution']
                algo_name = algo_names.get(fastest['algorithm'], fastest['algorithm'])
                f.write(f"- **가장 빠른 실행**: {algo_name} ({fastest['value']:.3f}초)\n")
            
            f.write("\n")
            
            # 전체 랭킹
            f.write("### 전체 랭킹\n\n")
            if 'ranking' in comparison:
                for rank_info in comparison['ranking']:
                    algo_name = algo_names.get(rank_info['algorithm'], rank_info['algorithm'])
                    f.write(f"{rank_info['rank']}. **{algo_name}** - "
                           f"성공률: {rank_info['success_rate']:.3f}, "
                           f"평균 경로: {rank_info['avg_path_length']:.1f}\n")
            
            f.write("\n")
            
            # 결론
            f.write("## 3. 결론\n\n")
            
            if comparison.get('ranking'):
                winner = comparison['ranking'][0]
                winner_name = algo_names.get(winner['algorithm'], winner['algorithm'])
                f.write(f"**최고 성능 알고리즘**: {winner_name}\n\n")
                
                if winner['algorithm'] == 'aco':
                    f.write("기본 ACO가 가장 좋은 성능을 보였습니다. ")
                    f.write("ML 모델의 추가적인 학습이나 하이퍼파라미터 튜닝이 필요할 수 있습니다.\n\n")
                else:
                    f.write("하이브리드 접근법이 기본 ACO보다 우수한 성능을 보였습니다. ")
                    f.write("ML 모델의 도움으로 더 효율적인 경로 탐색이 가능했습니다.\n\n")
            
            # 시스템 정보
            final_gpu_status = self.profiler.check_rtx3060_limits()
            f.write("## 4. 시스템 정보\n\n")
            f.write(f"- **최종 VRAM 사용률**: {final_gpu_status['vram_utilization_percent']:.1f}%\n")
            f.write(f"- **피크 GPU 온도**: {final_gpu_status['current_metrics']['temperature_c']:.1f}°C\n")
            f.write(f"- **총 벤치마크 시간**: {time.time() - self.benchmark_start_time:.0f}초\n")
        
        logger.info(f"최종 리포트 생성 완료: {report_path}")
        
        # 콘솔에 요약 출력
        self._print_summary(hybrid_results)
    
    def _print_summary(self, hybrid_results: Dict):
        """콘솔에 요약 결과 출력"""
        print("\n" + "="*80)
        print("벤치마크 완료 요약")
        print("="*80)
        
        comparison = hybrid_results.get('comparison', {})
        
        if 'ranking' in comparison and comparison['ranking']:
            print("\n알고리즘 순위:")
            for i, rank_info in enumerate(comparison['ranking'][:3]):
                medal = "1등" if i == 0 else "2등" if i == 1 else "3등"
                algo_names = {'aco': 'ACO', 'aco_cnn': 'ACO+CNN', 'aco_df': 'ACO+DeepForest'}
                algo_name = algo_names.get(rank_info['algorithm'], rank_info['algorithm'])
                
                print(f"  {medal}: {algo_name}")
                print(f"      성공률: {rank_info['success_rate']:.1%}")
                print(f"      평균 경로: {rank_info['avg_path_length']:.1f}")
        
        # 주요 지표 (None 체크 추가)
        if 'best_success_rate' in comparison and comparison['best_success_rate']['algorithm'] is not None:
            best_sr = comparison['best_success_rate']
            algo_name = best_sr['algorithm'].upper() if best_sr['algorithm'] else 'UNKNOWN'
            print(f"\n최고 성공률: {best_sr['value']:.1%} ({algo_name})")
        
        if 'fastest_execution' in comparison and comparison['fastest_execution']['algorithm'] is not None:
            fastest = comparison['fastest_execution']
            algo_name = fastest['algorithm'].upper() if fastest['algorithm'] else 'UNKNOWN'
            print(f"가장 빠른 실행: {fastest['value']:.3f}초 ({algo_name})")
        
        # GPU 상태
        try:
            gpu_status = self.profiler.check_rtx3060_limits()
            print(f"\n최종 VRAM 사용률: {gpu_status['vram_utilization_percent']:.1f}%")
        except Exception as e:
            print(f"\nGPU 상태 확인 실패: {e}")
        
        print("\n상세 결과는 output/results/ 디렉터리를 확인하세요!")
        print("="*80)
    
    def run_full_benchmark(self) -> Dict:
        """전체 벤치마크 실행"""
        self.benchmark_start_time = time.time()
        
        logger.info("미로 벤치마크 시작")
        logger.info(f"설정: {len(self.config.get('models', []))}개 ML 모델")
        
        # 시스템 요구사항 확인
        if not self.check_system_requirements():
            logger.error("시스템 요구사항을 만족하지 않습니다.")
            return {}
        
        try:
            # 1단계: ML 모델 학습
            ml_results = self.train_ml_models()
            
            # GPU 메모리 정리
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            
            # 2단계: 하이브리드 알고리즘 평가
            hybrid_results = self.run_hybrid_evaluation(ml_results)
            
            # 3단계: 최종 리포트 생성
            try:
                self.generate_final_report(ml_results, hybrid_results)
            except Exception as e:
                logger.error(f"리포트 생성 실패: {e}")
                # 리포트 생성 실패해도 계속 진행
            
            # 전체 결과 반환
            full_results = {
                'ml_training': ml_results,
                'hybrid_evaluation': hybrid_results,
                'total_time': time.time() - self.benchmark_start_time,
                'config_used': self.config
            }
            
            # 전체 결과 저장
            full_results_path = self.results_dir / 'full_benchmark_results.json'
            with open(full_results_path, 'w', encoding='utf-8') as f:
                json.dump(full_results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"전체 벤치마크 완료! 총 소요 시간: {full_results['total_time']:.0f}초")
            
            return full_results
            
        except KeyboardInterrupt:
            logger.info("사용자에 의해 벤치마크가 중단되었습니다.")
            return {}
        except Exception as e:
            logger.error(f"벤치마크 실행 중 오류 발생: {e}")
            raise


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="미로 알고리즘 벤치마크 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 전체 벤치마크 실행
  python scripts/run_benchmark.py

  # 특정 설정 파일로 실행
  python scripts/run_benchmark.py --config configs/my_config.json

  # 빠른 테스트 (적은 데이터)
  python scripts/run_benchmark.py --quick-test

  # ML 모델만 학습
  python scripts/run_benchmark.py --ml-only

  # 평가만 실행 (기존 모델 사용)
  python scripts/run_benchmark.py --eval-only
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/ml_training.json',
        help='설정 파일 경로 (기본값: configs/ml_training.json)'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        help='결과 저장 디렉터리 (설정 파일의 값을 오버라이드)'
    )
    
    parser.add_argument(
        '--quick-test', 
        action='store_true',
        help='빠른 테스트 모드 (적은 데이터와 간단한 설정)'
    )
    
    parser.add_argument(
        '--ml-only', 
        action='store_true',
        help='ML 모델 학습만 실행'
    )
    
    parser.add_argument(
        '--eval-only', 
        action='store_true',
        help='평가만 실행 (기존 학습된 모델 사용)'
    )
    
    parser.add_argument(
        '--force-retrain', 
        action='store_true',
        help='기존 모델이 있어도 강제로 재학습'
    )
    
    parser.add_argument(
        '--gpu-check', 
        action='store_true',
        help='GPU 상태만 확인하고 종료'
    )
    
    parser.add_argument(
        '--models', 
        nargs='+', 
        choices=['cnn', 'deep_forest'],
        help='학습할 모델 선택 (기본값: 모든 모델)'
    )
    
    parser.add_argument(
        '--test-mazes', 
        type=int,
        help='평가에 사용할 미로 개수'
    )
    
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true',
        help='상세 로그 출력'
    )
    
    return parser.parse_args()


def modify_config_for_quick_test(config: Dict) -> Dict:
    """빠른 테스트를 위한 설정 수정"""
    config = config.copy()
    
    # 데이터 크기 축소
    config['data_limits'] = {
        'max_train_samples': 100,
        'max_valid_samples': 30,
        'max_test_samples': 10
    }
    
    # CNN 설정 간소화
    config['cnn'].update({
        'epochs': 2,
        'batch_size': 4
    })
    
    # Deep Forest 설정 간소화
    config['deep_forest'].update({
        'n_layers': 1,
        'n_estimators': 20,
        'max_depth': 5
    })
    
    # ACO 설정 간소화
    config['aco_hybrid'].update({
        'n_ants': 10,
        'n_iterations': 20,
        'max_steps': 200
    })
    
    # 평가 미로 수 축소
    config['evaluation']['test_mazes'] = 5
    
    return config


def main():
    """메인 함수"""
    args = parse_arguments()
    
    # 로깅 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # GPU 상태만 확인
    if args.gpu_check:
        profiler = get_profiler()
        gpu_status = profiler.check_rtx3060_limits()
        
        print("GPU 상태 확인")
        print("="*50)
        print(f"VRAM 사용률: {gpu_status['vram_utilization_percent']:.1f}%")
        print(f"현재 VRAM: {gpu_status['current_metrics']['vram_used_mb']:.1f}MB")
        print(f"총 VRAM: {gpu_status['current_metrics']['vram_total_mb']:.1f}MB")
        print(f"GPU 온도: {gpu_status['current_metrics']['temperature_c']:.1f}°C")
        print(f"전력 소비: {gpu_status['current_metrics']['power_watts']:.1f}W")
        
        if gpu_status['warnings']:
            print("\n경고 사항:")
            for warning in gpu_status['warnings']:
                print(f"  - {warning}")
        else:
            print("\n모든 지표가 정상 범위입니다.")
        
        return
    
    # 설정 파일 로드
    if not os.path.exists(args.config):
        logger.error(f"설정 파일을 찾을 수 없습니다: {args.config}")
        logger.info("기본 설정 파일을 생성하시겠습니까? (configs/ml_training.json)")
        
        # 기본 설정 생성
        os.makedirs('configs', exist_ok=True)
        
        # 설정 템플릿을 여기에 작성하거나 별도 파일에서 복사
        logger.error("먼저 설정 파일을 생성하세요.")
        sys.exit(1)
    
    try:
        # 벤치마크 시스템 초기화
        benchmark = MazeBenchmark(args.config)
        
        # 명령행 인수로 설정 오버라이드
        if args.output_dir:
            benchmark.config['output_dir'] = args.output_dir
        
        if args.models:
            benchmark.config['models'] = args.models
        
        if args.test_mazes:
            benchmark.config['evaluation']['test_mazes'] = args.test_mazes
        
        if args.force_retrain:
            benchmark.config['force_retrain'] = True
        
        # 빠른 테스트 모드
        if args.quick_test:
            logger.info("빠른 테스트 모드 활성화")
            benchmark.config = modify_config_for_quick_test(benchmark.config)
        
        # 실행 모드에 따른 처리
        if args.ml_only:
            logger.info("ML 모델 학습만 실행")
            results = benchmark.train_ml_models()
            print(f"\nML 모델 학습 완료!")
            
        elif args.eval_only:
            logger.info("평가만 실행 (기존 모델 사용)")
            
            # 기존 모델 경로 설정
            ml_results = {
                'cnn': {'model_path': str(benchmark.models_dir / 'best_cnn_model.pth')},
                'deep_forest': {'model_path': str(benchmark.models_dir / 'deep_forest_model.joblib')}
            }
            
            # 모델 파일 존재 확인
            missing_models = []
            for model_name, result in ml_results.items():
                if not os.path.exists(result['model_path']):
                    missing_models.append(model_name)
            
            if missing_models:
                logger.error(f"다음 모델 파일을 찾을 수 없습니다: {missing_models}")
                logger.info("먼저 --ml-only로 모델을 학습하세요.")
                sys.exit(1)
            
            hybrid_results = benchmark.run_hybrid_evaluation(ml_results)
            try:
                benchmark.generate_final_report(ml_results, hybrid_results)
            except Exception as e:
                logger.error(f"리포트 생성 실패: {e}")
            print(f"\n평가 완료!")
            
        else:
            # 전체 벤치마크 실행
            logger.info("전체 벤치마크 실행")
            results = benchmark.run_full_benchmark()
            
            if results:
                print(f"\n벤치마크 완료!")
                print(f"결과 디렉터리: {benchmark.results_dir}")
            else:
                print(f"\n벤치마크 실패 또는 중단됨")
                sys.exit(1)
        
    except KeyboardInterrupt:
        print(f"\n⏹️ 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 실행 중 오류 발생: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()