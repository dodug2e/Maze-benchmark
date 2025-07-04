"""
DQN 벤치마크 시스템 통합 래퍼
기존 벤치마크 인터페이스와 호환되도록 DQN을 래핑
RTX 3060 최적화 및 자동화된 성능 측정 포함
"""

import time
import numpy as np
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
import json

from .dqn_solver import DQNSolver
from utils.profiler import get_profiler, profile_execution, measure_vram_usage
from utils.maze_io import load_maze_as_array

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """벤치마크 결과 데이터 클래스 (기존 시스템 호환)"""
    algorithm: str
    maze_id: str
    maze_size: Tuple[int, int]
    execution_time: float
    power_consumption: float
    vram_usage: float
    gpu_utilization: float
    cpu_utilization: float
    solution_found: bool
    solution_length: int
    total_steps: int
    max_steps: int
    failure_reason: Optional[str] = None
    additional_metrics: Optional[Dict] = None

class DQNBenchmarkWrapper:
    """DQN 벤치마크 래퍼 클래스"""
    
    def __init__(self, 
                 training_episodes: int = 2000,
                 max_steps: int = 1000,
                 learning_rate: float = 1e-3,
                 batch_size: int = 16,
                 memory_size: int = 8000,
                 model_save_dir: str = "models/dqn",
                 **kwargs):
        
        self.training_episodes = training_episodes
        self.max_steps = max_steps
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # DQN 솔버 설정
        self.dqn_config = {
            'episodes': training_episodes,
            'max_steps': max_steps,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'memory_size': memory_size,
            **kwargs
        }
        
        # 현재 활성 솔버
        self.solver = None
        
        # 학습된 모델 캐시 (메모리 효율성)
        self._trained_models = {}
        
        logger.info(f"DQN 벤치마크 래퍼 초기화 완료: {self.dqn_config}")
    
    def run_benchmark(self, 
                     maze_id: str, 
                     subset: str = "test",
                     force_retrain: bool = False) -> BenchmarkResult:
        """
        벤치마크 실행 (기존 시스템 호환 인터페이스)
        
        Args:
            maze_id: 미로 ID (예: "000001")
            subset: 데이터셋 분할 ("train", "valid", "test")
            force_retrain: 강제 재학습 여부
            
        Returns:
            BenchmarkResult: 벤치마크 결과
        """
        logger.info(f"DQN 벤치마크 시작: 미로 {maze_id}")
        
        # 성능 프로파일러 시작
        profiler = get_profiler()
        
        try:
            with profile_execution(f"DQN 벤치마크 - 미로 {maze_id}"):
                # 미로 데이터 로드
                maze_array, metadata = load_maze_as_array(maze_id, subset)
                
                # 시작점과 목표점 추출
                start = tuple(metadata.get('entrance', (0, 0)))
                goal = tuple(metadata.get('exit', (maze_array.shape[0]-1, maze_array.shape[1]-1)))
                
                logger.info(f"미로 정보: 크기={maze_array.shape}, 시작={start}, 목표={goal}")
                
                # 모델 캐시 키 생성
                model_key = f"{maze_id}_{maze_array.shape[0]}x{maze_array.shape[1]}"
                model_path = self.model_save_dir / f"dqn_{model_key}.pth"
                
                # 전체 실행 시간 측정 시작
                total_start_time = time.time()
                
                # 학습 또는 모델 로드
                training_stats = self._handle_training(
                    maze_array, start, goal, model_key, model_path, force_retrain
                )
                
                # 미로 해결 실행
                logger.info("DQN 미로 해결 시작...")
                solution_start = time.time()
                path, solve_result = self.solver.solve(maze_array, start, goal)
                execution_time = time.time() - solution_start
                total_time = time.time() - total_start_time
                
                # 성능 메트릭 수집
                final_metrics = profiler.get_current_metrics()
                summary_stats = profiler.get_summary_stats()
                
                # 결과 구성
                result = BenchmarkResult(
                    algorithm="DQN",
                    maze_id=maze_id,
                    maze_size=maze_array.shape,
                    execution_time=execution_time,
                    power_consumption=final_metrics.power_watts,
                    vram_usage=final_metrics.vram_used_mb,
                    gpu_utilization=final_metrics.gpu_percent,
                    cpu_utilization=final_metrics.cpu_percent,
                    solution_found=solve_result['success'],
                    solution_length=len(path) - 1 if solve_result['success'] else 0,
                    total_steps=solve_result['steps'],
                    max_steps=self.max_steps,
                    failure_reason=solve_result.get('reason') if not solve_result['success'] else None,
                    additional_metrics={
                        'total_reward': solve_result['reward'],
                        'training_episodes': self.training_episodes,
                        'training_stats': training_stats,
                        'model_path': str(model_path),
                        'total_execution_time': total_time,
                        'vram_peak': summary_stats.get('vram_used_mb', {}).get('peak', 0),
                        'avg_gpu_util': summary_stats.get('gpu_percent', {}).get('avg', 0),
                        'memory_efficiency': self._calculate_memory_efficiency(final_metrics)
                    }
                )
                
                logger.info(f"DQN 벤치마크 완료: {'성공' if result.solution_found else '실패'}")
                return result
                
        except Exception as e:
            logger.error(f"DQN 벤치마크 실행 중 오류: {e}")
            # 오류 시에도 기본 결과 반환
            final_metrics = profiler.get_current_metrics()
            
            return BenchmarkResult(
                algorithm="DQN",
                maze_id=maze_id,
                maze_size=(0, 0),
                execution_time=0.0,
                power_consumption=final_metrics.power_watts,
                vram_usage=final_metrics.vram_used_mb,
                gpu_utilization=final_metrics.gpu_percent,
                cpu_utilization=final_metrics.cpu_percent,
                solution_found=False,
                solution_length=0,
                total_steps=0,
                max_steps=self.max_steps,
                failure_reason=f"실행 오류: {str(e)}",
                additional_metrics={'error': str(e)}
            )
    
    def _handle_training(self, maze_array, start, goal, model_key, model_path, force_retrain):
        """학습 처리 (캐싱 포함)"""
        
        # 강제 재학습이거나 캐시된 모델이 없는 경우
        if force_retrain or model_key not in self._trained_models or not model_path.exists():
            logger.info("DQN 모델 학습 시작...")
            
            # 미로 크기에 따른 동적 설정 조정
            adjusted_config = self._adjust_config_for_maze_size(maze_array.shape)
            
            # 새 솔버 생성
            self.solver = DQNSolver(**adjusted_config)
            
            # 학습 실행
            training_result = self.solver.train(maze_array, start, goal, str(model_path))
            
            # 학습 결과 캐싱
            training_stats = {
                'final_success_rate': training_result['final_success_rate'],
                'average_reward': training_result['average_reward'],
                'training_history': training_result['training_history'],
                'config_used': adjusted_config
            }
            self._trained_models[model_key] = training_stats
            
            logger.info(f"DQN 학습 완료: 성공률 {training_result['final_success_rate']:.2%}")
            
        else:
            logger.info("기존 학습된 DQN 모델 로드...")
            
            # 기존 모델 로드
            if self.solver is None:
                adjusted_config = self._adjust_config_for_maze_size(maze_array.shape)
                self.solver = DQNSolver(**adjusted_config)
            
            self.solver.load_model(str(model_path))
            training_stats = self._trained_models[model_key]
        
        return training_stats
    
    def _adjust_config_for_maze_size(self, maze_shape):
        """미로 크기에 따른 DQN 설정 자동 조정"""
        config = self.dqn_config.copy()
        
        max_dim = max(maze_shape)
        
        if max_dim <= 50:
            # 작은 미로: 더 큰 배치 크기 사용 가능
            config.update({
                'batch_size': min(32, config['batch_size'] * 2),
                'memory_size': min(12000, config['memory_size'] * 1.5),
                'target_update': 500
            })
        elif max_dim <= 100:
            # 중간 미로: 기본 설정 유지
            pass
        else:
            # 큰 미로: 메모리 절약 설정
            config.update({
                'batch_size': max(8, config['batch_size'] // 2),
                'memory_size': max(4000, config['memory_size'] // 2),
                'target_update': 2000
            })
        
        logger.info(f"미로 크기 {maze_shape}에 대한 조정된 DQN 설정: "
                   f"batch_size={config['batch_size']}, memory_size={config['memory_size']}")
        
        return config
    
    def _calculate_memory_efficiency(self, metrics):
        """메모리 효율성 계산"""
        max_vram_mb = 6144  # RTX 3060 6GB
        vram_efficiency = 1.0 - (metrics.vram_used_mb / max_vram_mb)
        
        return {
            'vram_efficiency': vram_efficiency,
            'vram_utilization': metrics.vram_used_mb / max_vram_mb,
            'memory_pressure': 'high' if metrics.vram_used_mb > max_vram_mb * 0.9 else 'normal'
        }
    
    def batch_benchmark(self, 
                       maze_ids: List[str], 
                       subset: str = "test",
                       force_retrain: bool = False) -> List[BenchmarkResult]:
        """배치 벤치마크 실행"""
        results = []
        
        logger.info(f"DQN 배치 벤치마크 시작: {len(maze_ids)}개 미로")
        
        for i, maze_id in enumerate(maze_ids):
            logger.info(f"배치 벤치마크 진행: {i+1}/{len(maze_ids)} - 미로 {maze_id}")
            
            try:
                result = self.run_benchmark(maze_id, subset, force_retrain)
                results.append(result)
                
                # VRAM 정리 (RTX 3060 최적화)
                self._cleanup_memory()
                
                # 중간 결과 로깅
                if result.solution_found:
                    logger.info(f"  ✅ 성공: {result.solution_length}스텝, "
                              f"{result.execution_time:.2f}초, VRAM {result.vram_usage:.0f}MB")
                else:
                    logger.warning(f"  ❌ 실패: {result.failure_reason}")
                        
            except Exception as e:
                logger.error(f"미로 {maze_id} 벤치마크 실행 실패: {e}")
                continue
        
        # 배치 요약 통계
        successful = [r for r in results if r.solution_found]
        logger.info(f"DQN 배치 벤치마크 완료: {len(successful)}/{len(results)} 성공 "
                   f"({len(successful)/len(results)*100:.1f}%)")
        
        return results
    
    def _cleanup_memory(self):
        """메모리 정리 (RTX 3060 최적화)"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # 에이전트 메모리 정리
            if hasattr(self.solver, 'agent') and self.solver.agent:
                if hasattr(self.solver.agent, 'memory'):
                    # DQN의 경험 재생 버퍼 크기 축소
                    current_size = len(self.solver.agent.memory)
                    if current_size > self.solver.agent.memory.maxlen // 2:
                        # 절반만 유지
                        keep_size = self.solver.agent.memory.maxlen // 4
                        new_buffer = list(self.solver.agent.memory)[-keep_size:]
                        self.solver.agent.memory.clear()
                        self.solver.agent.memory.extend(new_buffer)
                        logger.debug(f"DQN 메모리 버퍼 정리: {current_size} → {len(new_buffer)}")
                        
        except Exception as e:
            logger.warning(f"메모리 정리 중 오류: {e}")
    
    def get_training_stats(self) -> Dict:
        """학습 통계 반환"""
        if not self._trained_models:
            return {"error": "학습된 모델이 없습니다."}
        
        # 가장 최근 학습 모델의 통계 반환
        latest_model = list(self._trained_models.values())[-1]
        
        stats = {
            'models_trained': len(self._trained_models),
            'latest_success_rate': latest_model['final_success_rate'],
            'latest_average_reward': latest_model['average_reward'],
            'training_episodes': self.training_episodes
        }
        
        # 전체 모델 통계
        all_success_rates = [model['final_success_rate'] for model in self._trained_models.values()]
        if all_success_rates:
            stats.update({
                'avg_success_rate': np.mean(all_success_rates),
                'best_success_rate': np.max(all_success_rates),
                'worst_success_rate': np.min(all_success_rates),
                'success_rate_std': np.std(all_success_rates)
            })
        
        return stats
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        info = {
            'algorithm': 'DQN',
            'config': self.dqn_config,
            'trained_models': list(self._trained_models.keys()),
            'model_save_dir': str(self.model_save_dir)
        }
        
        if self.solver and hasattr(self.solver, 'agent'):
            info.update({
                'network_architecture': str(self.solver.agent.q_network) if self.solver.agent else None,
                'device': str(self.solver.agent.device) if self.solver.agent else None
            })
        
        return info
    
    def export_results(self, results: List[BenchmarkResult], output_path: str):
        """결과를 JSON으로 내보내기"""
        export_data = {
            'algorithm': 'DQN',
            'benchmark_config': self.dqn_config,
            'total_experiments': len(results),
            'successful_experiments': len([r for r in results if r.solution_found]),
            'results': []
        }
        
        for result in results:
            result_dict = {
                'maze_id': result.maze_id,
                'maze_size': result.maze_size,
                'success': result.solution_found,
                'execution_time': result.execution_time,
                'solution_length': result.solution_length,
                'vram_usage': result.vram_usage,
                'gpu_utilization': result.gpu_utilization,
                'failure_reason': result.failure_reason,
                'additional_metrics': result.additional_metrics
            }
            export_data['results'].append(result_dict)
        
        # 요약 통계 추가
        successful_results = [r for r in results if r.solution_found]
        if successful_results:
            export_data['summary_statistics'] = {
                'success_rate': len(successful_results) / len(results),
                'avg_execution_time': np.mean([r.execution_time for r in successful_results]),
                'avg_solution_length': np.mean([r.solution_length for r in successful_results]),
                'avg_vram_usage': np.mean([r.vram_usage for r in results]),
                'avg_gpu_utilization': np.mean([r.gpu_utilization for r in results])
            }
        
        # 파일 저장
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"DQN 벤치마크 결과 내보내기 완료: {output_path}")
    
    def cleanup(self):
        """리소스 정리"""
        logger.info("DQN 벤치마크 래퍼 리소스 정리 시작...")
        
        # 메모리 정리
        self._cleanup_memory()
        
        # 솔버 정리
        if self.solver:
            del self.solver
            self.solver = None
        
        # 캐시 정리
        self._trained_models.clear()
        
        logger.info("DQN 벤치마크 래퍼 리소스 정리 완료")


# 기존 시스템과의 호환성을 위한 팩토리 함수
def create_dqn_benchmark(**kwargs) -> DQNBenchmarkWrapper:
    """DQN 벤치마크 래퍼 생성 팩토리 함수"""
    return DQNBenchmarkWrapper(**kwargs)


# RTX 3060 최적화 프리셋
def create_rtx3060_optimized_dqn(**kwargs) -> DQNBenchmarkWrapper:
    """RTX 3060 최적화 DQN 벤치마크 생성"""
    rtx3060_config = {
        'training_episodes': 2000,
        'max_steps': 1000,
        'learning_rate': 1e-3,
        'batch_size': 16,
        'memory_size': 8000,
        'target_update': 1000,
        'device': 'auto'
    }
    rtx3060_config.update(kwargs)
    return DQNBenchmarkWrapper(**rtx3060_config)


if __name__ == "__main__":
    # 테스트 코드
    print("DQN 벤치마크 래퍼 테스트...")
    
    # RTX 3060 최적화 설정으로 래퍼 생성
    wrapper = create_rtx3060_optimized_dqn(
        training_episodes=100,  # 테스트용 단축
        max_steps=200
    )
    
    try:
        # 모델 정보 출력
        model_info = wrapper.get_model_info()
        print(f"DQN 설정: {model_info['config']}")
        
        # 메모리 효율성 테스트
        from utils.profiler import get_profiler
        profiler = get_profiler()
        current_metrics = profiler.get_current_metrics()
        
        efficiency = wrapper._calculate_memory_efficiency(current_metrics)
        print(f"메모리 효율성: {efficiency}")
        
        print("✅ DQN 벤치마크 래퍼 초기화 성공!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
    finally:
        wrapper.cleanup()