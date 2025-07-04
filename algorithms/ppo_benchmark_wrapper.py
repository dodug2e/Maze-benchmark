"""
PPO 벤치마크 시스템 통합 래퍼
기존 벤치마크 인터페이스와 호환되도록 PPO를 래핑
"""

import time
import numpy as np
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import logging
from dataclasses import dataclass

from .ppo_solver import PPOSolver
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

class PPOBenchmarkWrapper:
    """PPO 벤치마크 래퍼 클래스"""
    
    def __init__(self, 
                 total_timesteps: int = 1_000_000,
                 max_episode_steps: int = 1000,
                 learning_rate: float = 3e-4,
                 model_save_dir: str = "models/ppo",
                 **kwargs):
        
        self.total_timesteps = total_timesteps
        self.max_episode_steps = max_episode_steps
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # PPO 솔버 초기화
        self.solver = PPOSolver(
            total_timesteps=total_timesteps,
            max_episode_steps=max_episode_steps,
            learning_rate=learning_rate,
            **kwargs
        )
        
        # 학습된 모델 캐시
        self._trained_models = {}
        
        logger.info(f"PPO 벤치마크 래퍼 초기화 완료")
    
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
        logger.info(f"PPO 벤치마크 시작: 미로 {maze_id}")
        
        # 성능 프로파일러 시작
        profiler = get_profiler()
        
        try:
            with profile_execution(f"PPO 벤치마크 - 미로 {maze_id}"):
                # 미로 데이터 로드
                maze_array, metadata = load_maze_as_array(maze_id, subset)
                
                # 시작점과 목표점 추출
                start = tuple(metadata.get('entrance', (0, 0)))
                goal = tuple(metadata.get('exit', (maze_array.shape[0]-1, maze_array.shape[1]-1)))
                
                logger.info(f"미로 크기: {maze_array.shape}, 시작: {start}, 목표: {goal}")
                
                # 모델 학습 또는 로드
                model_key = f"{maze_id}_{maze_array.shape[0]}x{maze_array.shape[1]}"
                model_path = self.model_save_dir / f"ppo_{model_key}.pth"
                
                start_time = time.time()
                
                if force_retrain or model_key not in self._trained_models or not model_path.exists():
                    logger.info("PPO 모델 학습 시작...")
                    training_result = self.solver.train(maze_array, start, goal, str(model_path))
                    self._trained_models[model_key] = training_result
                    logger.info(f"학습 완료: 성공률 {training_result['final_success_rate']:.2f}")
                else:
                    logger.info("기존 학습된 모델 로드...")
                    self.solver.load_model(str(model_path))
                
                # 미로 해결 실행
                logger.info("미로 해결 시작...")
                solution_start = time.time()
                path, solve_result = self.solver.solve(maze_array, start, goal)
                execution_time = time.time() - solution_start
                
                # 성능 메트릭 수집
                final_metrics = profiler.get_current_metrics()
                summary_stats = profiler.get_summary_stats()
                
                # 학습 통계 가져오기
                training_stats = self._get_training_stats()
                
                # 결과 구성
                result = BenchmarkResult(
                    algorithm="PPO",
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
                    max_steps=self.max_episode_steps,
                    failure_reason=solve_result.get('reason') if not solve_result['success'] else None,
                    additional_metrics={
                        'total_reward': solve_result['reward'],
                        'total_timesteps': self.total_timesteps,
                        'model_path': str(model_path),
                        'vram_peak': summary_stats.get('vram_used_mb', {}).get('peak', 0),
                        'avg_gpu_util': summary_stats.get('gpu_percent', {}).get('avg', 0),
                        'training_stats': training_stats
                    }
                )
                
                logger.info(f"PPO 벤치마크 완료: {'성공' if result.solution_found else '실패'}")
                return result
                
        except Exception as e:
            logger.error(f"PPO 벤치마크 실행 중 오류: {e}")
            # 오류 시에도 기본 결과 반환
            final_metrics = profiler.get_current_metrics()
            
            return BenchmarkResult(
                algorithm="PPO",
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
                max_steps=self.max_episode_steps,
                failure_reason=f"실행 오류: {str(e)}",
                additional_metrics={'error': str(e)}
            )
    
    def batch_benchmark(self, 
                       maze_ids: List[str], 
                       subset: str = "test",
                       force_retrain: bool = False) -> List[BenchmarkResult]:
        """배치 벤치마크 실행"""
        results = []
        
        for i, maze_id in enumerate(maze_ids):
            logger.info(f"배치 벤치마크 진행: {i+1}/{len(maze_ids)} - 미로 {maze_id}")
            
            try:
                result = self.run_benchmark(maze_id, subset, force_retrain)
                results.append(result)
                
                # VRAM 정리 (RTX 3060 최적화)
                if hasattr(self.solver, 'agent') and self.solver.agent:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            except Exception as e:
                logger.error(f"미로 {maze_id} 벤치마크 실행 실패: {e}")
                continue
        
        return results
    
    def _get_training_stats(self) -> Dict:
        """학습 통계 반환"""
        if not hasattr(self.solver, 'training_history'):
            return {"error": "학습 히스토리가 없습니다."}
        
        history = self.solver.training_history
        
        stats = {
            'total_episodes': len(history['episode_rewards']) if history['episode_rewards'] else 0,
            'final_success_rate': history['success_rate'][-1] if history['success_rate'] else 0.0,
            'average_reward': np.mean(history['episode_rewards'][-100:]) if history['episode_rewards'] else 0.0,
            'final_episode_length': history['episode_lengths'][-1] if history['episode_lengths'] else 0,
        }
        
        # PPO 특화 통계
        if history['policy_losses']:
            stats.update({
                'final_policy_loss': history['policy_losses'][-1],
                'final_value_loss': history['value_losses'][-1] if history['value_losses'] else 0.0,
                'policy_loss_trend': self._compute_trend(history['policy_losses']),
                'value_loss_trend': self._compute_trend(history['value_losses']) if history['value_losses'] else 0.0
            })
        
        return stats
    
    def _compute_trend(self, values: List[float], window: int = 50) -> str:
        """손실 값의 트렌드 계산"""
        if len(values) < window * 2:
            return "insufficient_data"
        
        recent = np.mean(values[-window:])
        earlier = np.mean(values[-window*2:-window])
        
        if recent < earlier * 0.9:
            return "decreasing"
        elif recent > earlier * 1.1:
            return "increasing"
        else:
            return "stable"
    
    def get_learning_curve_data(self) -> Dict:
        """학습 곡선 데이터 반환"""
        if not hasattr(self.solver, 'training_history'):
            return {}
        
        history = self.solver.training_history
        
        return {
            'episodes': list(range(len(history['episode_rewards']))),
            'rewards': history['episode_rewards'],
            'success_rates': history['success_rate'],
            'episode_lengths': history['episode_lengths'],
            'policy_losses': history['policy_losses'],
            'value_losses': history['value_losses']
        }
    
    def cleanup(self):
        """리소스 정리"""
        if hasattr(self.solver, 'agent') and self.solver.agent:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        logger.info("PPO 벤치마크 래퍼 리소스 정리 완료")


# 기존 시스템과의 호환성을 위한 팩토리 함수
def create_ppo_benchmark(**kwargs) -> PPOBenchmarkWrapper:
    """PPO 벤치마크 래퍼 생성"""
    return PPOBenchmarkWrapper(**kwargs)


if __name__ == "__main__":
    # 테스트 코드
    print("PPO 벤치마크 래퍼 테스트...")
    
    # RTX 3060 최적화 설정으로 래퍼 생성
    wrapper = create_ppo_benchmark(
        total_timesteps=50000,  # 테스트용 단축
        max_episode_steps=200,
        learning_rate=3e-4,
        buffer_size=512,
        batch_size=32
    )
    
    try:
        # 첫 번째 미로로 테스트 (실제 데이터셋 필요)
        # result = wrapper.run_benchmark("000001", "test")
        # print(f"테스트 결과: {result.solution_found}")
        print("래퍼 초기화 성공!")
        
    except Exception as e:
        print(f"테스트 실패: {e}")
    finally:
        wrapper.cleanup()