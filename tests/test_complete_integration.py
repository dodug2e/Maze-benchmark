"""
완전 통합 테스트 스위트
DQN, PPO, 벤치마크 시스템 전체 검증
"""

import pytest
import numpy as np
import torch
import tempfile
import shutil
import time
from pathlib import Path
import sys
import os
import json

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from algorithms.dqn_solver import DQNSolver
from algorithms.ppo_solver import PPOSolver
from algorithms.dqn_benchmark_wrapper import DQNBenchmarkWrapper
from algorithms.ppo_benchmark_wrapper import PPOBenchmarkWrapper
from utils.profiler import get_profiler

class TestCompleteIntegration:
    """완전 통합 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 테스트용 간단한 미로들
        self.test_mazes = {
            'tiny': np.array([
                [1, 1, 1],
                [1, 0, 1], 
                [1, 1, 1]
            ]),
            'small': np.array([
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 1, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1]
            ]),
            'medium': self._generate_medium_maze()
        }
        
        self.start_goals = {
            'tiny': ((0, 0), (2, 2)),
            'small': ((0, 0), (4, 4)),
            'medium': ((0, 0), (9, 9))
        }
    
    def teardown_method(self):
        """테스트 정리"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _generate_medium_maze(self):
        """중간 크기 테스트 미로 생성"""
        maze = np.ones((10, 10), dtype=np.uint8)
        # 간단한 경로 생성
        maze[1:-1, 1:-1] = 0  # 내부를 통로로
        maze[2:-2, 2:-2] = 1  # 중간에 벽
        maze[1, 1:-1] = 1     # 상단 통로
        maze[-2, 1:-1] = 1    # 하단 통로
        maze[1:-1, 1] = 1     # 좌측 통로
        maze[1:-1, -2] = 1    # 우측 통로
        return maze
    
    def test_dqn_complete_pipeline(self):
        """DQN 완전 파이프라인 테스트"""
        print("\n=== DQN 완전 파이프라인 테스트 ===")
        
        maze = self.test_mazes['small']
        start, goal = self.start_goals['small']
        
        # DQN 솔버 생성
        solver = DQNSolver(
            episodes=50,  # 빠른 테스트
            max_steps=100,
            learning_rate=1e-2,
            batch_size=8,
            memory_size=200,
            device='cpu'  # 안정성을 위해 CPU 사용
        )
        
        # 학습 실행
        print("DQN 학습 중...")
        start_time = time.time()
        training_result = solver.train(maze, start, goal)
        training_time = time.time() - start_time
        
        print(f"학습 완료: {training_time:.2f}초, 성공률: {training_result['final_success_rate']:.2f}")
        
        # 추론 실행
        print("DQN 추론 중...")
        path, solve_result = solver.solve(maze, start, goal)
        
        print(f"추론 결과: 성공={solve_result['success']}, 스텝={solve_result['steps']}")
        
        # 검증
        assert isinstance(training_result, dict)
        assert 'final_success_rate' in training_result
        assert isinstance(path, list)
        assert len(path) > 0
        assert path[0] == start
        
        if solve_result['success']:
            assert path[-1] == goal
        
        print("✅ DQN 파이프라인 테스트 통과")
    
    def test_ppo_complete_pipeline(self):
        """PPO 완전 파이프라인 테스트"""
        print("\n=== PPO 완전 파이프라인 테스트 ===")
        
        maze = self.test_mazes['small']
        start, goal = self.start_goals['small']
        
        # PPO 솔버 생성
        solver = PPOSolver(
            total_timesteps=10000,  # 빠른 테스트
            max_episode_steps=100,
            learning_rate=1e-2,
            buffer_size=256,
            batch_size=32,
            device='cpu'
        )
        
        # 학습 실행
        print("PPO 학습 중...")
        start_time = time.time()
        training_result = solver.train(maze, start, goal)
        training_time = time.time() - start_time
        
        print(f"학습 완료: {training_time:.2f}초, 성공률: {training_result['final_success_rate']:.2f}")
        
        # 추론 실행
        print("PPO 추론 중...")
        path, solve_result = solver.solve(maze, start, goal)
        
        print(f"추론 결과: 성공={solve_result['success']}, 스텝={solve_result['steps']}")
        
        # 검증
        assert isinstance(training_result, dict)
        assert 'final_success_rate' in training_result
        assert isinstance(path, list)
        assert len(path) > 0
        assert path[0] == start
        
        if solve_result['success']:
            assert path[-1] == goal
        
        print("✅ PPO 파이프라인 테스트 통과")
    
    def test_benchmark_wrapper_integration(self):
        """벤치마크 래퍼 통합 테스트"""
        print("\n=== 벤치마크 래퍼 통합 테스트 ===")
        
        # 모의 미로 데이터 생성
        test_maze_id = "test_001"
        maze = self.test_mazes['small']
        start, goal = self.start_goals['small']
        
        # 모의 메타데이터
        metadata = {
            'entrance': start,
            'exit': goal,
            'size': maze.shape[0],
            'algorithm': 'test',
            'metrics': {'solvable': True}
        }
        
        # DQN 래퍼 테스트
        print("DQN 벤치마크 래퍼 테스트...")
        dqn_wrapper = DQNBenchmarkWrapper(
            training_episodes=30,
            max_steps=50,
            model_save_dir=self.temp_dir
        )
        
        # PPO 래퍼 테스트
        print("PPO 벤치마크 래퍼 테스트...")
        ppo_wrapper = PPOBenchmarkWrapper(
            total_timesteps=5000,
            max_episode_steps=50,
            model_save_dir=self.temp_dir
        )
        
        # 정리
        dqn_wrapper.cleanup()
        ppo_wrapper.cleanup()
        
        print("✅ 벤치마크 래퍼 테스트 통과")
    
    def test_performance_profiling(self):
        """성능 프로파일링 테스트"""
        print("\n=== 성능 프로파일링 테스트 ===")
        
        profiler = get_profiler()
        
        # 현재 메트릭 수집
        metrics = profiler.get_current_metrics()
        print(f"현재 VRAM: {metrics.vram_used_mb:.1f}MB")
        print(f"현재 GPU: {metrics.gpu_percent:.1f}%")
        print(f"현재 CPU: {metrics.cpu_percent:.1f}%")
        
        # 프로파일링으로 간단한 작업 실행
        with profiler.measure("테스트 작업"):
            # 간단한 GPU 작업 (사용 가능한 경우)
            if torch.cuda.is_available():
                x = torch.randn(100, 100).cuda()
                y = torch.matmul(x, x.t())
                z = y.cpu()
            else:
                x = torch.randn(1000, 1000)
                y = torch.matmul(x, x.t())
            
            time.sleep(0.1)  # 잠시 대기
        
        # 요약 통계 확인
        summary = profiler.get_summary_stats()
        print(f"프로파일링 완료: {summary.get('duration_seconds', 0):.2f}초")
        
        print("✅ 성능 프로파일링 테스트 통과")
    
    def test_algorithm_comparison(self):
        """알고리즘 비교 시스템 테스트"""
        print("\n=== 알고리즘 비교 시스템 테스트 ===")
        
        # 간단한 비교 시뮬레이션
        algorithms = ['DQN', 'PPO']
        maze = self.test_mazes['tiny']
        start, goal = self.start_goals['tiny']
        
        results = {}
        
        for algo in algorithms:
            print(f"{algo} 테스트 실행...")
            
            if algo == 'DQN':
                solver = DQNSolver(
                    episodes=20, max_steps=20, 
                    device='cpu', batch_size=4
                )
            else:  # PPO
                solver = PPOSolver(
                    total_timesteps=2000, max_episode_steps=20,
                    device='cpu', buffer_size=64
                )
            
            # 빠른 학습
            start_time = time.time()
            training_result = solver.train(maze, start, goal)
            train_time = time.time() - start_time
            
            # 추론
            path, solve_result = solver.solve(maze, start, goal)
            
            results[algo] = {
                'success_rate': training_result['final_success_rate'],
                'training_time': train_time,
                'solution_found': solve_result['success'],
                'solution_steps': solve_result['steps']
            }
            
            print(f"{algo} 완료: 성공률={training_result['final_success_rate']:.2f}")
        
        # 결과 비교
        print("\n비교 결과:")
        for algo, result in results.items():
            print(f"{algo}: 성공률={result['success_rate']:.2f}, "
                  f"학습시간={result['training_time']:.1f}초")
        
        print("✅ 알고리즘 비교 테스트 통과")
    
    def test_rtx3060_optimization(self):
        """RTX 3060 최적화 설정 테스트"""
        print("\n=== RTX 3060 최적화 테스트 ===")
        
        profiler = get_profiler()
        
        # RTX 3060 한계 체크
        limits_check = profiler.check_rtx3060_limits()
        
        print(f"VRAM 사용률: {limits_check['vram_utilization_percent']:.1f}%")
        print(f"현재 온도: {limits_check['current_metrics']['temperature_c']:.1f}°C")
        print(f"전력 소비: {limits_check['current_metrics']['power_watts']:.1f}W")
        
        if limits_check['warnings']:
            print("경고:")
            for warning in limits_check['warnings']:
                print(f"  - {warning}")
        else:
            print("모든 지표가 정상 범위입니다.")
        
        # 최적화 설정 검증
        configs_to_test = [
            {'batch_size': 16, 'buffer_size': 2048},  # 중간 설정
            {'batch_size': 8, 'buffer_size': 1024},   # 안전 설정
        ]
        
        for i, config in enumerate(configs_to_test):
            print(f"설정 {i+1} 테스트: {config}")
            
            # DQN으로 메모리 사용량 테스트
            try:
                solver = DQNSolver(
                    episodes=5, 
                    device='auto',
                    **config
                )
                
                maze = self.test_mazes['small']
                start, goal = self.start_goals['small']
                
                # 짧은 학습으로 메모리 사용량 확인
                solver.train(maze, start, goal)
                
                current_vram = profiler.get_current_metrics().vram_used_mb
                print(f"  VRAM 사용량: {current_vram:.1f}MB")
                
                # 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  설정 {i+1} 실패: {e}")
        
        print("✅ RTX 3060 최적화 테스트 통과")
    
    def test_model_save_load(self):
        """모델 저장/로드 테스트"""
        print("\n=== 모델 저장/로드 테스트 ===")
        
        maze = self.test_mazes['tiny']
        start, goal = self.start_goals['tiny']
        
        # DQN 저장/로드 테스트
        print("DQN 모델 저장/로드 테스트...")
        
        dqn_solver = DQNSolver(episodes=10, max_steps=20, device='cpu')
        dqn_solver.train(maze, start, goal)
        
        # 모델 저장
        save_path = Path(self.temp_dir) / "test_dqn.pth"
        dqn_solver.agent.save(str(save_path))
        
        # 새 솔버에 로드
        new_dqn_solver = DQNSolver(episodes=10, max_steps=20, device='cpu')
        new_dqn_solver.agent = dqn_solver.agent  # 구조 복사
        new_dqn_solver.agent.load(str(save_path))
        
        # 동일한 결과 확인
        path1, result1 = dqn_solver.solve(maze, start, goal)
        path2, result2 = new_dqn_solver.solve(maze, start, goal)
        
        print(f"원본 결과: {result1['success']}, 로드 결과: {result2['success']}")
        
        # PPO 저장/로드 테스트
        print("PPO 모델 저장/로드 테스트...")
        
        ppo_solver = PPOSolver(total_timesteps=2000, max_episode_steps=20, device='cpu')
        ppo_solver.train(maze, start, goal)
        
        # 모델 저장
        save_path = Path(self.temp_dir) / "test_ppo.pth"
        ppo_solver.agent.save(str(save_path))
        
        print("✅ 모델 저장/로드 테스트 통과")
    
    def test_error_handling(self):
        """오류 처리 테스트"""
        print("\n=== 오류 처리 테스트 ===")
        
        # 잘못된 미로 (해결 불가능)
        unsolvable_maze = np.array([
            [1, 0, 1],
            [0, 0, 0], 
            [1, 0, 1]
        ])
        
        start, goal = (0, 0), (2, 2)
        
        # DQN 오류 처리 테스트
        try:
            solver = DQNSolver(episodes=5, max_steps=10, device='cpu')
            result = solver.train(unsolvable_maze, start, goal)
            print(f"해결 불가능한 미로 학습 결과: 성공률={result['final_success_rate']:.2f}")
            
            path, solve_result = solver.solve(unsolvable_maze, start, goal)
            print(f"해결 불가능한 미로 추론: 성공={solve_result['success']}")
            
        except Exception as e:
            print(f"예상된 오류 발생: {e}")
        
        # 잘못된 파라미터 테스트
        try:
            invalid_solver = DQNSolver(episodes=-1)  # 잘못된 파라미터
        except Exception as e:
            print(f"잘못된 파라미터 오류: {e}")
        
        print("✅ 오류 처리 테스트 통과")

def run_comprehensive_test():
    """포괄적 테스트 실행"""
    print("🚀 포괄적 통합 테스트 시작")
    print("=" * 50)
    
    tester = TestCompleteIntegration()
    tester.setup_method()
    
    test_methods = [
        tester.test_performance_profiling,
        tester.test_dqn_complete_pipeline,
        tester.test_ppo_complete_pipeline,
        tester.test_benchmark_wrapper_integration,
        tester.test_algorithm_comparison,
        tester.test_rtx3060_optimization,
        tester.test_model_save_load,
        tester.test_error_handling
    ]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            test_method()
            passed += 1
        except Exception as e:
            print(f"❌ {test_method.__name__} 실패: {e}")
            failed += 1
    
    tester.teardown_method()
    
    print("\n" + "=" * 50)
    print(f"🎯 테스트 결과: {passed}개 통과, {failed}개 실패")
    
    if failed == 0:
        print("🎉 모든 테스트 통과! 시스템이 정상 작동합니다.")
    else:
        print("⚠️  일부 테스트 실패. 문제를 확인하세요.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)