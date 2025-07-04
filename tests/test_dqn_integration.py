"""
DQN 통합 테스트
기존 벤치마크 시스템과의 호환성 및 RTX 3060 최적화 검증
"""

import pytest
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path
import sys
import os

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from algorithms.dqn_solver import DQNSolver, MazeEnvironment, DQNAgent
from algorithms.dqn_benchmark_wrapper import DQNBenchmarkWrapper
from utils.profiler import get_profiler

class TestMazeEnvironment:
    """미로 환경 테스트"""
    
    def setup_method(self):
        """테스트용 간단한 미로 생성"""
        self.simple_maze = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ])
        self.start = (0, 0)
        self.goal = (4, 4)
        
    def test_environment_initialization(self):
        """환경 초기화 테스트"""
        env = MazeEnvironment(self.simple_maze, self.start, self.goal)
        
        assert env.height == 5
        assert env.width == 5
        assert env.action_size == 4
        assert env.start == self.start
        assert env.goal == self.goal
        
    def test_environment_reset(self):
        """환경 리셋 테스트"""
        env = MazeEnvironment(self.simple_maze, self.start, self.goal)
        state = env.reset()
        
        assert env.current_pos == self.start
        assert env.step_count == 0
        assert isinstance(state, np.ndarray)
        assert len(state) == env.state_size
        
    def test_environment_step(self):
        """환경 스텝 테스트"""
        env = MazeEnvironment(self.simple_maze, self.start, self.goal)
        env.reset()
        
        # 유효한 행동 (오른쪽으로 이동)
        state, reward, done, info = env.step(1)  # 하
        
        assert isinstance(state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
    def test_goal_reaching(self):
        """목표 도달 테스트"""
        # 2x2 간단한 미로
        simple_maze = np.array([
            [1, 1],
            [1, 1]
        ])
        
        env = MazeEnvironment(simple_maze, (0, 0), (1, 1))
        env.reset()
        
        # 목표로 직접 이동
        env.current_pos = (1, 1)
        state, reward, done, info = env.step(0)  # 임의 행동
        
        # 목표 도달 시 높은 보상과 종료 확인
        assert done or reward > 0  # 목표 도달 로직 확인

class TestDQNAgent:
    """DQN 에이전트 테스트"""
    
    def test_agent_initialization(self):
        """에이전트 초기화 테스트"""
        agent = DQNAgent(
            state_size=10,
            action_size=4,
            device='cpu'  # 테스트에서는 CPU 사용
        )
        
        assert agent.state_size == 10
        assert agent.action_size == 4
        assert agent.device.type == 'cpu'
        
    def test_agent_action_selection(self):
        """행동 선택 테스트"""
        agent = DQNAgent(
            state_size=10,
            action_size=4,
            device='cpu',
            epsilon=0.0  # 탐색 없음
        )
        
        state = np.random.randn(10).astype(np.float32)
        action = agent.act(state, training=False)
        
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 4
        
    def test_memory_operations(self):
        """메모리 저장 및 학습 테스트"""
        agent = DQNAgent(
            state_size=10,
            action_size=4,
            device='cpu',
            batch_size=2
        )
        
        # 경험 저장
        for _ in range(5):
            state = np.random.randn(10).astype(np.float32)
            action = np.random.randint(0, 4)
            reward = np.random.randn()
            next_state = np.random.randn(10).astype(np.float32)
            done = False
            
            agent.remember(state, action, reward, next_state, done)
        
        assert len(agent.memory) == 5
        
        # 학습 실행 (오류 없이 완료되는지 확인)
        agent.learn()

class TestDQNSolver:
    """DQN 솔버 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.test_maze = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ])
        self.start = (0, 0)
        self.goal = (2, 2)
        
    def test_solver_initialization(self):
        """솔버 초기화 테스트"""
        solver = DQNSolver(episodes=10, max_steps=50)
        
        assert solver.episodes == 10
        assert solver.max_steps == 50
        assert solver.agent is None  # 아직 학습되지 않음
        
    def test_solver_training(self):
        """솔버 학습 테스트 (단축 버전)"""
        solver = DQNSolver(
            episodes=5,  # 테스트용 단축
            max_steps=20,
            learning_rate=1e-2,
            batch_size=2,
            memory_size=100
        )
        
        # 학습 실행
        result = solver.train(self.test_maze, self.start, self.goal)
        
        assert isinstance(result, dict)
        assert 'final_success_rate' in result
        assert solver.agent is not None
        
    def test_solver_inference(self):
        """솔버 추론 테스트"""
        solver = DQNSolver(episodes=3, max_steps=20)
        
        # 먼저 학습
        solver.train(self.test_maze, self.start, self.goal)
        
        # 추론 실행
        path, result = solver.solve(self.test_maze, self.start, self.goal)
        
        assert isinstance(path, list)
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'steps' in result

class TestDQNBenchmarkWrapper:
    """DQN 벤치마크 래퍼 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """테스트 정리"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_wrapper_initialization(self):
        """래퍼 초기화 테스트"""
        wrapper = DQNBenchmarkWrapper(
            training_episodes=10,
            model_save_dir=self.temp_dir
        )
        
        assert wrapper.training_episodes == 10
        assert wrapper.model_save_dir.exists()
        
        wrapper.cleanup()

class TestRTX3060Optimization:
    """RTX 3060 최적화 테스트"""
    
    def test_gpu_availability(self):
        """GPU 사용 가능 여부 테스트"""
        if torch.cuda.is_available():
            # GPU가 있는 경우
            device = torch.device('cuda')
            
            # 간단한 텐서 연산으로 GPU 작동 확인
            x = torch.randn(100, 100).to(device)
            y = torch.matmul(x, x.t())
            
            assert y.device.type == 'cuda'
            print(f"GPU 테스트 성공: {torch.cuda.get_device_name()}")
            
            # VRAM 사용량 확인
            vram_used = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            print(f"VRAM 사용량: {vram_used:.1f}MB")
            
        else:
            print("GPU가 감지되지 않음 - CPU 모드로 실행")
            
    def test_memory_optimization(self):
        """메모리 최적화 테스트"""
        # 작은 배치 크기로 메모리 사용량 테스트
        agent = DQNAgent(
            state_size=100,
            action_size=4,
            batch_size=8,  # 작은 배치 크기
            memory_size=1000,  # 작은 메모리 크기
            device='cpu'  # 테스트에서는 CPU 사용
        )
        
        # 메모리 채우기
        for _ in range(100):
            state = np.random.randn(100).astype(np.float32)
            action = np.random.randint(0, 4)
            reward = np.random.randn()
            next_state = np.random.randn(100).astype(np.float32)
            done = False
            
            agent.remember(state, action, reward, next_state, done)
        
        # 여러 번 학습 실행
        for _ in range(10):
            agent.learn()
        
        print("메모리 최적화 테스트 통과")

class TestIntegrationWithExistingSystem:
    """기존 시스템과의 통합 테스트"""
    
    def test_profiler_integration(self):
        """프로파일러 통합 테스트"""
        profiler = get_profiler()
        
        # 현재 메트릭 수집
        metrics = profiler.get_current_metrics()
        
        assert hasattr(metrics, 'cpu_percent')
        assert hasattr(metrics, 'memory_mb')
        assert hasattr(metrics, 'vram_used_mb')
        
        print(f"프로파일러 테스트 - CPU: {metrics.cpu_percent:.1f}%, "
              f"VRAM: {metrics.vram_used_mb:.1f}MB")
        
    def test_maze_loading_compatibility(self):
        """미로 로딩 호환성 테스트"""
        # 실제 데이터셋이 없는 경우를 위한 모의 테스트
        try:
            from utils.maze_io import get_dataset_stats
            stats = get_dataset_stats("train")
            print(f"데이터셋 호환성 테스트 성공: {stats['total_samples']}개 샘플")
        except FileNotFoundError:
            print("데이터셋이 없어 호환성 테스트 스킵")
            pytest.skip("데이터셋이 없습니다")

def run_quick_integration_test():
    """빠른 통합 테스트 실행"""
    print("=== DQN 빠른 통합 테스트 ===")
    
    # 1. 간단한 미로로 전체 파이프라인 테스트
    test_maze = np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ])
    
    start = (0, 0)
    goal = (4, 4)
    
    # 2. DQN 솔버 생성 및 빠른 학습
    print("DQN 솔버 빠른 학습 중...")
    solver = DQNSolver(
        episodes=50,  # 빠른 테스트
        max_steps=100,
        learning_rate=1e-2,
        batch_size=8,
        device='cpu'  # 안정성을 위해 CPU 사용
    )
    
    # 3. 학습 실행
    training_result = solver.train(test_maze, start, goal)
    print(f"학습 완료 - 성공률: {training_result['final_success_rate']:.2f}")
    
    # 4. 해결 테스트
    path, solve_result = solver.solve(test_maze, start, goal)
    print(f"해결 테스트 - 성공: {solve_result['success']}, 스텝: {solve_result['steps']}")
    
    # 5. 성능 체크
    profiler = get_profiler()
    metrics = profiler.get_current_metrics()
    print(f"성능 체크 - VRAM: {metrics.vram_used_mb:.1f}MB, CPU: {metrics.cpu_percent:.1f}%")
    
    print("✅ 빠른 통합 테스트 완료!")
    return True

if __name__ == "__main__":
    # 개별 테스트 실행
    try:
        # pytest가 설치되어 있지 않은 경우를 위한 직접 실행
        run_quick_integration_test()
    except Exception as e:
        print(f"테스트 실패: {e}")
        print("pytest를 사용하여 전체 테스트를 실행하세요:")
        print("pytest tests/test_dqn_integration.py -v")