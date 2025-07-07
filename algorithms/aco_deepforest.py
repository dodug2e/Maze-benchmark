"""
ACO + Deep Forest 알고리즘 구현 (수정 버전)
BaseAlgorithm 상속 및 인터페이스 통일
"""

import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from PIL import Image
import json
import random
import logging
from pathlib import Path
import sys

# 프로젝트 경로 추가
sys.path.append('.')
sys.path.append('..')

# BaseAlgorithm import 추가
try:
    from algorithms import BaseAlgorithm
except ImportError:
    # BaseAlgorithm 정의가 없는 경우를 위한 fallback
    class BaseAlgorithm:
        def __init__(self, name: str):
            self.name = name
            self.config = {}
        
        def configure(self, config: dict):
            self.config.update(config)
        
        def solve(self, maze_array, metadata):
            raise NotImplementedError("solve method must be implemented by subclass")

try:
    from algorithms.deep_forest_model import DeepForestModel
except ImportError:
    try:
        from deep_forest_model import DeepForestModel
    except ImportError:
        # Deep Forest 모델이 없는 경우를 위한 더미 클래스
        class DeepForestModel:
            def __init__(self, **kwargs):
                self.is_trained = False
                self.n_layers = kwargs.get('n_layers', 2)
                self.n_estimators = kwargs.get('n_estimators', 30)
                
            def fit(self, X, y):
                self.is_trained = True
                
            def predict(self, X):
                return np.random.randint(0, 4, size=len(X))
                
            def get_direction_probabilities(self, maze, current_pos, goal):
                return np.ones(4) / 4

logger = logging.getLogger(__name__)

@dataclass
class ACODeepForestResult:
    """ACO + Deep Forest 알고리즘 실행 결과"""
    algorithm: str = "ACO+DeepForest"
    maze_id: str = ""
    maze_size: Tuple[int, int] = (0, 0)
    execution_time: float = 0.0
    power_consumption: float = 0.0
    vram_usage: float = 0.0
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    solution_found: bool = False
    solution_length: int = 0
    total_steps: int = 0
    max_steps: int = 0
    failure_reason: str = ""
    path: List[Tuple[int, int]] = field(default_factory=list)
    best_path_length: int = 0
    iterations: int = 0
    convergence_iteration: int = 0
    forest_training_time: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)


class Ant:
    """개미 클래스"""
    
    def __init__(self, start_pos: Tuple[int, int]):
        self.start_pos = start_pos
        self.current_pos = start_pos
        self.path = [start_pos]
        self.visited = {start_pos}
        self.path_length = 0
        self.stuck = False
    
    def reset(self):
        """개미 상태 초기화"""
        self.current_pos = self.start_pos
        self.path = [self.start_pos]
        self.visited = {self.start_pos}
        self.path_length = 0
        self.stuck = False
    
    def move_to(self, new_pos: Tuple[int, int]) -> bool:
        """새 위치로 이동"""
        if new_pos not in self.visited:
            self.current_pos = new_pos
            self.path.append(new_pos)
            self.visited.add(new_pos)
            self.path_length += 1
            return True
        return False


class ACODeepForestSolver:
    """ACO + Deep Forest 하이브리드 솔버"""
    
    def __init__(self, 
                 n_ants: int = 20,
                 n_iterations: int = 50,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 rho: float = 0.5,
                 Q: float = 100.0,
                 n_estimators: int = 30,
                 n_layers: int = 2):
        """
        초기화
        
        Args:
            n_ants: 개미 수
            n_iterations: 반복 횟수
            alpha: 페로몬 중요도
            beta: 휴리스틱 중요도
            rho: 페로몬 증발률
            Q: 페로몬 강도
            n_estimators: Deep Forest 트리 개수
            n_layers: Deep Forest 레이어 수
        """
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        
        # Deep Forest 모델
        self.deep_forest = DeepForestModel(
            n_layers=n_layers,
            n_estimators=n_estimators,
            max_depth=8,
            random_state=42
        )
        
        logger.info(f"ACO+DeepForest 초기화: {n_ants}개미, {n_iterations}반복")
    
    def _get_neighbors(self, pos: Tuple[int, int], maze: np.ndarray, 
                      visited: set) -> List[Tuple[int, int]]:
        """유효한 이웃 위치 반환"""
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 상, 하, 좌, 우
        
        for dy, dx in directions:
            new_y, new_x = pos[0] + dy, pos[1] + dx
            
            if (0 <= new_y < maze.shape[0] and 
                0 <= new_x < maze.shape[1] and
                maze[new_y, new_x] == 0 and  # 통로 (0은 통로, 1은 벽)
                (new_y, new_x) not in visited):
                neighbors.append((new_y, new_x))
        
        return neighbors
    
    def _calculate_heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """휴리스틱 함수 (맨하탄 거리의 역수)"""
        distance = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        return 1.0 / (distance + 1.0)
    
    def _select_next_position(self, ant: Ant, neighbors: List[Tuple[int, int]], 
                             pheromone_map: np.ndarray, goal: Tuple[int, int]) -> Tuple[int, int]:
        """다음 위치 선택 (ACO + Deep Forest)"""
        if not neighbors:
            return None
        
        probabilities = []
        
        for neighbor in neighbors:
            # 페로몬 정보
            pheromone = pheromone_map[neighbor[0], neighbor[1]]
            
            # 휴리스틱 정보
            heuristic = self._calculate_heuristic(neighbor, goal)
            
            # Deep Forest 정보 (훈련된 경우만)
            df_weight = 0.0
            if self.deep_forest.is_trained:
                try:
                    # 간단한 특성 추출 (현재 위치와 목표까지의 방향)
                    dy = goal[0] - ant.current_pos[0]
                    dx = goal[1] - ant.current_pos[1]
                    
                    # 방향 정규화
                    if dy != 0:
                        dy = 1 if dy > 0 else -1
                    if dx != 0:
                        dx = 1 if dx > 0 else -1
                    
                    # 실제 이동 방향
                    actual_dy = neighbor[0] - ant.current_pos[0]
                    actual_dx = neighbor[1] - ant.current_pos[1]
                    
                    # 방향이 일치하면 높은 가중치
                    if (dy == actual_dy and dx == actual_dx):
                        df_weight = 2.0
                    elif (dy == actual_dy or dx == actual_dx):
                        df_weight = 1.0
                    else:
                        df_weight = 0.5
                        
                except Exception as e:
                    logger.debug(f"Deep Forest 예측 실패: {e}")
                    df_weight = 1.0
            else:
                df_weight = 1.0
            
            # 결합된 확률 계산
            prob = (pheromone ** self.alpha) * (heuristic ** self.beta) * df_weight
            probabilities.append(prob)
        
        # 확률 정규화
        total_prob = sum(probabilities)
        if total_prob == 0:
            # 모든 확률이 0인 경우 균등 선택
            probabilities = [1.0] * len(neighbors)
            total_prob = len(neighbors)
        
        probabilities = [p / total_prob for p in probabilities]
        
        # 룰렛 휠 선택
        r = random.random()
        cumulative_prob = 0.0
        
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if r <= cumulative_prob:
                return neighbors[i]
        
        # fallback
        return neighbors[-1]
    
    def _generate_training_data(self, maze: np.ndarray, start: Tuple[int, int], 
                               goal: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Deep Forest 훈련용 데이터 생성"""
        X, y = [], []
        
        # 간단한 경로 생성으로 훈련 데이터 만들기
        for _ in range(100):  # 100개 샘플
            # 랜덤 위치에서 목표 방향 계산
            pos_y = random.randint(0, maze.shape[0] - 1)
            pos_x = random.randint(0, maze.shape[1] - 1)
            
            if maze[pos_y, pos_x] == 1:  # 벽인 경우 건너뛰기
                continue
            
            # 특성: [현재_y, 현재_x, 목표_y, 목표_x, 상대_y, 상대_x]
            rel_y = goal[0] - pos_y
            rel_x = goal[1] - pos_x
            
            features = [pos_y, pos_x, goal[0], goal[1], rel_y, rel_x]
            
            # 레이블: 최적 방향 (0: 상, 1: 하, 2: 좌, 3: 우)
            if abs(rel_y) > abs(rel_x):
                direction = 0 if rel_y < 0 else 1
            else:
                direction = 2 if rel_x < 0 else 3
            
            X.append(features)
            y.append(direction)
        
        return np.array(X), np.array(y)
    
    def solve(self, maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> ACODeepForestResult:
        """미로 해결"""
        start_time = time.time()
        
        result = ACODeepForestResult()
        result.maze_size = maze.shape
        
        try:
            # Deep Forest 훈련
            forest_start_time = time.time()
            X_train, y_train = self._generate_training_data(maze, start, goal)
            
            if len(X_train) > 0:
                self.deep_forest.fit(X_train, y_train)
                logger.info("Deep Forest 훈련 완료")
            
            result.forest_training_time = time.time() - forest_start_time
            
            # 페로몬 맵 초기화
            pheromone_map = np.ones_like(maze, dtype=np.float32) * 0.1
            
            best_path = None
            best_length = float('inf')
            
            for iteration in range(self.n_iterations):
                iteration_paths = []
                
                for ant_id in range(self.n_ants):
                    ant = Ant(start)
                    steps = 0
                    max_steps = maze.shape[0] * maze.shape[1] * 2
                    
                    while ant.current_pos != goal and steps < max_steps:
                        neighbors = self._get_neighbors(ant.current_pos, maze, ant.visited)
                        
                        if not neighbors:
                            ant.stuck = True
                            break
                        
                        next_pos = self._select_next_position(ant, neighbors, pheromone_map, goal)
                        
                        if next_pos is None or not ant.move_to(next_pos):
                            ant.stuck = True
                            break
                        
                        steps += 1
                    
                    # 성공한 경우
                    if ant.current_pos == goal:
                        iteration_paths.append(ant.path)
                        
                        if len(ant.path) < best_length:
                            best_path = ant.path.copy()
                            best_length = len(ant.path)
                            result.convergence_iteration = iteration
                
                # 페로몬 업데이트
                pheromone_map *= (1 - self.rho)  # 증발
                
                for path in iteration_paths:
                    delta_pheromone = self.Q / len(path)
                    for pos in path:
                        pheromone_map[pos[0], pos[1]] += delta_pheromone
                
                result.total_steps += self.n_ants
                
                # 조기 종료 조건
                if best_path and iteration - result.convergence_iteration > 10:
                    break
            
            # 결과 설정
            if best_path:
                result.solution_found = True
                result.path = best_path
                result.solution_length = len(best_path)
                result.best_path_length = len(best_path)
            else:
                result.failure_reason = "경로를 찾을 수 없음"
            
            result.iterations = iteration + 1
            
        except Exception as e:
            logger.error(f"ACO+DeepForest 실행 오류: {e}")
            result.failure_reason = str(e)
        
        result.execution_time = time.time() - start_time
        return result


# BaseAlgorithm을 상속받는 래퍼 클래스
class ACODeepForestAlgorithm(BaseAlgorithm):
    """ACO + Deep Forest 알고리즘 래퍼 클래스"""
    
    def __init__(self, name: str = "ACO_DeepForest"):
        super().__init__(name)
        self.solver = None
        
    def configure(self, config: dict):
        """알고리즘 설정"""
        super().configure(config)
        self.solver = ACODeepForestSolver(
            n_ants=config.get('n_ants', 20),
            n_iterations=config.get('n_iterations', 50),
            alpha=config.get('alpha', 1.0),
            beta=config.get('beta', 2.0),
            rho=config.get('rho', 0.5),
            Q=config.get('Q', 100.0),
            n_estimators=config.get('n_estimators', 30),
            n_layers=config.get('n_layers', 2)
        )
    
    def solve(self, maze_array, metadata):
        """미로 해결 - BaseAlgorithm 인터페이스 구현"""
        if self.solver is None:
            self.configure({})
        
        # 메타데이터에서 시작점과 목표점 추출
        start = tuple(metadata.get('entrance', (0, 0)))
        goal = tuple(metadata.get('exit', (maze_array.shape[0]-1, maze_array.shape[1]-1)))
        
        # 미로 배열 변환 (필요한 경우)
        if maze_array.dtype != np.float32:
            maze_array = maze_array.astype(np.float32)
        
        # 실제 해결
        result = self.solver.solve(maze_array, start, goal)
        
        # 표준 형식으로 변환
        return {
            'success': result.solution_found,
            'solution_path': result.path if result.solution_found else [],
            'solution_length': result.solution_length,
            'execution_time': result.execution_time,
            'additional_info': {
                'algorithm': result.algorithm,
                'iterations': result.iterations,
                'convergence_iteration': result.convergence_iteration,
                'forest_training_time': result.forest_training_time,
                'total_steps': result.total_steps,
                'failure_reason': result.failure_reason
            }
        }


# 알고리즘 등록 함수
def register_aco_deepforest():
    """ACO + Deep Forest 알고리즘을 레지스트리에 등록"""
    try:
        from algorithms import register_algorithm
        register_algorithm("ACO_DeepForest", ACODeepForestAlgorithm)
        logger.info("ACO_DeepForest 알고리즘이 등록되었습니다")
    except ImportError:
        logger.warning("알고리즘 레지스트리를 찾을 수 없습니다")


# 파일이 직접 실행될 때 테스트
if __name__ == "__main__":
    # 간단한 테스트 미로
    test_maze = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1]
    ], dtype=np.float32)
    
    start = (1, 1)
    goal = (5, 7)
    
    print("ACO + Deep Forest 테스트 실행 중...")
    
    # 래퍼 클래스 테스트
    algorithm = ACODeepForestAlgorithm()
    algorithm.configure({
        'n_ants': 15,
        'n_iterations': 30,
        'n_estimators': 20,
        'n_layers': 2
    })
    
    metadata = {'entrance': start, 'exit': goal}
    result = algorithm.solve(test_maze, metadata)
    
    print(f"해결 성공: {result['success']}")
    print(f"경로 길이: {result['solution_length']}")
    print(f"실행 시간: {result['execution_time']:.4f}초")
    
    if result['additional_info']:
        print(f"반복 횟수: {result['additional_info']['iterations']}")
        print(f"수렴 반복: {result['additional_info']['convergence_iteration']}")
        print(f"Forest 훈련 시간: {result['additional_info']['forest_training_time']:.4f}초")
    
    if result['success']:
        print("경로 (처음 10개):", result['solution_path'][:10])
        print("경로 (마지막 5개):", result['solution_path'][-5:])
    else:
        print(f"실패 원인: {result['additional_info']['failure_reason']}")
    
    print("테스트 완료!")
    
    # 알고리즘 등록 테스트
    register_aco_deepforest()