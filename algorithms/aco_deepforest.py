"""
ACO + Deep Forest 알고리즘 구현 (수정 버전)
자료형 불일치 문제 해결 및 경로 통일
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

try:
    from algorithms.deep_forest_model import DeepForestModel
except ImportError:
    # 수정된 DeepForest 모델 import
    from deep_forest_model import DeepForestModel

logger = logging.getLogger(__name__)

@dataclass
class ACODeepForestResult:
    """ACO + Deep Forest 알고리즘 실행 결과 (수정 버전)"""
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
    """개미 클래스 (자료형 통일)"""
    
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
    """ACO + Deep Forest 하이브리드 솔버 (수정 버전)"""
    
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
        
        # Deep Forest 모델 (자료형 통일)
        self.deep_forest = DeepForestModel(
            n_layers=n_layers,
            n_estimators=n_estimators,
            max_depth=8,  # RTX 3060 최적화
            random_state=42
        )
        
        # 자료형 통일
        self.dtype = np.float32
        
        logger.info(f"ACO+DeepForest 초기화: {n_ants}개미, {n_iterations}반복, dtype={self.dtype}")
    
    def _get_neighbors(self, pos: Tuple[int, int], maze: np.ndarray, 
                      visited: set) -> List[Tuple[int, int]]:
        """유효한 이웃 위치 반환"""
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 상, 하, 좌, 우
        
        for dy, dx in directions:
            new_y, new_x = pos[0] + dy, pos[1] + dx
            
            if (0 <= new_y < maze.shape[0] and 
                0 <= new_x < maze.shape[1] and
                maze[new_y, new_x] == 0 and  # 통로
                (new_y, new_x) not in visited):
                neighbors.append((new_y, new_x))
        
        return neighbors
    
    def calculate_transition_probability(self, current_pos: Tuple[int, int],
                                       neighbor: Tuple[int, int],
                                       goal: Tuple[int, int],
                                       pheromone_map: np.ndarray,
                                       maze: np.ndarray,
                                       all_neighbors: List[Tuple[int, int]]) -> float:
        """전이 확률 계산 (Deep Forest 휴리스틱 포함)"""
        
        # 페로몬 정보
        pheromone = pheromone_map[neighbor[0], neighbor[1]]
        
        # Deep Forest 휴리스틱 (훈련된 경우만)
        if self.deep_forest.is_trained:
            try:
                direction_probs = self.deep_forest.get_direction_probabilities(
                    maze, current_pos, goal
                )
                
                # 방향 매핑
                dy = neighbor[0] - current_pos[0]
                dx = neighbor[1] - current_pos[1]
                
                if dy == -1:
                    direction_idx = 0  # 위
                elif dy == 1:
                    direction_idx = 1  # 아래
                elif dx == -1:
                    direction_idx = 2  # 왼쪽
                else:  # dx == 1
                    direction_idx = 3  # 오른쪽
                
                heuristic = float(direction_probs[direction_idx])
                
            except Exception as e:
                logger.warning(f"Deep Forest 휴리스틱 계산 실패: {e}")
                # 맨하탄 거리 기반 휴리스틱으로 대체
                heuristic = 1.0 / (1.0 + abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1]))
        else:
            # 맨하탄 거리 기반 휴리스틱
            heuristic = 1.0 / (1.0 + abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1]))
        
        # 전이 확률 계산
        if pheromone == 0 and heuristic == 0:
            return 0.0
        
        probability = (pheromone ** self.alpha) * (heuristic ** self.beta)
        return float(probability)
    
    def select_next_position(self, ant: Ant, goal: Tuple[int, int],
                           pheromone_map: np.ndarray, maze: np.ndarray) -> Optional[Tuple[int, int]]:
        """다음 위치 선택 (룰렛 휠 선택)"""
        
        unvisited_neighbors = self._get_neighbors(ant.current_pos, maze, ant.visited)
        
        if not unvisited_neighbors:
            return None
        
        # 각 이웃에 대한 확률 계산
        probabilities = []
        for neighbor in unvisited_neighbors:
            prob = self.calculate_transition_probability(
                ant.current_pos, neighbor, goal, pheromone_map, maze, unvisited_neighbors
            )
            probabilities.append(prob)
        
        # 룰렛 휠 선택
        total_prob = sum(probabilities)
        if total_prob == 0:
            return random.choice(unvisited_neighbors)
        
        # 확률 정규화
        probabilities = [p / total_prob for p in probabilities]
        
        # numpy를 사용한 선택
        try:
            choice_idx = np.random.choice(len(unvisited_neighbors), p=probabilities)
            return unvisited_neighbors[choice_idx]
        except Exception:
            # 예외 발생시 랜덤 선택
            return random.choice(unvisited_neighbors)
    
    def update_pheromone(self, pheromone_map: np.ndarray, ants: List[Ant], goal: Tuple[int, int]):
        """페로몬 업데이트"""
        # 페로몬 증발
        pheromone_map *= (1 - self.rho)
        
        # 각 개미의 경로에 페로몬 추가
        for ant in ants:
            if ant.current_pos == goal and len(ant.path) > 1:
                pheromone_strength = self.Q / ant.path_length
                for pos in ant.path:
                    pheromone_map[pos[0], pos[1]] += pheromone_strength
    
    def solve(self, maze: np.ndarray, start: Tuple[int, int], 
             goal: Tuple[int, int], max_steps: int = 10000) -> ACODeepForestResult:
        """
        ACO + Deep Forest 알고리즘으로 미로 해결
        """
        start_time = time.time()
        result = ACODeepForestResult()
        result.maze_size = maze.shape
        result.max_steps = max_steps
        
        # 자료형 통일
        maze = maze.astype(self.dtype)
        
        # 시작점이나 목표점이 벽인 경우
        if maze[start[0], start[1]] == 0:
            result.failure_reason = "시작점이 벽입니다"
            result.execution_time = time.time() - start_time
            return result
        
        if maze[goal[0], goal[1]] == 0:
            result.failure_reason = "목표점이 벽입니다"
            result.execution_time = time.time() - start_time
            return result
        
        # Deep Forest 훈련
        forest_start = time.time()
        try:
            self.deep_forest.train(maze, start, goal)
            result.forest_training_time = time.time() - forest_start
            result.feature_importance = {"deep_forest_trained": True}
        except Exception as e:
            logger.warning(f"Deep Forest 훈련 실패: {e}")
            result.failure_reason = f"Deep Forest 훈련 실패: {e}"
            result.execution_time = time.time() - start_time
            return result
        
        # 초기화
        rows, cols = maze.shape
        pheromone_map = np.ones((rows, cols), dtype=self.dtype)
        
        best_path = None
        best_length = float('inf')
        
        # ACO 반복
        for iteration in range(self.n_iterations):
            # 개미들 초기화
            ants = [Ant(start) for _ in range(self.n_ants)]
            
            # 각 개미가 경로 탐색
            for ant in ants:
                steps = 0
                while ant.current_pos != goal and steps < max_steps // self.n_ants:
                    next_pos = self.select_next_position(ant, goal, pheromone_map, maze)
                    
                    if next_pos is None:
                        ant.stuck = True
                        break
                    
                    ant.move_to(next_pos)
                    steps += 1
                
                # 최적 경로 업데이트
                if ant.current_pos == goal and ant.path_length < best_length:
                    best_path = ant.path.copy()
                    best_length = ant.path_length
                    result.convergence_iteration = iteration
            
            # 페로몬 업데이트
            self.update_pheromone(pheromone_map, ants, goal)
            
            # 조기 종료 조건
            if best_path is not None and iteration > 10:
                successful_ants = sum(1 for ant in ants if ant.current_pos == goal)
                if successful_ants >= self.n_ants * 0.8:  # 80% 이상 성공시
                    break
        
        # 결과 설정
        result.iterations = iteration + 1
        
        if best_path is not None:
            result.solution_found = True
            result.path = best_path
            result.solution_length = len(best_path)
            result.best_path_length = best_length
            result.total_steps = sum(len(ant.path) for ant in ants)
        else:
            result.failure_reason = "해결책을 찾지 못했습니다"
            result.total_steps = max_steps
        
        result.execution_time = time.time() - start_time
        return result


# 유틸리티 함수들 (경로 통일)
def load_maze_from_image(img_path: str) -> np.ndarray:
    """이미지에서 미로 로드 (자료형 통일)"""
    try:
        img = Image.open(img_path).convert("L")
        maze = (np.asarray(img, dtype=np.float32) < 128).astype(np.float32)
        return maze
    except Exception as e:
        raise ValueError(f"이미지 로드 실패: {e}")

def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """메타데이터 로드"""
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 키 정규화
        if 'entrance' in metadata and 'start' not in metadata:
            metadata['start'] = metadata['entrance']
        if 'exit' in metadata and 'goal' not in metadata:
            metadata['goal'] = metadata['exit']
        elif 'exit_point' in metadata and 'goal' not in metadata:
            metadata['goal'] = metadata['exit_point']
            
        return metadata
    except Exception as e:
        raise ValueError(f"메타데이터 로드 실패: {e}")

def run_aco_deepforest_benchmark(sample_id: str, subset: str = "test",
                                n_ants: int = 20, n_iterations: int = 50) -> ACODeepForestResult:
    """
    ACO + Deep Forest 벤치마크 실행 (경로 통일)
    """
    # 경로 통일
    base_path = Path("datasets") / subset
    img_path = base_path / "img" / f"{sample_id}.png"  # 통일된 경로
    metadata_path = base_path / "metadata" / f"{sample_id}.json"
    
    try:
        # 데이터 로드
        maze = load_maze_from_image(str(img_path))
        metadata = load_metadata(str(metadata_path))
        
        # 시작점과 목표점 추출
        start = tuple(metadata['start'])
        goal = tuple(metadata['goal'])
        
        # ACO + Deep Forest 솔버 초기화 및 실행
        solver = ACODeepForestSolver(n_ants=n_ants, n_iterations=n_iterations)
        result = solver.solve(maze, start, goal)
        
        # 결과에 메타데이터 추가
        result.maze_id = sample_id
        
        return result
        
    except Exception as e:
        result = ACODeepForestResult()
        result.maze_id = sample_id
        result.failure_reason = f"실행 오류: {e}"
        return result


# 테스트 및 예제 실행
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
    solver = ACODeepForestSolver(n_ants=15, n_iterations=30, n_estimators=20, n_layers=2)
    result = solver.solve(test_maze, start, goal)
    
    print(f"해결 성공: {result.solution_found}")
    print(f"경로 길이: {result.solution_length}")
    print(f"실행 시간: {result.execution_time:.4f}초")
    print(f"Deep Forest 훈련 시간: {result.forest_training_time:.4f}초")
    print(f"총 반복: {result.iterations}")
    print(f"수렴 반복: {result.convergence_iteration}")
    print(f"탐색 스텝: {result.total_steps}")
    
    if result.solution_found:
        print("경로 (처음 10개):", result.path[:10])
        print("경로 (마지막 5개):", result.path[-5:])
    else:
        print(f"실패 원인: {result.failure_reason}")
        
    print("테스트 완료!")