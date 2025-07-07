"""
ACO (Ant Colony Optimization) 알고리즘 구현
미로 벤치마크용 - 랩 세미나 버전
"""

import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import random
from algorithms import BaseAlgorithm

@dataclass
class ACOResult:
    """ACO 알고리즘 실행 결과"""
    algorithm: str = "ACO"
    maze_id: str = ""
    maze_size: Tuple[int, int] = (0, 0)
    execution_time: float = 0.0
    solution_found: bool = False
    solution_length: int = 0
    total_steps: int = 0
    failure_reason: str = ""
    path: List[Tuple[int, int]] = None
    iterations: int = 0
    convergence_iteration: int = 0

class Ant:
    """개미 클래스"""
    
    def __init__(self, start_pos: Tuple[int, int]):
        self.start_pos = start_pos
        self.current_pos = start_pos
        self.path = [start_pos]
        self.visited = {start_pos}
        self.path_length = 0
        self.is_alive = True
        
    def reset(self):
        """개미 상태 리셋"""
        self.current_pos = self.start_pos
        self.path = [self.start_pos]
        self.visited = {self.start_pos}
        self.path_length = 0
        self.is_alive = True
        
    def move(self, new_pos: Tuple[int, int]):
        """개미 이동"""
        if self.is_alive:
            self.current_pos = new_pos
            self.path.append(new_pos)
            self.visited.add(new_pos)
            self.path_length += 1

class ACOSolver:
    """ACO 미로 해결기"""
    
    def __init__(self, 
                 n_ants: int = 30,
                 n_iterations: int = 100,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 rho: float = 0.1,
                 Q: float = 100.0):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # 페로몬 가중치
        self.beta = beta    # 휴리스틱 가중치
        self.rho = rho      # 페로몬 증발률
        self.Q = Q          # 페로몬 강도
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
    def euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """유클리드 거리 계산"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_valid_neighbors(self, pos: Tuple[int, int], maze: np.ndarray) -> List[Tuple[int, int]]:
        """유효한 이웃 위치들 반환"""
        neighbors = []
        rows, cols = maze.shape
        
        for dx, dy in self.directions:
            new_x, new_y = pos[0] + dx, pos[1] + dy
            
            if (0 <= new_x < rows and 0 <= new_y < cols and 
                maze[new_x, new_y] == 0):
                neighbors.append((new_x, new_y))
                
        return neighbors
    
    def calculate_transition_probability(self, 
                                       current_pos: Tuple[int, int],
                                       next_pos: Tuple[int, int],
                                       goal: Tuple[int, int],
                                       pheromone_map: np.ndarray,
                                       valid_neighbors: List[Tuple[int, int]]) -> float:
        """전이 확률 계산"""
        pheromone = pheromone_map[next_pos[0], next_pos[1]]
        distance_heuristic = 1.0 / (self.euclidean_distance(next_pos, goal) + 1e-8)
        
        numerator = (pheromone ** self.alpha) * (distance_heuristic ** self.beta)
        
        denominator = 0.0
        for neighbor in valid_neighbors:
            p = pheromone_map[neighbor[0], neighbor[1]]
            h = 1.0 / (self.euclidean_distance(neighbor, goal) + 1e-8)
            denominator += (p ** self.alpha) * (h ** self.beta)
        
        if denominator == 0:
            return 1.0 / len(valid_neighbors)
        
        return numerator / denominator
    
    def select_next_position(self, 
                           ant: Ant,
                           maze: np.ndarray,
                           goal: Tuple[int, int],
                           pheromone_map: np.ndarray) -> Optional[Tuple[int, int]]:
        """다음 위치 선택"""
        valid_neighbors = self.get_valid_neighbors(ant.current_pos, maze)
        unvisited_neighbors = [pos for pos in valid_neighbors if pos not in ant.visited]
        
        if not unvisited_neighbors:
            return None
        
        probabilities = []
        for neighbor in unvisited_neighbors:
            prob = self.calculate_transition_probability(
                ant.current_pos, neighbor, goal, pheromone_map, unvisited_neighbors
            )
            probabilities.append(prob)
        
        total_prob = sum(probabilities)
        if total_prob == 0:
            return random.choice(unvisited_neighbors)
        
        probabilities = [p / total_prob for p in probabilities]
        return unvisited_neighbors[np.random.choice(len(unvisited_neighbors), p=probabilities)]
    
    def update_pheromone(self, pheromone_map: np.ndarray, ants: List[Ant], goal: Tuple[int, int]):
        """페로몬 업데이트"""
        pheromone_map *= (1 - self.rho)
        
        for ant in ants:
            if ant.current_pos == goal:
                pheromone_strength = self.Q / ant.path_length
                for pos in ant.path:
                    pheromone_map[pos[0], pos[1]] += pheromone_strength
    
    def solve(self, maze: np.ndarray, start: Tuple[int, int], 
             goal: Tuple[int, int], max_steps: int = 10000) -> ACOResult:
        """ACO 알고리즘으로 미로 해결"""
        start_time = time.time()
        result = ACOResult()
        result.maze_size = maze.shape
        
        if maze[start[0], start[1]] == 0:
            result.failure_reason = "시작점이 벽입니다"
            result.execution_time = time.time() - start_time
            return result
        
        if maze[goal[0], goal[1]] == 0:
            result.failure_reason = "목표점이 벽입니다"
            result.execution_time = time.time() - start_time
            return result
        
        rows, cols = maze.shape
        pheromone_map = np.ones((rows, cols)) * 0.1
        
        ants = [Ant(start) for _ in range(self.n_ants)]
        
        best_path = None
        best_path_length = float('inf')
        convergence_iteration = 0
        total_steps = 0
        
        for iteration in range(self.n_iterations):
            for ant in ants:
                ant.reset()
                
                while ant.is_alive and ant.current_pos != goal and total_steps < max_steps:
                    total_steps += 1
                    
                    next_pos = self.select_next_position(ant, maze, goal, pheromone_map)
                    
                    if next_pos is None:
                        ant.is_alive = False
                        break
                    
                    ant.move(next_pos)
                    
                    if len(ant.path) > rows * cols:
                        ant.is_alive = False
                        break
                
                if ant.current_pos == goal and ant.path_length < best_path_length:
                    best_path = ant.path.copy()
                    best_path_length = ant.path_length
                    convergence_iteration = iteration
            
            self.update_pheromone(pheromone_map, ants, goal)
            
            if total_steps >= max_steps:
                break
        
        if best_path is not None:
            result.solution_found = True
            result.path = best_path
            result.solution_length = len(best_path)
        else:
            result.failure_reason = "해결책을 찾을 수 없습니다"
        
        result.total_steps = total_steps
        result.iterations = iteration + 1
        result.convergence_iteration = convergence_iteration
        result.execution_time = time.time() - start_time
        
        return result

class ACOAlgorithm(BaseAlgorithm):
    """ACO 알고리즘 래퍼 클래스"""
    
    def __init__(self, name: str = "ACO"):
        super().__init__(name)
        self.solver = None
        
    def configure(self, config: dict):
        """알고리즘 설정"""
        super().configure(config)
        self.solver = ACOSolver(
            n_ants=config.get('num_ants', 30),
            n_iterations=config.get('max_iterations', 100),
            alpha=config.get('alpha', 1.0),
            beta=config.get('beta', 2.0),
            rho=config.get('rho', 0.1),
            Q=config.get('Q', 100.0)
        )
    
    def solve(self, maze_array, metadata):
        """미로 해결"""
        if self.solver is None:
            self.configure({})
            
        start = tuple(metadata.get('entrance', (0, 0)))
        goal = tuple(metadata.get('exit', (maze_array.shape[0]-1, maze_array.shape[1]-1)))
        
        result = self.solver.solve(maze_array, start, goal)
        
        return {
            'success': result.solution_found,
            'solution_path': result.path if result.solution_found else [],
            'solution_length': result.solution_length,
            'execution_time': result.execution_time,
            'additional_info': {
                'iterations': result.iterations,
                'convergence_iteration': result.convergence_iteration,
                'total_steps': result.total_steps,
                'failure_reason': result.failure_reason
            }
        }