"""
ACO + CNN 알고리즘 구현 (algorithms/aco_cnn.py)
미로 벤치마크용 - 랩 세미나 버전

ACO (Ant Colony Optimization) + CNN 하이브리드 접근법:
1. CNN이 미로의 특징을 학습하여 휴리스틱 정보 제공
2. ACO가 페로몬 기반으로 경로 탐색 및 최적화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from PIL import Image
import json
import random

@dataclass
class ACOCNNResult:
    """ACO + CNN 알고리즘 실행 결과"""
    algorithm: str = "ACO+CNN"
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
    path: List[Tuple[int, int]] = None
    best_path_length: int = 0
    iterations: int = 0
    convergence_iteration: int = 0

class MazeFeatureExtractor(nn.Module):
    """CNN 기반 미로 특징 추출기"""
    
    def __init__(self, input_size: int = 64):
        super().__init__()
        self.input_size = input_size
        
        # CNN 레이어들
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        
        # 출력 레이어 (휴리스틱 맵 생성)
        self.heuristic_output = nn.Conv2d(32, 1, kernel_size=1)
        
        # 배치 정규화
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)
        
        # 드롭아웃
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        Args:
            x: 입력 미로 텐서 (B, 1, H, W)
        Returns:
            휴리스틱 맵 (B, 1, H, W)
        """
        # Encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        # Decoder
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # 휴리스틱 맵 생성
        heuristic = torch.sigmoid(self.heuristic_output(x))
        
        return heuristic

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

class ACOCNNSolver:
    """ACO + CNN 하이브리드 미로 해결기"""
    
    def __init__(self, 
                 n_ants: int = 20,
                 n_iterations: int = 100,
                 alpha: float = 1.0,      # 페로몬 가중치
                 beta: float = 2.0,       # 휴리스틱 가중치
                 gamma: float = 1.5,      # CNN 휴리스틱 가중치
                 rho: float = 0.1,        # 페로몬 증발률
                 Q: float = 100.0,        # 페로몬 강도
                 device: str = None):
        """
        초기화
        Args:
            n_ants: 개미 개수
            n_iterations: 반복 횟수
            alpha: 페로몬 가중치
            beta: 거리 기반 휴리스틱 가중치
            gamma: CNN 휴리스틱 가중치
            rho: 페로몬 증발률
            Q: 페로몬 강도
            device: PyTorch 디바이스
        """
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.Q = Q
        
        # 디바이스 설정
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # CNN 모델 초기화
        self.cnn = MazeFeatureExtractor().to(self.device)
        self.cnn.eval()  # 추론 모드
        
        # 방향벡터 (상, 하, 좌, 우)
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
    def preprocess_maze(self, maze: np.ndarray) -> torch.Tensor:
        """
        미로 전처리 (CNN 입력용)
        Args:
            maze: 미로 배열
        Returns:
            전처리된 텐서
        """
        # 텐서 변환 및 정규화
        maze_tensor = torch.FloatTensor(maze).unsqueeze(0).unsqueeze(0)
        maze_tensor = maze_tensor.to(self.device)
        
        # 0-1 정규화
        maze_tensor = maze_tensor.float()
        
        return maze_tensor
    
    def get_cnn_heuristic(self, maze: np.ndarray) -> np.ndarray:
        """
        CNN을 사용하여 휴리스틱 맵 생성
        Args:
            maze: 미로 배열
        Returns:
            휴리스틱 맵
        """
        with torch.no_grad():
            maze_tensor = self.preprocess_maze(maze)
            heuristic_map = self.cnn(maze_tensor)
            return heuristic_map.squeeze().cpu().numpy()
    
    def euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """유클리드 거리 계산"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_valid_neighbors(self, pos: Tuple[int, int], maze: np.ndarray) -> List[Tuple[int, int]]:
        """유효한 이웃 위치들 반환"""
        neighbors = []
        rows, cols = maze.shape
        
        for dx, dy in self.directions:
            new_x, new_y = pos[0] + dx, pos[1] + dy
            
            # 경계 체크 및 벽 체크
            if (0 <= new_x < rows and 0 <= new_y < cols and 
                maze[new_x, new_y] == 0):
                neighbors.append((new_x, new_y))
                
        return neighbors
    
    def calculate_transition_probability(self, 
                                       current_pos: Tuple[int, int],
                                       next_pos: Tuple[int, int],
                                       goal: Tuple[int, int],
                                       pheromone_map: np.ndarray,
                                       cnn_heuristic: np.ndarray,
                                       valid_neighbors: List[Tuple[int, int]]) -> float:
        """
        전이 확률 계산 (ACO + CNN 하이브리드)
        Args:
            current_pos: 현재 위치
            next_pos: 다음 위치
            goal: 목표 위치
            pheromone_map: 페로몬 맵
            cnn_heuristic: CNN 휴리스틱 맵
            valid_neighbors: 유효한 이웃들
        Returns:
            전이 확률
        """
        # 페로몬 강도
        pheromone = pheromone_map[next_pos[0], next_pos[1]]
        
        # 거리 기반 휴리스틱 (역수 - 가까울수록 높은 값)
        distance_heuristic = 1.0 / (self.euclidean_distance(next_pos, goal) + 1e-8)
        
        # CNN 휴리스틱
        cnn_value = cnn_heuristic[next_pos[0], next_pos[1]]
        
        # 분자 계산
        numerator = (pheromone ** self.alpha) * (distance_heuristic ** self.beta) * (cnn_value ** self.gamma)
        
        # 분모 계산 (모든 유효한 이웃들에 대해)
        denominator = 0.0
        for neighbor in valid_neighbors:
            p = pheromone_map[neighbor[0], neighbor[1]]
            h = 1.0 / (self.euclidean_distance(neighbor, goal) + 1e-8)
            c = cnn_heuristic[neighbor[0], neighbor[1]]
            denominator += (p ** self.alpha) * (h ** self.beta) * (c ** self.gamma)
        
        if denominator == 0:
            return 1.0 / len(valid_neighbors)
        
        return numerator / denominator
    
    def select_next_position(self, 
                           ant: Ant,
                           maze: np.ndarray,
                           goal: Tuple[int, int],
                           pheromone_map: np.ndarray,
                           cnn_heuristic: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        다음 위치 선택 (룰렛 휠 선택)
        Args:
            ant: 개미 객체
            maze: 미로
            goal: 목표 위치
            pheromone_map: 페로몬 맵
            cnn_heuristic: CNN 휴리스틱 맵
        Returns:
            선택된 다음 위치
        """
        valid_neighbors = self.get_valid_neighbors(ant.current_pos, maze)
        
        # 방문하지 않은 이웃들만 필터링
        unvisited_neighbors = [pos for pos in valid_neighbors if pos not in ant.visited]
        
        if not unvisited_neighbors:
            return None  # 갈 곳이 없음
        
        # 각 이웃에 대한 확률 계산
        probabilities = []
        for neighbor in unvisited_neighbors:
            prob = self.calculate_transition_probability(
                ant.current_pos, neighbor, goal, pheromone_map, cnn_heuristic, unvisited_neighbors
            )
            probabilities.append(prob)
        
        # 룰렛 휠 선택
        total_prob = sum(probabilities)
        if total_prob == 0:
            return random.choice(unvisited_neighbors)
        
        probabilities = [p / total_prob for p in probabilities]
        return np.random.choice(unvisited_neighbors, p=probabilities)
    
    def update_pheromone(self, pheromone_map: np.ndarray, ants: List[Ant], goal: Tuple[int, int]):
        """
        페로몬 업데이트
        Args:
            pheromone_map: 페로몬 맵
            ants: 개미들
            goal: 목표 위치
        """
        # 페로몬 증발
        pheromone_map *= (1 - self.rho)
        
        # 각 개미의 경로에 페로몬 추가
        for ant in ants:
            if ant.current_pos == goal:  # 목표에 도달한 개미만
                pheromone_strength = self.Q / ant.path_length
                for pos in ant.path:
                    pheromone_map[pos[0], pos[1]] += pheromone_strength
    
    def solve(self, maze: np.ndarray, start: Tuple[int, int], 
             goal: Tuple[int, int], max_steps: int = 10000) -> ACOCNNResult:
        """
        ACO + CNN 알고리즘으로 미로 해결
        Args:
            maze: 미로 배열
            start: 시작점
            goal: 목표점
            max_steps: 최대 스텝 수
        Returns:
            ACOCNNResult 객체
        """
        start_time = time.time()
        result = ACOCNNResult()
        result.maze_size = maze.shape
        result.max_steps = max_steps
        
        # 시작점이나 목표점이 벽인 경우
        if maze[start[0], start[1]] == 1:
            result.failure_reason = "시작점이 벽입니다"
            result.execution_time = time.time() - start_time
            return result
        
        if maze[goal[0], goal[1]] == 1:
            result.failure_reason = "목표점이 벽입니다"
            result.execution_time = time.time() - start_time
            return result
        
        # 초기화
        rows, cols = maze.shape
        pheromone_map = np.ones((rows, cols)) * 0.1  # 초기 페로몬
        
        # CNN 휴리스틱 맵 생성
        cnn_heuristic = self.get_cnn_heuristic(maze)
        
        # 개미들 초기화
        ants = [Ant(start) for _ in range(self.n_ants)]
        
        best_path = None
        best_path_length = float('inf')
        convergence_iteration = 0
        total_steps = 0
        
        # ACO 메인 루프
        for iteration in range(self.n_iterations):
            # 각 개미 이동
            for ant in ants:
                ant.reset()
                
                # 각 개미가 목표에 도달하거나 막힐 때까지 이동
                while ant.is_alive and ant.current_pos != goal and total_steps < max_steps:
                    total_steps += 1
                    
                    next_pos = self.select_next_position(
                        ant, maze, goal, pheromone_map, cnn_heuristic
                    )
                    
                    if next_pos is None:
                        ant.is_alive = False
                        break
                    
                    ant.move(next_pos)
                    
                    # 경로가 너무 길면 중단
                    if len(ant.path) > rows * cols:
                        ant.is_alive = False
                        break
                
                # 목표에 도달한 경우 최적 경로 업데이트
                if ant.current_pos == goal and ant.path_length < best_path_length:
                    best_path = ant.path.copy()
                    best_path_length = ant.path_length
                    convergence_iteration = iteration
            
            # 페로몬 업데이트
            self.update_pheromone(pheromone_map, ants, goal)
            
            # 조기 종료 조건
            if total_steps >= max_steps:
                break
        
        # 결과 설정
        if best_path is not None:
            result.solution_found = True
            result.path = best_path
            result.solution_length = len(best_path)
            result.best_path_length = best_path_length
        else:
            result.failure_reason = "해결책을 찾을 수 없습니다"
        
        result.total_steps = total_steps
        result.iterations = iteration + 1
        result.convergence_iteration = convergence_iteration
        result.execution_time = time.time() - start_time
        
        return result

def load_maze_from_image(image_path: str) -> np.ndarray:
    """이미지에서 미로 배열 로드"""
    try:
        img = Image.open(image_path).convert('L')
        maze = np.array(img)
        maze = (maze < 128).astype(np.uint8)
        return maze
    except Exception as e:
        raise ValueError(f"이미지 로드 실패: {e}")

def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """메타데이터 JSON 파일 로드"""
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"메타데이터 로드 실패: {e}")

def run_aco_cnn_benchmark(sample_id: str, subset: str = "test",
                         n_ants: int = 20, n_iterations: int = 50) -> ACOCNNResult:
    """
    ACO + CNN 벤치마크 실행
    Args:
        sample_id: 샘플 ID
        subset: 데이터셋 subset
        n_ants: 개미 개수
        n_iterations: 반복 횟수
    Returns:
        ACOCNNResult 객체
    """
    # 파일 경로 구성
    base_path = f"datasets/{subset}"
    img_path = f"{base_path}/img/{sample_id}.png"
    metadata_path = f"{base_path}/metadata/{sample_id}.json"
    
    try:
        # 데이터 로드
        maze = load_maze_from_image(img_path)
        metadata = load_metadata(metadata_path)
        
        # 시작점과 목표점 추출
        start = tuple(metadata['start'])
        goal = tuple(metadata['goal'])
        
        # ACO + CNN 솔버 초기화 및 실행
        solver = ACOCNNSolver(n_ants=n_ants, n_iterations=n_iterations)
        result = solver.solve(maze, start, goal)
        
        # 결과에 메타데이터 추가
        result.maze_id = sample_id
        
        return result
        
    except Exception as e:
        result = ACOCNNResult()
        result.maze_id = sample_id
        result.failure_reason = f"실행 오류: {e}"
        return result

# 테스트 및 예제 실행
if __name__ == "__main__":
    # 간단한 테스트 미로
    test_maze = np.array([
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ])
    
    start = (0, 0)
    goal = (6, 6)
    
    print("ACO + CNN 테스트 실행 중...")
    solver = ACOCNNSolver(n_ants=10, n_iterations=20)
    result = solver.solve(test_maze, start, goal)
    
    print(f"해결 성공: {result.solution_found}")
    print(f"경로 길이: {result.solution_length}")
    print(f"실행 시간: {result.execution_time:.4f}초")
    print(f"총 반복: {result.iterations}")
    print(f"수렴 반복: {result.convergence_iteration}")
    print(f"탐색 스텝: {result.total_steps}")
    
    if result.solution_found:
        print("경로 (처음 10개):", result.path[:10])

from algorithms import BaseAlgorithm

class ACOCNNAlgorithm(BaseAlgorithm):
    """ACO+CNN 알고리즘 래퍼 클래스"""
    
    def __init__(self, name: str = "ACO+CNN"):
        super().__init__(name)
        self.solver = None
        
    def configure(self, config: dict):
        """알고리즘 설정"""
        super().configure(config)
        self.solver = ACOCNNSolver(
            n_ants=config.get('num_ants', 20),
            n_iterations=config.get('max_iterations', 100),
            alpha=config.get('alpha', 1.0),
            beta=config.get('beta', 2.0),
            gamma=config.get('gamma', 1.5),
            device=config.get('device', None)
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