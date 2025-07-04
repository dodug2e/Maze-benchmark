"""
A* 알고리즘 구현 (algorithms/astar.py)
미로 벤치마크용 - 랩 세미나 버전
"""

import heapq
import time
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from PIL import Image
import json
from algorithms import BaseAlgorithm

class AStarAlgorithm(BaseAlgorithm):
    """A* 알고리즘 래퍼 클래스"""
    
    def __init__(self, name: str = "A*"):
        super().__init__(name)
        self.solver = None
        
    def configure(self, config: dict):
        """알고리즘 설정"""
        super().configure(config)
        self.solver = AStarSolver(
            diagonal_movement=config.get('diagonal_movement', False)
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
                'total_steps': result.total_steps,
                'failure_reason': result.failure_reason
            }
        }

@dataclass
class AStarResult:
    """A* 알고리즘 실행 결과"""
    algorithm: str = "A*"
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

class AStarSolver:
    """A* 알고리즘 미로 해결기"""
    
    def __init__(self, diagonal_movement: bool = False):
        """
        초기화
        Args:
            diagonal_movement: 대각선 이동 허용 여부
        """
        self.diagonal_movement = diagonal_movement
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 상하좌우
        
        if diagonal_movement:
            self.directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])  # 대각선
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """
        휴리스틱 함수 (맨하탄 거리 또는 유클리드 거리)
        Args:
            a: 현재 위치
            b: 목표 위치
        Returns:
            추정 거리
        """
        if self.diagonal_movement:
            # 유클리드 거리 (대각선 이동 허용시)
            return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
        else:
            # 맨하탄 거리 (상하좌우만 허용시)
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, pos: Tuple[int, int], maze: np.ndarray) -> List[Tuple[int, int]]:
        """
        현재 위치에서 이동 가능한 이웃 노드들을 반환
        Args:
            pos: 현재 위치
            maze: 미로 배열 (0: 통로, 1: 벽)
        Returns:
            이동 가능한 위치들의 리스트
        """
        neighbors = []
        rows, cols = maze.shape
        
        for dx, dy in self.directions:
            new_x, new_y = pos[0] + dx, pos[1] + dy
            
            # 경계 체크
            if 0 <= new_x < rows and 0 <= new_y < cols:
                # 벽이 아닌 경우만 추가
                if maze[new_x, new_y] == 0:
                    neighbors.append((new_x, new_y))
        
        return neighbors
    
    def reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], 
                        current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        경로 재구성
        Args:
            came_from: 각 노드의 이전 노드 정보
            current: 현재 노드 (목표 지점)
        Returns:
            시작점부터 목표점까지의 경로
        """
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)  # 시작점 추가
        return path[::-1]  # 역순으로 반환
    
    def solve(self, maze: np.ndarray, start: Tuple[int, int], 
             goal: Tuple[int, int], max_steps: int = 100000) -> AStarResult:
        """
        A* 알고리즘으로 미로 해결
        Args:
            maze: 미로 배열 (0: 통로, 1: 벽)
            start: 시작점 (row, col)
            goal: 목표점 (row, col)
            max_steps: 최대 탐색 스텝 수
        Returns:
            AStarResult 객체
        """
        start_time = time.time()
        result = AStarResult()
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
        
        # A* 알고리즘 초기화
        open_set = [(0, start)]  # (f_score, position)
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        visited = set()
        steps = 0
        
        while open_set and steps < max_steps:
            steps += 1
            
            # f_score가 가장 낮은 노드 선택
            current_f, current = heapq.heappop(open_set)
            
            # 목표 도달 확인
            if current == goal:
                result.solution_found = True
                result.path = self.reconstruct_path(came_from, current)
                result.solution_length = len(result.path)
                result.total_steps = steps
                result.execution_time = time.time() - start_time
                return result
            
            visited.add(current)
            
            # 이웃 노드들 탐색
            for neighbor in self.get_neighbors(current, maze):
                if neighbor in visited:
                    continue
                
                # 이웃까지의 실제 거리 계산
                if self.diagonal_movement and abs(neighbor[0] - current[0]) + abs(neighbor[1] - current[1]) == 2:
                    # 대각선 이동
                    tentative_g_score = g_score[current] + 1.414  # sqrt(2)
                else:
                    # 직선 이동
                    tentative_g_score = g_score[current] + 1.0
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # 해결 실패
        result.failure_reason = f"최대 스텝 수({max_steps}) 초과 또는 경로 없음"
        result.total_steps = steps
        result.execution_time = time.time() - start_time
        return result

def load_maze_from_image(image_path: str) -> np.ndarray:
    """
    이미지에서 미로 배열 로드
    Args:
        image_path: 이미지 파일 경로
    Returns:
        미로 배열 (0: 통로, 1: 벽)
    """
    try:
        img = Image.open(image_path).convert('L')  # 그레이스케일 변환
        maze = np.array(img)
        
        # 이진화 (임계값 128)
        maze = (maze < 128).astype(np.uint8)
        return maze
    except Exception as e:
        raise ValueError(f"이미지 로드 실패: {e}")

def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """
    메타데이터 JSON 파일 로드
    Args:
        metadata_path: JSON 파일 경로
    Returns:
        메타데이터 딕셔너리
    """
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"메타데이터 로드 실패: {e}")

def run_astar_benchmark(sample_id: str, subset: str = "test", 
                       diagonal_movement: bool = False) -> AStarResult:
    """
    A* 벤치마크 실행
    Args:
        sample_id: 샘플 ID (예: "000001")
        subset: 데이터셋 subset ("train", "valid", "test")
        diagonal_movement: 대각선 이동 허용 여부
    Returns:
        AStarResult 객체
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
        
        # A* 솔버 초기화 및 실행
        solver = AStarSolver(diagonal_movement=diagonal_movement)
        result = solver.solve(maze, start, goal)
        
        # 결과에 메타데이터 추가
        result.maze_id = sample_id
        result.algorithm = f"A*{'_diagonal' if diagonal_movement else ''}"
        
        return result
        
    except Exception as e:
        result = AStarResult()
        result.maze_id = sample_id
        result.algorithm = f"A*{'_diagonal' if diagonal_movement else ''}"
        result.failure_reason = f"실행 오류: {e}"
        return result

# 테스트 및 예제 실행
if __name__ == "__main__":
    # 간단한 테스트 미로
    test_maze = np.array([
        [0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ])
    
    start = (0, 0)
    goal = (4, 4)
    
    solver = AStarSolver(diagonal_movement=False)
    result = solver.solve(test_maze, start, goal)
    
    print(f"해결 성공: {result.solution_found}")
    print(f"경로 길이: {result.solution_length}")
    print(f"실행 시간: {result.execution_time:.4f}초")
    print(f"탐색 스텝: {result.total_steps}")
    
    if result.solution_found:
        print("경로:", result.path)

from algorithms import BaseAlgorithm

class AStarAlgorithm(BaseAlgorithm):
    """A* 알고리즘 래퍼 클래스"""
    
    def __init__(self, name: str = "A*"):
        super().__init__(name)
        self.solver = None
        
    def configure(self, config: dict):
        """알고리즘 설정"""
        super().configure(config)
        self.solver = AStarSolver(
            diagonal_movement=config.get('diagonal_movement', False)
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
                'total_steps': result.total_steps,
                'failure_reason': result.failure_reason
            }
        }