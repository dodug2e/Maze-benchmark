"""
ACO + Deep Forest 알고리즘 구현 (algorithms/aco_deepforest.py)
미로 벤치마크용 - 랩 세미나 버전

ACO (Ant Colony Optimization) + Deep Forest 하이브리드 접근법:
1. Deep Forest가 미로의 지역적 특징을 학습하여 휴리스틱 정보 제공
2. ACO가 페로몬 기반으로 경로 탐색 및 최적화
3. Deep Forest는 CNN보다 메모리 효율적이고 해석가능
"""

import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from PIL import Image
import json
import random
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

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
    path: List[Tuple[int, int]] = None
    best_path_length: int = 0
    iterations: int = 0
    convergence_iteration: int = 0
    forest_training_time: float = 0.0
    feature_importance: Dict[str, float] = None

class DeepForestFeatureExtractor:
    """Deep Forest 기반 미로 특징 추출기"""
    
    def __init__(self, 
                 n_estimators: int = 100,
                 n_layers: int = 3,
                 window_size: int = 5):
        """
        초기화
        Args:
            n_estimators: 각 포레스트의 트리 개수
            n_layers: Deep Forest 레이어 수
            window_size: 특징 추출 윈도우 크기
        """
        self.n_estimators = n_estimators
        self.n_layers = n_layers
        self.window_size = window_size
        self.scaler = StandardScaler()
        
        # 각 레이어의 포레스트들
        self.forests = []
        for layer in range(n_layers):
            layer_forests = {
                'rf': RandomForestClassifier(n_estimators=n_estimators, 
                                           random_state=42 + layer,
                                           n_jobs=-1),
                'et': ExtraTreesClassifier(n_estimators=n_estimators,
                                         random_state=42 + layer,
                                         n_jobs=-1)
            }
            self.forests.append(layer_forests)
        
        self.is_trained = False
        self.feature_importance_ = {}
        
    def extract_local_features(self, maze: np.ndarray, pos: Tuple[int, int]) -> np.ndarray:
        """
        지역 특징 추출
        Args:
            maze: 미로 배열
            pos: 현재 위치
        Returns:
            특징 벡터
        """
        rows, cols = maze.shape
        r, c = pos
        w = self.window_size // 2
        
        # 윈도우 영역 추출 (패딩 적용)
        window = np.ones((self.window_size, self.window_size))  # 기본값: 벽
        
        for i in range(self.window_size):
            for j in range(self.window_size):
                maze_r = r - w + i
                maze_c = c - w + j
                
                if 0 <= maze_r < rows and 0 <= maze_c < cols:
                    window[i, j] = maze[maze_r, maze_c]
        
        # 기본 특징들
        features = []
        
        # 1. 윈도우 내 벽의 비율
        wall_ratio = np.sum(window == 1) / (self.window_size * self.window_size)
        features.append(wall_ratio)
        
        # 2. 중심으로부터의 거리별 벽 밀도
        center = self.window_size // 2
        for radius in range(1, center + 1):
            mask = np.zeros_like(window)
            for i in range(self.window_size):
                for j in range(self.window_size):
                    if abs(i - center) + abs(j - center) == radius:  # 맨하탄 거리
                        mask[i, j] = 1
            
            if np.sum(mask) > 0:
                density = np.sum((window == 1) & (mask == 1)) / np.sum(mask)
                features.append(density)
        
        # 3. 방향별 벽 개수
        directions = [
            window[:center, center],      # 위
            window[center+1:, center],    # 아래
            window[center, :center],      # 왼쪽
            window[center, center+1:]     # 오른쪽
        ]
        
        for direction in directions:
            wall_count = np.sum(direction == 1)
            features.append(wall_count)
        
        # 4. 대각선 방향별 벽 개수
        diagonals = [
            np.diag(window),                    # 주대각선
            np.diag(np.fliplr(window))         # 반대각선
        ]
        
        for diagonal in diagonals:
            wall_count = np.sum(diagonal == 1)
            features.append(wall_count)
        
        # 5. 윈도우의 연결성 (connected components)
        # 간단한 연결성 측정: 4방향 연결된 빈 공간의 개수
        empty_spaces = (window == 0).astype(int)
        connectivity = np.sum(empty_spaces)
        features.append(connectivity)
        
        # 6. 엔트로피 (불확실성 측정)
        if np.sum(window == 0) > 0 and np.sum(window == 1) > 0:
            p0 = np.sum(window == 0) / (self.window_size * self.window_size)
            p1 = np.sum(window == 1) / (self.window_size * self.window_size)
            entropy = -p0 * np.log2(p0) - p1 * np.log2(p1)
        else:
            entropy = 0.0
        features.append(entropy)
        
        return np.array(features)
    
    def generate_training_data(self, maze: np.ndarray, 
                             start: Tuple[int, int], 
                             goal: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        훈련 데이터 생성
        Args:
            maze: 미로 배열
            start: 시작점
            goal: 목표점
        Returns:
            특징 행렬, 레이블 벡터
        """
        rows, cols = maze.shape
        features = []
        labels = []
        
        # 모든 빈 공간에 대해 특징 추출
        for r in range(rows):
            for c in range(cols):
                if maze[r, c] == 0:  # 빈 공간만
                    feature = self.extract_local_features(maze, (r, c))
                    features.append(feature)
                    
                    # 레이블: 목표점까지의 맨하탄 거리의 역수 (가까울수록 높은 값)
                    distance = abs(r - goal[0]) + abs(c - goal[1])
                    label = 1.0 / (distance + 1)  # 0이 되지 않도록 +1
                    labels.append(label)
        
        features = np.array(features)
        labels = np.array(labels)
        
        # 레이블을 분류 문제로 변환 (상위 25%를 1, 나머지를 0)
        threshold = np.percentile(labels, 75)
        binary_labels = (labels >= threshold).astype(int)
        
        return features, binary_labels
    
    def train(self, maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]):
        """
        Deep Forest 훈련
        Args:
            maze: 미로 배열
            start: 시작점
            goal: 목표점
        """
        # 훈련 데이터 생성
        X, y = self.generate_training_data(maze, start, goal)
        
        if len(X) == 0:
            raise ValueError("훈련 데이터를 생성할 수 없습니다")
        
        # 특징 정규화
        X = self.scaler.fit_transform(X)
        
        # Deep Forest 훈련
        layer_input = X.copy()
        
        for layer_idx, layer_forests in enumerate(self.forests):
            layer_predictions = []
            
            # 각 포레스트 훈련
            for forest_name, forest in layer_forests.items():
                forest.fit(layer_input, y)
                
                # 확률 예측
                if hasattr(forest, 'predict_proba'):
                    proba = forest.predict_proba(layer_input)
                    if proba.shape[1] == 2:  # 이진 분류
                        layer_predictions.append(proba[:, 1])
                    else:
                        layer_predictions.append(proba[:, 0])
                else:
                    pred = forest.predict(layer_input)
                    layer_predictions.append(pred)
            
            # 다음 레이어를 위한 입력 준비
            if layer_idx < len(self.forests) - 1:
                layer_output = np.column_stack(layer_predictions)
                layer_input = np.column_stack([layer_input, layer_output])
        
        # 특징 중요도 계산 (마지막 레이어 기준)
        last_layer = self.forests[-1]
        for forest_name, forest in last_layer.items():
            if hasattr(forest, 'feature_importances_'):
                self.feature_importance_[forest_name] = forest.feature_importances_
        
        self.is_trained = True
    
    def predict_heuristic(self, maze: np.ndarray, pos: Tuple[int, int]) -> float:
        """
        위치에 대한 휴리스틱 값 예측
        Args:
            maze: 미로 배열
            pos: 위치
        Returns:
            휴리스틱 값 (0-1)
        """
        if not self.is_trained:
            return 0.5  # 기본값
        
        # 특징 추출
        features = self.extract_local_features(maze, pos).reshape(1, -1)
        features = self.scaler.transform(features)
        
        # Deep Forest 예측
        layer_input = features.copy()
        
        for layer_idx, layer_forests in enumerate(self.forests):
            layer_predictions = []
            
            for forest_name, forest in layer_forests.items():
                if hasattr(forest, 'predict_proba'):
                    proba = forest.predict_proba(layer_input)
                    if proba.shape[1] == 2:
                        layer_predictions.append(proba[:, 1])
                    else:
                        layer_predictions.append(proba[:, 0])
                else:
                    pred = forest.predict(layer_input)
                    layer_predictions.append(pred)
            
            # 다음 레이어를 위한 입력 준비
            if layer_idx < len(self.forests) - 1:
                layer_output = np.column_stack(layer_predictions)
                layer_input = np.column_stack([layer_input, layer_output])
        
        # 최종 예측값 (모든 포레스트의 평균)
        final_predictions = np.array([pred[0] for pred in layer_predictions])
        return np.mean(final_predictions)

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

class ACODeepForestSolver:
    """ACO + Deep Forest 하이브리드 미로 해결기"""
    
    def __init__(self, 
                 n_ants: int = 20,
                 n_iterations: int = 100,
                 alpha: float = 1.0,      # 페로몬 가중치
                 beta: float = 2.0,       # 휴리스틱 가중치
                 gamma: float = 1.5,      # Deep Forest 휴리스틱 가중치
                 rho: float = 0.1,        # 페로몬 증발률
                 Q: float = 100.0,        # 페로몬 강도
                 n_estimators: int = 50,  # 포레스트 크기
                 n_layers: int = 2):      # Deep Forest 레이어 수
        """
        초기화
        Args:
            n_ants: 개미 개수
            n_iterations: 반복 횟수
            alpha: 페로몬 가중치
            beta: 거리 기반 휴리스틱 가중치
            gamma: Deep Forest 휴리스틱 가중치
            rho: 페로몬 증발률
            Q: 페로몬 강도
            n_estimators: 각 포레스트의 트리 개수
            n_layers: Deep Forest 레이어 수
        """
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.Q = Q
        
        # Deep Forest 특징 추출기
        self.deep_forest = DeepForestFeatureExtractor(
            n_estimators=n_estimators,
            n_layers=n_layers
        )
        
        # 방향벡터 (상, 하, 좌, 우)
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
                                       maze: np.ndarray,
                                       valid_neighbors: List[Tuple[int, int]]) -> float:
        """
        전이 확률 계산 (ACO + Deep Forest 하이브리드)
        """
        # 페로몬 강도
        pheromone = pheromone_map[next_pos[0], next_pos[1]]
        
        # 거리 기반 휴리스틱
        distance_heuristic = 1.0 / (self.euclidean_distance(next_pos, goal) + 1e-8)
        
        # Deep Forest 휴리스틱
        df_heuristic = self.deep_forest.predict_heuristic(maze, next_pos)
        
        # 분자 계산
        numerator = (pheromone ** self.alpha) * (distance_heuristic ** self.beta) * (df_heuristic ** self.gamma)
        
        # 분모 계산
        denominator = 0.0
        for neighbor in valid_neighbors:
            p = pheromone_map[neighbor[0], neighbor[1]]
            h = 1.0 / (self.euclidean_distance(neighbor, goal) + 1e-8)
            d = self.deep_forest.predict_heuristic(maze, neighbor)
            denominator += (p ** self.alpha) * (h ** self.beta) * (d ** self.gamma)
        
        if denominator == 0:
            return 1.0 / len(valid_neighbors)
        
        return numerator / denominator
    
    def select_next_position(self, 
                           ant: Ant,
                           maze: np.ndarray,
                           goal: Tuple[int, int],
                           pheromone_map: np.ndarray) -> Optional[Tuple[int, int]]:
        """다음 위치 선택 (룰렛 휠 선택)"""
        valid_neighbors = self.get_valid_neighbors(ant.current_pos, maze)
        unvisited_neighbors = [pos for pos in valid_neighbors if pos not in ant.visited]
        
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
        
        probabilities = [p / total_prob for p in probabilities]
        return np.random.choice(unvisited_neighbors, p=probabilities)
    
    def update_pheromone(self, pheromone_map: np.ndarray, ants: List[Ant], goal: Tuple[int, int]):
        """페로몬 업데이트"""
        # 페로몬 증발
        pheromone_map *= (1 - self.rho)
        
        # 각 개미의 경로에 페로몬 추가
        for ant in ants:
            if ant.current_pos == goal:
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
        
        # 시작점이나 목표점이 벽인 경우
        if maze[start[0], start[1]] == 1:
            result.failure_reason = "시작점이 벽입니다"
            result.execution_time = time.time() - start_time
            return result
        
        if maze[goal[0], goal[1]] == 1:
            result.failure_reason = "목표점이 벽입니다"
            result.execution_time = time.time() - start_time
            return result
        
        # Deep Forest 훈련
        forest_start = time.time()
        try:
            self.deep_forest.train(maze, start, goal)
            result.forest_training_time = time.time() - forest_start
            result.feature_importance = self.deep_forest.feature_importance_
        except Exception as e:
            result.failure_reason = f"Deep Forest 훈련 실패: {e}"
            result.execution_time = time.time() - start_time
            return result
        
        # 초기화
        rows, cols = maze.shape
        pheromone_map = np.ones((rows, cols)) * 0.1
        
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
                
                # 최적 경로 업데이트
                if ant.current_pos == goal and ant.path_length < best_path_length:
                    best_path = ant.path.copy()
                    best_path_length = ant.path_length
                    convergence_iteration = iteration
            
            # 페로몬 업데이트
            self.update_pheromone(pheromone_map, ants, goal)
            
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

def run_aco_deepforest_benchmark(sample_id: str, subset: str = "test",
                                n_ants: int = 20, n_iterations: int = 50) -> ACODeepForestResult:
    """
    ACO + Deep Forest 벤치마크 실행
    """
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
    # 테스트 미로
    test_maze = np.array([
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    
    start = (0, 0)
    goal = (8, 8)
    
    print("ACO + Deep Forest 테스트 실행 중...")
    solver = ACODeepForestSolver(n_ants=15, n_iterations=30, n_estimators=30, n_layers=2)
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
    
    if result.feature_importance:
        print("특징 중요도:", result.feature_importance)