"""
하이브리드 ACO 구현
ACO + CNN, ACO + Deep Forest 결합
"""

import numpy as np
import random
import logging
from typing import List, Tuple, Optional, Dict, Any
import time
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

class Ant:
    """개미 클래스"""
    
    def __init__(self, start_pos: Tuple[int, int], goal_pos: Tuple[int, int]):
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.current_pos = start_pos
        self.path: List[Tuple[int, int]] = [start_pos]
        self.visited = {start_pos}
        self.path_length = 0
        self.is_stuck = False
        self.reached_goal = False
    
    def move(self, next_pos: Tuple[int, int]):
        """다음 위치로 이동"""
        self.path.append(next_pos)
        self.visited.add(next_pos)
        self.current_pos = next_pos
        self.path_length += 1
        
        if next_pos == self.goal_pos:
            self.reached_goal = True
    
    def reset(self):
        """개미 상태 초기화"""
        self.current_pos = self.start_pos
        self.path = [self.start_pos]
        self.visited = {self.start_pos}
        self.path_length = 0
        self.is_stuck = False
        self.reached_goal = False


class BaseACO:
    """기본 ACO 알고리즘"""
    
    def __init__(self,
                 maze: np.ndarray,
                 start_pos: Tuple[int, int],
                 goal_pos: Tuple[int, int],
                 n_ants: int = 20,
                 n_iterations: int = 100,
                 alpha: float = 1.0,    # 페로몬 중요도
                 beta: float = 2.0,     # 휴리스틱 중요도
                 rho: float = 0.1,      # 페로몬 증발률
                 Q: float = 100.0,      # 페로몬 강도
                 max_steps: int = 1000):
        
        self.maze = maze
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.max_steps = max_steps
        
        # 페로몬 매트릭스 초기화
        self.pheromone = np.ones_like(maze, dtype=np.float32) * 0.1
        
        # 휴리스틱 정보 (목표까지의 거리의 역수)
        self.heuristic = self._calculate_heuristic()
        
        # 결과 저장
        self.best_path = None
        self.best_length = float('inf')
        self.iteration_history = []
        
    def _calculate_heuristic(self) -> np.ndarray:
        """휴리스틱 정보 계산 (목표까지의 맨하탄 거리의 역수)"""
        h, w = self.maze.shape
        heuristic = np.zeros_like(self.maze, dtype=np.float32)
        
        for i in range(h):
            for j in range(w):
                if self.maze[i, j] == 1:  # 통로인 경우만
                    distance = abs(i - self.goal_pos[0]) + abs(j - self.goal_pos[1])
                    heuristic[i, j] = 1.0 / (distance + 1.0)
        
        return heuristic
    
    def _get_valid_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """유효한 이웃 위치들 반환"""
        y, x = pos
        neighbors = []
        
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 상하좌우
            ny, nx = y + dy, x + dx
            
            if (0 <= ny < self.maze.shape[0] and 
                0 <= nx < self.maze.shape[1] and 
                self.maze[ny, nx] == 1):
                neighbors.append((ny, nx))
        
        return neighbors
    
    def _calculate_transition_probability(self, ant: Ant, neighbors: List[Tuple[int, int]]) -> np.ndarray:
        """전이 확률 계산"""
        if not neighbors:
            return np.array([])
        
        probabilities = []
        
        for neighbor in neighbors:
            y, x = neighbor
            
            # 이미 방문한 위치는 낮은 확률
            if neighbor in ant.visited:
                prob = 0.01
            else:
                pheromone_val = self.pheromone[y, x] ** self.alpha
                heuristic_val = self.heuristic[y, x] ** self.beta
                prob = pheromone_val * heuristic_val
            
            probabilities.append(prob)
        
        probabilities = np.array(probabilities)
        
        # 정규화
        total = probabilities.sum()
        if total > 0:
            probabilities = probabilities / total
        else:
            # 모든 확률이 0인 경우 균등 분포
            probabilities = np.ones(len(neighbors)) / len(neighbors)
        
        return probabilities
    
    def _select_next_position(self, ant: Ant) -> Optional[Tuple[int, int]]:
        """다음 위치 선택 (기본 ACO 방식)"""
        neighbors = self._get_valid_neighbors(ant.current_pos)
        
        if not neighbors:
            return None
        
        probabilities = self._calculate_transition_probability(ant, neighbors)
        
        # 확률에 따라 선택
        selected_idx = np.random.choice(len(neighbors), p=probabilities)
        return neighbors[selected_idx]
    
    def _move_ant(self, ant: Ant) -> bool:
        """개미 한 스텝 이동"""
        if ant.reached_goal or ant.is_stuck:
            return False
        
        next_pos = self._select_next_position(ant)
        
        if next_pos is None:
            ant.is_stuck = True
            return False
        
        ant.move(next_pos)
        
        # 최대 스텝 수 체크
        if ant.path_length >= self.max_steps:
            ant.is_stuck = True
            return False
        
        return True
    
    def _update_pheromone(self, ants: List[Ant]):
        """페로몬 업데이트"""
        # 페로몬 증발
        self.pheromone *= (1 - self.rho)
        
        # 성공한 개미들의 페로몬 추가
        for ant in ants:
            if ant.reached_goal and len(ant.path) > 1:
                pheromone_delta = self.Q / len(ant.path)
                
                for i in range(len(ant.path) - 1):
                    y, x = ant.path[i]
                    self.pheromone[y, x] += pheromone_delta
        
        # 페로몬 값 제한
        self.pheromone = np.clip(self.pheromone, 0.01, 10.0)
    
    def solve(self) -> Optional[List[Tuple[int, int]]]:
        """ACO 알고리즘 실행"""
        logger.info(f"기본 ACO 시작: {self.n_ants}마리 개미, {self.n_iterations}회 반복")
        
        for iteration in range(self.n_iterations):
            # 개미들 초기화
            ants = [Ant(self.start_pos, self.goal_pos) for _ in range(self.n_ants)]
            
            # 모든 개미가 움직일 때까지 반복
            max_steps_per_iteration = self.max_steps
            for step in range(max_steps_per_iteration):
                active_ants = [ant for ant in ants if not ant.reached_goal and not ant.is_stuck]
                
                if not active_ants:
                    break
                
                for ant in active_ants:
                    self._move_ant(ant)
            
            # 성공한 개미들 확인
            successful_ants = [ant for ant in ants if ant.reached_goal]
            
            if successful_ants:
                # 가장 짧은 경로 찾기
                best_ant = min(successful_ants, key=lambda a: len(a.path))
                
                if len(best_ant.path) < self.best_length:
                    self.best_path = best_ant.path.copy()
                    self.best_length = len(best_ant.path)
                    logger.info(f"반복 {iteration}: 새로운 최단 경로 발견 (길이: {self.best_length})")
            
            # 페로몬 업데이트
            self._update_pheromone(ants)
            
            # 이번 반복 결과 기록
            iteration_result = {
                'iteration': iteration,
                'successful_ants': len(successful_ants),
                'best_length_so_far': self.best_length,
                'current_best': len(successful_ants[0].path) if successful_ants else None
            }
            self.iteration_history.append(iteration_result)
            
            if iteration % 10 == 0:
                logger.info(f"반복 {iteration}: 성공한 개미 {len(successful_ants)}마리, "
                           f"최단 거리: {self.best_length}")
        
        logger.info(f"기본 ACO 완료: 최단 경로 길이 {self.best_length}")
        return self.best_path


class HybridACO_CNN(BaseACO):
    """ACO + CNN 하이브리드 알고리즘"""
    
    def __init__(self, maze, start_pos, goal_pos, cnn_model=None, 
                 cnn_weight=0.3, **kwargs):
        super().__init__(maze, start_pos, goal_pos, **kwargs)
        self.cnn_model = cnn_model
        self.cnn_weight = cnn_weight  # CNN 예측의 가중치
        
        # CNN 모델이 있으면 평가 모드로 설정
        if self.cnn_model:
            self.cnn_model.eval()
            if torch.cuda.is_available():
                self.cnn_model = self.cnn_model.cuda()
    
    def _select_next_position(self, ant: Ant) -> Optional[Tuple[int, int]]:
        """CNN 예측을 고려한 다음 위치 선택"""
        neighbors = self._get_valid_neighbors(ant.current_pos)
        
        if not neighbors:
            return None
        
        # 기본 ACO 확률 계산
        aco_probs = self._calculate_transition_probability(ant, neighbors)
        
        # CNN 예측이 가능한 경우
        if self.cnn_model:
            try:
                cnn_probs = self._get_cnn_probabilities(ant, neighbors)
                
                # ACO와 CNN 확률 결합
                combined_probs = (1 - self.cnn_weight) * aco_probs + self.cnn_weight * cnn_probs
                
            except Exception as e:
                logger.warning(f"CNN 예측 실패: {e}, ACO만 사용")
                combined_probs = aco_probs
        else:
            combined_probs = aco_probs
        
        # 선택
        selected_idx = np.random.choice(len(neighbors), p=combined_probs)
        return neighbors[selected_idx]
    
    def _get_cnn_probabilities(self, ant: Ant, neighbors: List[Tuple[int, int]]) -> np.ndarray:
        """CNN을 사용한 방향 예측"""
        # 미로를 텐서로 변환
        maze_tensor = torch.FloatTensor(self.maze).unsqueeze(0).unsqueeze(0)
        
        if torch.cuda.is_available():
            maze_tensor = maze_tensor.cuda()
        
        with torch.no_grad():
            # CNN 예측
            outputs = self.cnn_model(maze_tensor)
            direction_probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # 방향 매핑 (0: 상, 1: 하, 2: 좌, 3: 우)
        direction_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # 이웃들과 방향 매핑
        neighbor_probs = []
        current_y, current_x = ant.current_pos
        
        for neighbor in neighbors:
            neighbor_y, neighbor_x = neighbor
            dy = neighbor_y - current_y
            dx = neighbor_x - current_x
            
            # 방향 찾기
            direction_idx = None
            for i, (d_y, d_x) in enumerate(direction_map):
                if dy == d_y and dx == d_x:
                    direction_idx = i
                    break
            
            if direction_idx is not None:
                neighbor_probs.append(direction_probs[direction_idx])
            else:
                neighbor_probs.append(0.25)  # 균등 확률
        
        neighbor_probs = np.array(neighbor_probs)
        
        # 정규화
        total = neighbor_probs.sum()
        if total > 0:
            neighbor_probs = neighbor_probs / total
        else:
            neighbor_probs = np.ones(len(neighbors)) / len(neighbors)
        
        return neighbor_probs


class HybridACO_DeepForest(BaseACO):
    """ACO + Deep Forest 하이브리드 알고리즘"""
    
    def __init__(self, maze, start_pos, goal_pos, df_model=None, 
                 df_weight=0.3, **kwargs):
        super().__init__(maze, start_pos, goal_pos, **kwargs)
        self.df_model = df_model
        self.df_weight = df_weight  # Deep Forest 예측의 가중치
    
    def _select_next_position(self, ant: Ant) -> Optional[Tuple[int, int]]:
        """Deep Forest 예측을 고려한 다음 위치 선택"""
        neighbors = self._get_valid_neighbors(ant.current_pos)
        
        if not neighbors:
            return None
        
        # 기본 ACO 확률 계산
        aco_probs = self._calculate_transition_probability(ant, neighbors)
        
        # Deep Forest 예측이 가능한 경우
        if self.df_model:
            try:
                df_probs = self._get_df_probabilities(ant, neighbors)
                
                # ACO와 Deep Forest 확률 결합
                combined_probs = (1 - self.df_weight) * aco_probs + self.df_weight * df_probs
                
            except Exception as e:
                logger.warning(f"Deep Forest 예측 실패: {e}, ACO만 사용")
                combined_probs = aco_probs
        else:
            combined_probs = aco_probs
        
        # 선택
        selected_idx = np.random.choice(len(neighbors), p=combined_probs)
        return neighbors[selected_idx]
    
    def _get_df_probabilities(self, ant: Ant, neighbors: List[Tuple[int, int]]) -> np.ndarray:
        """Deep Forest를 사용한 방향 예측 (가변 크기 미로 지원)"""
        # 목표 위치 설정 (특성 추출에 필요)
        self.df_model.goal_pos = self.goal_pos
        
        # 특성 추출 (적응적 윈도우 크기 사용)
        features = self.df_model._extract_features(self.maze, ant.current_pos, adaptive_window=True)
        features = features.reshape(1, -1)
        
        # Deep Forest 예측
        direction = self.df_model.predict(features)[0]
        
        # 방향을 확률로 변환
        direction_probs = np.array([0.1, 0.1, 0.1, 0.1])  # 기본 확률
        direction_probs[direction] = 0.7  # 예측된 방향에 높은 확률
        
        # 방향 매핑
        direction_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 상하좌우
        
        # 이웃들과 방향 매핑
        neighbor_probs = []
        current_y, current_x = ant.current_pos
        
        for neighbor in neighbors:
            neighbor_y, neighbor_x = neighbor
            dy = neighbor_y - current_y
            dx = neighbor_x - current_x
            
            # 방향 찾기
            direction_idx = None
            for i, (d_y, d_x) in enumerate(direction_map):
                if dy == d_y and dx == d_x:
                    direction_idx = i
                    break
            
            if direction_idx is not None:
                neighbor_probs.append(direction_probs[direction_idx])
            else:
                neighbor_probs.append(0.25)  # 균등 확률
        
        neighbor_probs = np.array(neighbor_probs)
        
        # 정규화
        total = neighbor_probs.sum()
        if total > 0:
            neighbor_probs = neighbor_probs / total
        else:
            neighbor_probs = np.ones(len(neighbors)) / len(neighbors)
        
        return neighbor_probs


class HybridSolver:
    """하이브리드 ACO 솔버 통합 클래스"""
    
    def __init__(self, 
                 maze: np.ndarray,
                 start_pos: Tuple[int, int],
                 goal_pos: Tuple[int, int]):
        self.maze = maze
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.models = {}
        self.results = {}
    
    def load_cnn_model(self, model_path: str):
        """CNN 모델 로드"""
        try:
            from algorithms.cnn_model import MazePathCNN
            
            # 모델 생성
            model = MazePathCNN(input_size=200, num_classes=4)
            
            # 체크포인트 로드
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            self.models['cnn'] = model
            logger.info(f"CNN 모델 로드 완료: {model_path}")
            
        except Exception as e:
            logger.error(f"CNN 모델 로드 실패: {e}")
    
    def load_deep_forest_model(self, model_path: str):
        """Deep Forest 모델 로드"""
        try:
            from algorithms.deep_forest_model import MazeDeepForest
            
            model = MazeDeepForest()
            model.load_model(model_path)
            
            self.models['deep_forest'] = model
            logger.info(f"Deep Forest 모델 로드 완료: {model_path}")
            
        except Exception as e:
            logger.error(f"Deep Forest 모델 로드 실패: {e}")
    
    def solve_with_base_aco(self, **kwargs) -> Dict:
        """기본 ACO로 해결"""
        logger.info("기본 ACO 실행")
        
        # cnn_weight, df_weight 제거
        clean_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ['cnn_weight', 'df_weight']}
        
        start_time = time.time()
        solver = BaseACO(self.maze, self.start_pos, self.goal_pos, **clean_kwargs)
        path = solver.solve()
        execution_time = time.time() - start_time
        
        result = {
            'algorithm': 'aco',
            'path': path,
            'path_length': len(path) if path else 0,
            'execution_time': execution_time,
            'success': path is not None,
            'iteration_history': solver.iteration_history
        }
        
        self.results['aco'] = result
        return result
    
    def solve_with_cnn_hybrid(self, cnn_weight=0.3, **kwargs) -> Dict:
        """CNN 하이브리드로 해결"""
        if 'cnn' not in self.models:
            raise ValueError("CNN 모델이 로드되지 않음")
        
        logger.info(f"ACO+CNN 하이브리드 실행 (CNN 가중치: {cnn_weight})")
        
        # cnn_weight, df_weight 제거
        clean_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ['cnn_weight', 'df_weight']}
        
        start_time = time.time()
        solver = HybridACO_CNN(
            self.maze, self.start_pos, self.goal_pos,
            cnn_model=self.models['cnn'],
            cnn_weight=cnn_weight,
            **clean_kwargs
        )
        path = solver.solve()
        execution_time = time.time() - start_time
        
        result = {
            'algorithm': 'aco_cnn',
            'path': path,
            'path_length': len(path) if path else 0,
            'execution_time': execution_time,
            'success': path is not None,
            'cnn_weight': cnn_weight,
            'iteration_history': solver.iteration_history
        }
        
        self.results['aco_cnn'] = result
        return result
    
    def solve_with_deep_forest_hybrid(self, df_weight=0.3, **kwargs) -> Dict:
        """Deep Forest 하이브리드로 해결"""
        if 'deep_forest' not in self.models:
            raise ValueError("Deep Forest 모델이 로드되지 않음")
        
        logger.info(f"ACO+Deep Forest 하이브리드 실행 (DF 가중치: {df_weight})")
        
        # cnn_weight, df_weight 제거
        clean_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ['cnn_weight', 'df_weight']}
        
        start_time = time.time()
        solver = HybridACO_DeepForest(
            self.maze, self.start_pos, self.goal_pos,
            df_model=self.models['deep_forest'],
            df_weight=df_weight,
            **clean_kwargs
        )
        path = solver.solve()
        execution_time = time.time() - start_time
        
        result = {
            'algorithm': 'aco_df',
            'path': path,
            'path_length': len(path) if path else 0,
            'execution_time': execution_time,
            'success': path is not None,
            'df_weight': df_weight,
            'iteration_history': solver.iteration_history
        }
        
        self.results['aco_df'] = result
        return result
    
    def solve_all(self, **aco_kwargs) -> Dict:
        """모든 알고리즘으로 해결"""
        all_results = {}
        
        # 기본 ACO
        try:
            all_results['aco'] = self.solve_with_base_aco(**aco_kwargs)
        except Exception as e:
            logger.error(f"기본 ACO 실행 실패: {e}")
            all_results['aco'] = {'error': str(e)}
        
        # CNN 하이브리드
        if 'cnn' in self.models:
            try:
                all_results['aco_cnn'] = self.solve_with_cnn_hybrid(**aco_kwargs)
            except Exception as e:
                logger.error(f"ACO+CNN 실행 실패: {e}")
                all_results['aco_cnn'] = {'error': str(e)}
        
        # Deep Forest 하이브리드
        if 'deep_forest' in self.models:
            try:
                all_results['aco_df'] = self.solve_with_deep_forest_hybrid(**aco_kwargs)
            except Exception as e:
                logger.error(f"ACO+Deep Forest 실행 실패: {e}")
                all_results['aco_df'] = {'error': str(e)}
        
        return all_results
    
    def compare_results(self) -> Dict:
        """결과 비교 분석"""
        if not self.results:
            return {}
        
        comparison = {
            'best_path_length': float('inf'),
            'fastest_execution': float('inf'),
            'success_rate': {},
            'algorithm_ranking': []
        }
        
        for algo_name, result in self.results.items():
            if 'error' in result:
                continue
            
            # 최단 경로
            if result['success'] and result['path_length'] < comparison['best_path_length']:
                comparison['best_path_length'] = result['path_length']
                comparison['best_algorithm'] = algo_name
            
            # 최빠른 실행
            if result['execution_time'] < comparison['fastest_execution']:
                comparison['fastest_execution'] = result['execution_time']
                comparison['fastest_algorithm'] = algo_name
            
            # 성공률
            comparison['success_rate'][algo_name] = 1.0 if result['success'] else 0.0
        
        # 알고리즘 순위 (경로 길이 기준)
        successful_results = [(name, res) for name, res in self.results.items() 
                            if 'error' not in res and res['success']]
        successful_results.sort(key=lambda x: x[1]['path_length'])
        
        comparison['algorithm_ranking'] = [name for name, _ in successful_results]
        
        return comparison


# 사용 예시
if __name__ == "__main__":
    # 테스트용 미로 생성
    maze = np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ])
    
    start_pos = (0, 0)
    goal_pos = (4, 4)
    
    # 하이브리드 솔버 생성
    solver = HybridSolver(maze, start_pos, goal_pos)
    
    # 기본 ACO 테스트
    aco_result = solver.solve_with_base_aco(n_ants=10, n_iterations=20)
    print(f"기본 ACO: 성공={aco_result['success']}, 경로 길이={aco_result['path_length']}")
    
    # 모델이 있다면 하이브리드 테스트
    # solver.load_cnn_model('models/best_cnn_model.pth')
    # solver.load_deep_forest_model('models/deep_forest_model.joblib')
    
    # 결과 비교
    comparison = solver.compare_results()
    print(f"비교 결과: {comparison}")