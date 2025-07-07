"""
Deep Forest 모델 구현 (수정 버전)
자료형 불일치 문제 해결 및 RTX 3060 최적화
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Any
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import time
import gc
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)

class DeepForestModel:
    """Deep Forest 모델 - 자료형 통일 버전"""
    
    def __init__(self, 
                 n_layers: int = 2,
                 n_estimators: int = 50,
                 max_depth: int = 10,
                 min_improvement: float = 0.005,
                 patience: int = 1,
                 random_state: int = 42):
        """
        Deep Forest 초기화
        RTX 3060 (6GB VRAM)에 최적화된 파라미터
        """
        self.n_layers = n_layers
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_improvement = min_improvement
        self.patience = patience
        self.random_state = random_state
        
        # 모델 저장소
        self.forests = []
        self.scalers = []
        self.feature_importance_ = {}
        self.is_trained = False
        
        # 자료형 통일을 위한 설정
        self.dtype = np.float32  # 모든 연산을 float32로 통일
        
        logger.info(f"Deep Forest 초기화: {n_layers}층, {n_estimators}개 트리, dtype={self.dtype}")
    
    def _extract_features_from_maze(self, maze: np.ndarray, 
                                   current_pos: Tuple[int, int],
                                   goal_pos: Tuple[int, int]) -> np.ndarray:
        """
        미로에서 지역적 특징 추출 (자료형 통일)
        """
        maze = maze.astype(self.dtype)  # 자료형 통일
        h, w = maze.shape
        y, x = current_pos
        gy, gx = goal_pos
        
        features = []
        
        # 1. 현재 위치 정보
        features.extend([
            float(y) / h,  # 정규화된 y 좌표
            float(x) / w,  # 정규화된 x 좌표
        ])
        
        # 2. 목표까지의 거리
        manhattan_dist = abs(y - gy) + abs(x - gx)
        euclidean_dist = np.sqrt((y - gy)**2 + (x - gx)**2)
        features.extend([
            float(manhattan_dist) / (h + w),  # 정규화된 맨하탄 거리
            float(euclidean_dist) / np.sqrt(h**2 + w**2),  # 정규화된 유클리드 거리
        ])
        
        # 3. 지역 윈도우 특징 (5x5)
        window_size = 5
        half_window = window_size // 2
        
        local_window = np.ones((window_size, window_size), dtype=self.dtype)  # 기본값은 벽
        
        for dy in range(-half_window, half_window + 1):
            for dx in range(-half_window, half_window + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    local_window[dy + half_window, dx + half_window] = maze[ny, nx]
        
        # 윈도우를 1D로 변환하여 특징으로 사용
        features.extend(local_window.flatten().astype(self.dtype))
        
        # 4. 방향별 접근 가능성
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 상, 하, 좌, 우
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                features.append(float(maze[ny, nx]))
            else:
                features.append(0.0)  # 경계 밖은 벽으로 간주
        
        # 5. 목표 방향 정보
        direction_to_goal = [
            1.0 if gy < y else 0.0,  # 위쪽
            1.0 if gy > y else 0.0,  # 아래쪽
            1.0 if gx < x else 0.0,  # 왼쪽
            1.0 if gx > x else 0.0,  # 오른쪽
        ]
        features.extend(direction_to_goal)
        
        return np.array(features, dtype=self.dtype)
    
    def _prepare_training_data_from_maze(self, maze: np.ndarray, 
                                        start: Tuple[int, int], 
                                        goal: Tuple[int, int],
                                        n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        단일 미로에서 훈련 데이터 생성 (A* 경로 기반)
        """
        from collections import deque
        import heapq
        
        maze = maze.astype(self.dtype)
        
        # A* 알고리즘으로 최적 경로 찾기
        def heuristic(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
        def get_neighbors(pos):
            neighbors = []
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = pos[0] + dy, pos[1] + dx
                if (0 <= ny < maze.shape[0] and 0 <= nx < maze.shape[1] and 
                    maze[ny, nx] == 0):  # 통로
                    neighbors.append((ny, nx))
            return neighbors
        
        # A* 경로 탐색
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                # 경로 재구성
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                break
            
            for neighbor in get_neighbors(current):
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        else:
            logger.warning("A* 알고리즘이 경로를 찾지 못했습니다.")
            return np.array([]), np.array([])
        
        # 경로에서 훈련 데이터 생성
        X, y = [], []
        
        for i in range(len(path) - 1):
            current_pos = path[i]
            next_pos = path[i + 1]
            
            # 특징 추출
            features = self._extract_features_from_maze(maze, current_pos, goal)
            X.append(features)
            
            # 라벨 생성 (방향)
            dy = next_pos[0] - current_pos[0]
            dx = next_pos[1] - current_pos[1]
            
            if dy == -1:
                label = 0  # 위
            elif dy == 1:
                label = 1  # 아래
            elif dx == -1:
                label = 2  # 왼쪽
            else:  # dx == 1
                label = 3  # 오른쪽
            
            y.append(label)
        
        if len(X) == 0:
            return np.array([]), np.array([])
        
        X = np.array(X, dtype=self.dtype)
        y = np.array(y, dtype=np.int32)
        
        logger.info(f"생성된 훈련 데이터: {X.shape[0]}개 샘플, {X.shape[1]}개 특징")
        return X, y
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DeepForestModel':
        """
        Deep Forest 모델 훈련 (자료형 통일)
        """
        # 자료형 통일
        X = X.astype(self.dtype)
        y = y.astype(np.int32)
        
        logger.info(f"Deep Forest 훈련 시작: X{X.shape}, y{y.shape}")
        
        # 초기화
        self.forests = []
        self.scalers = []
        
        current_X = X.copy()
        best_score = 0.0
        patience_counter = 0
        
        for layer in range(self.n_layers):
            logger.info(f"레이어 {layer + 1}/{self.n_layers} 훈련 중...")
            
            # 스케일러 
            scaler = StandardScaler()
            current_X_scaled = scaler.fit_transform(current_X)
            self.scalers.append(scaler)
            
            # 포레스트 생성 (메모리 최적화)
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state + layer,
                n_jobs=2,  # RTX 3060 환경에서 CPU 코어 제한
                max_features='sqrt'
            )
            
            et = ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state + layer + 100,
                n_jobs=2,
                max_features='sqrt'
            )
            
            # 훈련
            rf.fit(current_X_scaled, y)
            et.fit(current_X_scaled, y)
            
            self.forests.append({'rf': rf, 'et': et})
            
            # 성능 평가
            rf_pred = rf.predict_proba(current_X_scaled)
            et_pred = et.predict_proba(current_X_scaled)
            
            # 다음 레이어를 위한 특징 생성
            if layer < self.n_layers - 1:
                combined_features = np.hstack([rf_pred, et_pred])
                current_X = np.hstack([current_X, combined_features.astype(self.dtype)])
            
            # 조기 종료 체크
            current_score = (rf.score(current_X_scaled, y) + et.score(current_X_scaled, y)) / 2
            
            if current_score - best_score < self.min_improvement:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"조기 종료: 레이어 {layer + 1}에서 개선이 없음")
                    break
            else:
                best_score = current_score
                patience_counter = 0
            
            logger.info(f"레이어 {layer + 1} 완료, 정확도: {current_score:.4f}")
            
            # 메모리 정리
            gc.collect()
        
        self.is_trained = True
        logger.info("Deep Forest 훈련 완료")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측 (자료형 통일)"""
        if not self.is_trained:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        X = X.astype(self.dtype)
        current_X = X.copy()
        
        for layer, (forest, scaler) in enumerate(zip(self.forests, self.scalers)):
            current_X_scaled = scaler.transform(current_X)
            
            rf_pred = forest['rf'].predict_proba(current_X_scaled)
            et_pred = forest['et'].predict_proba(current_X_scaled)
            
            if layer < len(self.forests) - 1:
                combined_features = np.hstack([rf_pred, et_pred])
                current_X = np.hstack([current_X, combined_features.astype(self.dtype)])
            else:
                # 마지막 레이어에서는 평균 앙상블
                final_pred = (rf_pred + et_pred) / 2
                return np.argmax(final_pred, axis=1).astype(np.int32)
        
        return np.array([])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """확률 예측"""
        if not self.is_trained:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        X = X.astype(self.dtype)
        current_X = X.copy()
        
        for layer, (forest, scaler) in enumerate(zip(self.forests, self.scalers)):
            current_X_scaled = scaler.transform(current_X)
            
            rf_pred = forest['rf'].predict_proba(current_X_scaled)
            et_pred = forest['et'].predict_proba(current_X_scaled)
            
            if layer < len(self.forests) - 1:
                combined_features = np.hstack([rf_pred, et_pred])
                current_X = np.hstack([current_X, combined_features.astype(self.dtype)])
            else:
                return ((rf_pred + et_pred) / 2).astype(self.dtype)
        
        return np.array([])
    
    def train(self, maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]):
        """단일 미로에서 훈련"""
        X, y = self._prepare_training_data_from_maze(maze, start, goal)
        
        if len(X) == 0:
            raise ValueError("훈련 데이터를 생성할 수 없습니다.")
        
        return self.fit(X, y)
    
    def get_direction_probabilities(self, maze: np.ndarray, 
                                  current_pos: Tuple[int, int], 
                                  goal: Tuple[int, int]) -> np.ndarray:
        """현재 위치에서 각 방향에 대한 확률 반환"""
        if not self.is_trained:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        features = self._extract_features_from_maze(maze, current_pos, goal)
        features = features.reshape(1, -1)
        
        probabilities = self.predict_proba(features)[0]
        return probabilities.astype(self.dtype)
    
    def save(self, filepath: str):
        """모델 저장"""
        save_data = {
            'forests': self.forests,
            'scalers': self.scalers,
            'n_layers': self.n_layers,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'dtype': str(self.dtype),
            'is_trained': self.is_trained
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"모델이 저장되었습니다: {filepath}")
    
    def load(self, filepath: str):
        """모델 로드"""
        save_data = joblib.load(filepath)
        
        self.forests = save_data['forests']
        self.scalers = save_data['scalers']
        self.n_layers = save_data['n_layers']
        self.n_estimators = save_data['n_estimators']
        self.max_depth = save_data['max_depth']
        self.dtype = np.dtype(save_data['dtype'])
        self.is_trained = save_data['is_trained']
        
        logger.info(f"모델이 로드되었습니다: {filepath}")


class DeepForestTrainer:
    """Deep Forest 훈련 관리자"""
    
    def __init__(self, model_config: Dict, save_dir: str = "models"):
        self.config = model_config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.model = DeepForestModel(**model_config)
        
    def train(self, maze_loader, train_ids: List[str], val_ids: List[str]) -> Dict:
        """훈련 실행"""
        logger.info("Deep Forest 훈련 데이터 준비 중...")
        
        # 메모리 절약을 위해 배치 단위로 처리
        batch_size = 100  # RTX 3060에 맞춘 배치 크기
        
        all_X, all_y = [], []
        
        # 훈련 데이터 준비
        for i in range(0, len(train_ids), batch_size):
            batch_ids = train_ids[i:i+batch_size]
            logger.info(f"배치 {i//batch_size + 1}/{(len(train_ids)-1)//batch_size + 1} 처리 중...")
            
            for sample_id in batch_ids:
                try:
                    img, metadata, array = maze_loader.load_sample(sample_id, "train")
                    
                    if array is not None:
                        maze_array = array
                    else:
                        maze_array = maze_loader.convert_image_to_array(img)
                    
                    start = tuple(metadata['start'])
                    goal = tuple(metadata['goal'])
                    
                    # 각 미로에서 훈련 데이터 생성
                    X, y = self.model._prepare_training_data_from_maze(maze_array, start, goal)
                    
                    if len(X) > 0:
                        all_X.append(X)
                        all_y.append(y)
                    
                except Exception as e:
                    logger.warning(f"샘플 {sample_id} 처리 실패: {e}")
                    continue
            
            # 메모리 정리
            gc.collect()
        
        if not all_X:
            raise ValueError("유효한 훈련 데이터가 없습니다.")
        
        # 데이터 결합
        X_train = np.vstack(all_X)
        y_train = np.hstack(all_y)
        
        logger.info(f"총 훈련 데이터: {X_train.shape}")
        
        # 모델 훈련
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # 검증 데이터로 평가
        val_X, val_y = [], []
        for sample_id in val_ids[:50]:  # 메모리 절약
            try:
                img, metadata, array = maze_loader.load_sample(sample_id, "valid")
                
                if array is not None:
                    maze_array = array
                else:
                    maze_array = maze_loader.convert_image_to_array(img)
                
                start = tuple(metadata['start'])
                goal = tuple(metadata['goal'])
                
                X, y = self.model._prepare_training_data_from_maze(maze_array, start, goal)
                
                if len(X) > 0:
                    val_X.append(X)
                    val_y.append(y)
                    
            except Exception as e:
                logger.warning(f"검증 샘플 {sample_id} 처리 실패: {e}")
        
        if val_X:
            X_val = np.vstack(val_X)
            y_val = np.hstack(val_y)
            
            val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
        else:
            val_accuracy = 0.0
        
        # 훈련 정확도
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        # 모델 저장
        model_path = self.save_dir / "deep_forest_model.joblib"
        self.model.save(str(model_path))
        
        result = {
            'training_time': training_time,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'model_path': str(model_path),
            'n_samples': len(X_train)
        }
        
        logger.info(f"훈련 완료 - 훈련 정확도: {train_accuracy:.4f}, 검증 정확도: {val_accuracy:.4f}")
        
        return result
    
    def evaluate(self, maze_loader, test_ids: List[str]) -> Dict:
        """테스트 평가"""
        logger.info("테스트 평가 시작...")
        
        test_X, test_y = [], []
        
        for sample_id in test_ids[:30]:  # 메모리 절약
            try:
                img, metadata, array = maze_loader.load_sample(sample_id, "test")
                
                if array is not None:
                    maze_array = array
                else:
                    maze_array = maze_loader.convert_image_to_array(img)
                
                start = tuple(metadata['start'])
                goal = tuple(metadata['goal'])
                
                X, y = self.model._prepare_training_data_from_maze(maze_array, start, goal)
                
                if len(X) > 0:
                    test_X.append(X)
                    test_y.append(y)
                    
            except Exception as e:
                logger.warning(f"테스트 샘플 {sample_id} 처리 실패: {e}")
        
        if not test_X:
            return {'test_accuracy': 0.0, 'error': '테스트 데이터 없음'}
        
        X_test = np.vstack(test_X)
        y_test = np.hstack(test_y)
        
        test_pred = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        # 분류 리포트
        report = classification_report(y_test, test_pred, 
                                     target_names=['Up', 'Down', 'Left', 'Right'])
        
        logger.info(f"테스트 정확도: {test_accuracy:.4f}")
        
        return {
            'test_accuracy': test_accuracy,
            'classification_report': report,
            'n_test_samples': len(X_test)
        }


# 사용 예시 및 테스트
if __name__ == "__main__":
    import sys
    sys.path.append('..')
    
    # 간단한 테스트 미로
    test_maze = np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ], dtype=np.float32)
    
    start = (1, 1)
    goal = (5, 5)
    
    print("Deep Forest 테스트 시작...")
    
    # 모델 생성 및 훈련
    model = DeepForestModel(n_layers=2, n_estimators=20)
    
    try:
        model.train(test_maze, start, goal)
        print("훈련 완료!")
        
        # 예측 테스트
        probabilities = model.get_direction_probabilities(test_maze, (2, 1), goal)
        print(f"방향 확률: {probabilities}")
        
        directions = ['Up', 'Down', 'Left', 'Right']
        best_direction = directions[np.argmax(probabilities)]
        print(f"추천 방향: {best_direction}")
        
    except Exception as e:
        print(f"테스트 실패: {e}")
        import traceback
        traceback.print_exc()