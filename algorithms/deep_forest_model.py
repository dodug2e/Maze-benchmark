"""
Deep Forest 모델 구현 - 미로 경로 예측용
메모리 효율적인 구현 (RTX 3060 최적화)
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging
from typing import List, Tuple, Dict, Optional
import time
import os

logger = logging.getLogger(__name__)

class DeepForestLayer:
    """Deep Forest의 단일 레이어"""
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: int = None,
                 random_state: int = 42,
                 n_jobs: int = -1):
        
        # Random Forest와 Extra Trees를 함께 사용
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
            oob_score=True
        )
        
        self.et = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state + 1,
            n_jobs=n_jobs,
            bootstrap=True,
            oob_score=True
        )
        
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """레이어 학습"""
        logger.info(f"Deep Forest 레이어 학습 시작 - 입력 크기: {X.shape}")
        
        start_time = time.time()
        
        # Random Forest 학습
        self.rf.fit(X, y)
        rf_time = time.time() - start_time
        
        # Extra Trees 학습
        et_start = time.time()
        self.et.fit(X, y)
        et_time = time.time() - et_start
        
        self.is_fitted = True
        
        logger.info(f"RF 학습 완료: {rf_time:.2f}초, OOB 스코어: {self.rf.oob_score_:.4f}")
        logger.info(f"ET 학습 완료: {et_time:.2f}초, OOB 스코어: {self.et.oob_score_:.4f}")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """확률 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        # Random Forest와 Extra Trees의 확률 예측을 결합
        rf_proba = self.rf.predict_proba(X)
        et_proba = self.et.predict_proba(X)
        
        # 두 예측을 연결 (concatenate)
        combined_proba = np.hstack([rf_proba, et_proba])
        
        return combined_proba
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """클래스 예측"""
        probas = self.predict_proba(X)
        
        # RF와 ET의 예측을 평균내어 최종 예측
        rf_proba = probas[:, :self.rf.n_classes_]
        et_proba = probas[:, self.rf.n_classes_:]
        
        avg_proba = (rf_proba + et_proba) / 2
        return np.argmax(avg_proba, axis=1)


class MazeDeepForest:
    """미로 경로 예측을 위한 Deep Forest 모델"""
    
    def __init__(self,
                 n_layers: int = 3,
                 n_estimators: int = 100,
                 max_depth: int = None,
                 min_improvement: float = 0.001,
                 patience: int = 2,
                 random_state: int = 42):
        
        self.n_layers = n_layers
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_improvement = min_improvement
        self.patience = patience
        self.random_state = random_state
        
        self.layers: List[DeepForestLayer] = []
        self.best_layer_count = 0
        self.training_history = []
        
    def _extract_features(self, maze: np.ndarray, 
                         current_pos: Tuple[int, int],
                         adaptive_window: bool = True) -> np.ndarray:
        """미로에서 특성 추출 (가변 크기 미로 지원)"""
        h, w = maze.shape
        y, x = current_pos
        
        # 미로 크기에 따른 적응적 윈도우 크기 결정
        if adaptive_window:
            # 미로가 클수록 더 큰 윈도우 사용
            if min(h, w) <= 60:
                window_size = 5
            elif min(h, w) <= 120:
                window_size = 7
            else:
                window_size = 9
        else:
            window_size = 5
        
        features = []
        
        # 1. 현재 위치 주변 패턴 (adaptive window_size x window_size)
        half_window = window_size // 2
        local_pattern = []
        
        for dy in range(-half_window, half_window + 1):
            for dx in range(-half_window, half_window + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    local_pattern.append(maze[ny, nx])
                else:
                    local_pattern.append(0)  # 경계 외부는 벽으로 처리
        
        features.extend(local_pattern)
        
        # 2. 방향별 거리 특성 (미로 크기에 비례한 최대 탐색 거리)
        max_search_distance = min(20, min(h, w) // 4)  # 미로 크기의 1/4 또는 최대 20
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 상하좌우
        
        for dy, dx in directions:
            distance = 0
            ny, nx = y + dy, x + dx
            
            # 벽이나 경계까지의 거리 계산
            while (0 <= ny < h and 0 <= nx < w and 
                   maze[ny, nx] == 1 and distance < max_search_distance):
                distance += 1
                ny, nx = ny + dy, nx + dx
            
            # 정규화된 거리 (0~1 범위)
            normalized_distance = distance / max_search_distance
            features.append(normalized_distance)
        
        # 3. 각 방향의 경로 밀도 (적응적 탐색 범위)
        density_range = min(8, min(h, w) // 8)  # 미로 크기에 비례
        
        for dy, dx in directions:
            path_count = 0
            total_count = 0
            
            for dist in range(1, density_range + 1):
                ny, nx = y + dy * dist, x + dx * dist
                if 0 <= ny < h and 0 <= nx < w:
                    total_count += 1
                    if maze[ny, nx] == 1:
                        path_count += 1
            
            density = path_count / total_count if total_count > 0 else 0
            features.append(density)
        
        # 4. 전역 특성 (정규화된 좌표와 미로 특성)
        features.extend([
            y / h,  # 정규화된 y 좌표
            x / w,  # 정규화된 x 좌표
            np.sum(maze) / (h * w),  # 전체 경로 밀도
            h / 201.0,  # 정규화된 미로 높이 (최대 201 기준)
            w / 201.0,  # 정규화된 미로 너비
            min(h, w) / 201.0,  # 정규화된 최소 크기
        ])
        
        # 5. 목표 방향 특성 (목표가 있는 경우)
        if hasattr(self, 'goal_pos') and self.goal_pos:
            goal_y, goal_x = self.goal_pos
            
            # 목표까지의 정규화된 거리와 방향
            goal_distance = ((goal_y - y) ** 2 + (goal_x - x) ** 2) ** 0.5
            normalized_goal_distance = goal_distance / (h + w)  # 미로 둘레 대비
            
            # 목표 방향 벡터 (정규화)
            if goal_distance > 0:
                goal_direction_y = (goal_y - y) / goal_distance
                goal_direction_x = (goal_x - x) / goal_distance
            else:
                goal_direction_y = goal_direction_x = 0
            
            features.extend([
                normalized_goal_distance,
                goal_direction_y,
                goal_direction_x
            ])
        else:
            # 목표 정보가 없으면 0으로 패딩
            features.extend([0, 0, 0])
        
        # 6. 지역적 연결성 특성
        # 현재 위치에서 접근 가능한 방향의 수
        accessible_directions = 0
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and maze[ny, nx] == 1:
                accessible_directions += 1
        
        features.append(accessible_directions / 4.0)  # 정규화 (최대 4방향)
        
        return np.array(features, dtype=np.float32)
    
    def prepare_training_data(self, maze_loader, sample_ids: list, 
                            subset: str = "train") -> Tuple[np.ndarray, np.ndarray]:
        """학습 데이터 준비 (가변 크기 미로 지원)"""
        logger.info(f"Deep Forest 학습 데이터 준비: {len(sample_ids)}개 샘플")
        
        all_features = []
        all_labels = []
        
        # 미로 크기 분포 확인
        maze_sizes = []
        
        for i, sample_id in enumerate(sample_ids):
            if i % 100 == 0:
                logger.info(f"처리 중: {i}/{len(sample_ids)}")
            
            try:
                # 미로 데이터 로드
                img, metadata, array = maze_loader.load_sample(sample_id, subset)
                
                if array is not None:
                    maze_array = array
                else:
                    maze_array = maze_loader.convert_image_to_array(img)
                
                maze_sizes.append(maze_array.shape)
                
                # 시작점과 끝점
                start = tuple(metadata.get('entrance', (1, 1)))
                goal = tuple(metadata.get('exit', (maze_array.shape[0]-2, maze_array.shape[1]-2)))
                
                # 목표 위치 저장 (특성 추출에서 사용)
                self.goal_pos = goal
                
                # A* 알고리즘으로 최적 경로 찾기
                optimal_path = self._find_optimal_path(maze_array, start, goal)
                
                if optimal_path and len(optimal_path) > 1:
                    # 경로의 각 위치에서 특성 추출
                    for j in range(len(optimal_path) - 1):
                        current_pos = optimal_path[j]
                        next_pos = optimal_path[j + 1]
                        
                        # 방향 라벨 계산
                        direction = self._calculate_direction(current_pos, next_pos)
                        
                        # 적응적 특성 추출
                        features = self._extract_features(maze_array, current_pos, adaptive_window=True)
                        
                        all_features.append(features)
                        all_labels.append(direction)
                
            except Exception as e:
                logger.warning(f"샘플 {sample_id} 처리 실패: {e}")
                continue
        
        # 미로 크기 분포 로깅
        unique_sizes = set(maze_sizes)
        logger.info(f"미로 크기 분포: {len(unique_sizes)}가지 크기")
        for size in sorted(unique_sizes):
            count = maze_sizes.count(size)
            logger.info(f"  {size[0]}x{size[1]}: {count}개")
        
        X = np.array(all_features)
        y = np.array(all_labels)
        
        logger.info(f"학습 데이터 준비 완료: {X.shape}, 라벨 분포: {np.bincount(y)}")
        logger.info(f"특성 개수: {X.shape[1]}개 (적응적 윈도우 크기 사용)")
        
        return X, y
    
    def _find_optimal_path(self, maze: np.ndarray, 
                          start: Tuple[int, int], 
                          goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """A* 알고리즘으로 최적 경로 찾기"""
        from heapq import heappush, heappop
        
        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        def get_neighbors(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
            y, x = pos
            neighbors = []
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if (0 <= ny < maze.shape[0] and 0 <= nx < maze.shape[1] and 
                    maze[ny, nx] == 1):
                    neighbors.append((ny, nx))
            return neighbors
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = heappop(open_set)[1]
            
            if current == goal:
                # 경로 재구성
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor in get_neighbors(current):
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))
        
        return None
    
    def _calculate_direction(self, current: Tuple[int, int], 
                           next_pos: Tuple[int, int]) -> int:
        """두 위치 간의 방향 계산"""
        dy = next_pos[0] - current[0]
        dx = next_pos[1] - current[1]
        
        if dy == -1:
            return 0  # 위
        elif dy == 1:
            return 1  # 아래
        elif dx == -1:
            return 2  # 왼쪽
        elif dx == 1:
            return 3  # 오른쪽
        else:
            return 0  # 기본값
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None):
        """Deep Forest 학습"""
        logger.info(f"Deep Forest 학습 시작: {X.shape[0]}개 샘플, {X.shape[1]}개 특성")
        
        self.layers = []
        current_X = X.copy()
        best_val_score = 0
        patience_counter = 0
        
        for layer_idx in range(self.n_layers):
            logger.info(f"\n=== 레이어 {layer_idx + 1} 학습 ===")
            
            # 새 레이어 생성
            layer = DeepForestLayer(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state + layer_idx
            )
            
            # 레이어 학습
            layer.fit(current_X, y)
            self.layers.append(layer)
            
            # 검증 성능 평가
            if X_val is not None and y_val is not None:
                val_X = self._transform_layers(X_val, layer_idx + 1)
                val_pred = self._predict_with_layers(val_X, layer_idx + 1)
                val_score = accuracy_score(y_val, val_pred)
                
                logger.info(f"레이어 {layer_idx + 1} 검증 정확도: {val_score:.4f}")
                
                # 조기 종료 검사
                if val_score > best_val_score + self.min_improvement:
                    best_val_score = val_score
                    self.best_layer_count = layer_idx + 1
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.patience:
                    logger.info(f"조기 종료: {self.patience}번 연속 개선 없음")
                    break
            else:
                self.best_layer_count = layer_idx + 1
            
            # 다음 레이어를 위해 입력 데이터 변환
            if layer_idx < self.n_layers - 1:
                layer_output = layer.predict_proba(current_X)
                current_X = np.hstack([current_X, layer_output])
                
                self.training_history.append({
                    'layer': layer_idx + 1,
                    'input_features': current_X.shape[1],
                    'validation_score': val_score if X_val is not None else None
                })
        
        logger.info(f"Deep Forest 학습 완료: {self.best_layer_count}개 레이어 사용")
        
        return self
    
    def _transform_layers(self, X: np.ndarray, n_layers: int) -> np.ndarray:
        """지정된 수의 레이어까지 데이터 변환"""
        current_X = X.copy()
        
        for i in range(min(n_layers, len(self.layers))):
            layer_output = self.layers[i].predict_proba(current_X)
            current_X = np.hstack([current_X, layer_output])
        
        return current_X
    
    def _predict_with_layers(self, X: np.ndarray, n_layers: int) -> np.ndarray:
        """지정된 수의 레이어로 예측"""
        if n_layers == 0 or not self.layers:
            return np.zeros(X.shape[0])
        
        # 마지막 레이어의 예측 사용
        layer_idx = min(n_layers - 1, len(self.layers) - 1)
        transformed_X = self._transform_layers(X, layer_idx)
        
        return self.layers[layer_idx].predict(transformed_X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측"""
        return self._predict_with_layers(X, self.best_layer_count)
    
    def predict_direction(self, maze: np.ndarray, 
                         current_pos: Tuple[int, int]) -> int:
        """현재 위치에서 최적 방향 예측"""
        features = self._extract_features(maze, current_pos)
        features = features.reshape(1, -1)  # 단일 샘플을 위한 reshape
        
        direction = self.predict(features)[0]
        return direction
    
    def save_model(self, filepath: str):
        """모델 저장"""
        model_data = {
            'layers': self.layers,
            'best_layer_count': self.best_layer_count,
            'training_history': self.training_history,
            'n_layers': self.n_layers,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Deep Forest 모델이 저장되었습니다: {filepath}")
    
    def load_model(self, filepath: str):
        """모델 로드"""
        model_data = joblib.load(filepath)
        
        self.layers = model_data['layers']
        self.best_layer_count = model_data['best_layer_count']
        self.training_history = model_data['training_history']
        self.n_layers = model_data['n_layers']
        self.n_estimators = model_data['n_estimators']
        self.max_depth = model_data['max_depth']
        self.random_state = model_data['random_state']
        
        logger.info(f"Deep Forest 모델이 로드되었습니다: {filepath}")
        logger.info(f"레이어 수: {self.best_layer_count}, 학습 히스토리: {len(self.training_history)}개")
    
    def get_feature_importance(self) -> Dict:
        """특성 중요도 반환 (가변 크기 미로 대응)"""
        if not self.layers:
            return {}
        
        # 첫 번째 레이어의 특성 중요도만 사용 (원본 특성에 대한)
        first_layer = self.layers[0]
        
        rf_importance = first_layer.rf.feature_importances_
        et_importance = first_layer.et.feature_importances_
        
        # 평균 중요도 계산
        avg_importance = (rf_importance + et_importance) / 2
        
        # 특성 이름 생성 (적응적 윈도우 크기 고려)
        feature_names = []
        
        # 주변 패턴 특성 (적응적 크기 - 추정치)
        pattern_count = len([name for name in feature_names if name.startswith('pattern_')])
        if pattern_count == 0:  # 첫 번째 호출
            # 기본적으로 7x7 패턴 가정 (가장 일반적인 크기)
            for i in range(49):  # 7x7 = 49
                feature_names.append(f"pattern_{i}")
        
        # 방향별 거리 특성 (4개)
        directions = ['up', 'down', 'left', 'right']
        for direction in directions:
            feature_names.append(f"distance_{direction}")
        
        # 방향별 밀도 특성 (4개)
        for direction in directions:
            feature_names.append(f"density_{direction}")
        
        # 전역 특성 (6개)
        feature_names.extend([
            'norm_y', 'norm_x', 'path_density', 
            'norm_height', 'norm_width', 'norm_min_size'
        ])
        
        # 목표 관련 특성 (3개)
        feature_names.extend(['goal_distance', 'goal_dir_y', 'goal_dir_x'])
        
        # 연결성 특성 (1개)
        feature_names.append('accessibility')
        
        # 실제 특성 개수에 맞춰 조정
        actual_feature_count = len(avg_importance)
        if len(feature_names) > actual_feature_count:
            feature_names = feature_names[:actual_feature_count]
        elif len(feature_names) < actual_feature_count:
            # 부족한 경우 일반적인 이름으로 채움
            for i in range(len(feature_names), actual_feature_count):
                feature_names.append(f"feature_{i}")
        
        importance_dict = dict(zip(feature_names, avg_importance))
        
        # 중요도 순으로 정렬
        sorted_importance = dict(sorted(importance_dict.items(), 
                                      key=lambda x: x[1], 
                                      reverse=True))
        
        return sorted_importance
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        info = {
            'total_layers': len(self.layers),
            'best_layer_count': self.best_layer_count,
            'n_estimators_per_layer': self.n_estimators,
            'max_depth': self.max_depth,
            'training_history': self.training_history
        }
        
        if self.layers:
            # 첫 번째 레이어의 OOB 스코어
            first_layer = self.layers[0]
            info['first_layer_rf_oob'] = first_layer.rf.oob_score_
            info['first_layer_et_oob'] = first_layer.et.oob_score_
            
            # 총 트리 개수
            total_trees = len(self.layers) * self.n_estimators * 2  # RF + ET
            info['total_trees'] = total_trees
        
        return info


class DeepForestTrainer:
    """Deep Forest 학습 관리 클래스"""
    
    def __init__(self, 
                 model_config: Dict = None,
                 save_dir: str = "models"):
        
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 기본 설정
        default_config = {
            'n_layers': 3,
            'n_estimators': 100,
            'max_depth': None,
            'min_improvement': 0.001,
            'patience': 2,
            'random_state': 42
        }
        
        if model_config:
            default_config.update(model_config)
        
        self.model_config = default_config
        self.model = None
    
    def train(self, maze_loader, train_ids: list, val_ids: list):
        """전체 학습 파이프라인"""
        logger.info("Deep Forest 학습 시작")
        
        # 모델 생성
        self.model = MazeDeepForest(**self.model_config)
        
        # 학습 데이터 준비
        logger.info("학습 데이터 준비 중...")
        X_train, y_train = self.model.prepare_training_data(
            maze_loader, train_ids, "train"
        )
        
        logger.info("검증 데이터 준비 중...")
        X_val, y_val = self.model.prepare_training_data(
            maze_loader, val_ids, "valid"
        )
        
        # 모델 학습
        start_time = time.time()
        self.model.fit(X_train, y_train, X_val, y_val)
        training_time = time.time() - start_time
        
        # 성능 평가
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        
        logger.info(f"\n=== Deep Forest 학습 완료 ===")
        logger.info(f"학습 시간: {training_time:.2f}초")
        logger.info(f"최종 레이어 수: {self.model.best_layer_count}")
        logger.info(f"학습 정확도: {train_acc:.4f}")
        logger.info(f"검증 정확도: {val_acc:.4f}")
        
        # 특성 중요도 출력
        importance = self.model.get_feature_importance()
        logger.info("\n상위 10개 중요 특성:")
        for i, (feature, imp) in enumerate(list(importance.items())[:10]):
            logger.info(f"  {i+1}. {feature}: {imp:.4f}")
        
        # 모델 저장
        model_path = os.path.join(self.save_dir, "deep_forest_model.joblib")
        self.model.save_model(model_path)
        
        return {
            'training_time': training_time,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'best_layer_count': self.model.best_layer_count,
            'model_path': model_path
        }
    
    def evaluate(self, maze_loader, test_ids: list):
        """테스트 데이터로 성능 평가"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        logger.info("테스트 데이터 평가 중...")
        X_test, y_test = self.model.prepare_training_data(
            maze_loader, test_ids, "test"
        )
        
        test_pred = self.model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        
        # 분류 리포트
        report = classification_report(y_test, test_pred, 
                                     target_names=['Up', 'Down', 'Left', 'Right'])
        
        logger.info(f"테스트 정확도: {test_acc:.4f}")
        logger.info(f"\n분류 리포트:\n{report}")
        
        return {
            'test_accuracy': test_acc,
            'classification_report': report,
            'predictions': test_pred,
            'true_labels': y_test
        }


# 사용 예시
if __name__ == "__main__":
    import os
    from utils.maze_io import get_loader
    
    # 미로 데이터 로더
    maze_loader = get_loader()
    
    # 샘플 ID 가져오기
    train_ids = maze_loader.get_sample_ids("train")[:1000]  # 메모리 절약을 위해 1000개만
    val_ids = maze_loader.get_sample_ids("valid")[:500]
    test_ids = maze_loader.get_sample_ids("test")[:200]
    
    # Deep Forest 설정
    config = {
        'n_layers': 2,  # RTX 3060을 위해 레이어 수 축소
        'n_estimators': 50,  # 메모리 절약
        'max_depth': 10,
        'min_improvement': 0.005,
        'patience': 1
    }
    
    # 학습
    trainer = DeepForestTrainer(config)
    
    try:
        train_result = trainer.train(maze_loader, train_ids, val_ids)
        print(f"학습 완료: 검증 정확도 {train_result['val_accuracy']:.4f}")
        
        # 테스트 평가
        test_result = trainer.evaluate(maze_loader, test_ids)
        print(f"테스트 정확도: {test_result['test_accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {e}")
        raise