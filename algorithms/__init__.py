"""
알고리즘 패키지 초기화
자동으로 알고리즘 클래스들을 탐지하고 레지스트리에 등록
"""

import importlib
import inspect
from pathlib import Path
from typing import Dict, Type, Optional
import logging

logger = logging.getLogger(__name__)

# 알고리즘 베이스 클래스 (모든 알고리즘이 상속해야 함)
class BaseAlgorithm:
    """모든 미로 해결 알고리즘의 베이스 클래스"""
    
    def __init__(self, name: str):
        self.name = name
        self.config = {}
    
    def configure(self, config: dict):
        """알고리즘 설정"""
        self.config.update(config)
    
    def solve(self, maze_array, metadata):
        """
        미로 해결 메서드 (서브클래스에서 구현)
        
        Args:
            maze_array: numpy array, 미로 데이터 (0=벽, 1=통로)
            metadata: dict, 미로 메타데이터 (시작점, 목표점 등)
            
        Returns:
            dict: {
                'success': bool,
                'solution_path': list,
                'solution_length': int,
                'execution_time': float,
                'additional_info': dict
            }
        """
        raise NotImplementedError("solve method must be implemented by subclass")
    
    def get_algorithm_info(self):
        """알고리즘 정보 반환"""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'config': self.config
        }

# 알고리즘 레지스트리
_algorithm_registry: Dict[str, Type[BaseAlgorithm]] = {}

def register_algorithm(name: str, algorithm_class: Type[BaseAlgorithm]):
    """알고리즘을 레지스트리에 등록"""
    if not issubclass(algorithm_class, BaseAlgorithm):
        raise ValueError(f"Algorithm class {algorithm_class} must inherit from BaseAlgorithm")
    
    _algorithm_registry[name] = algorithm_class
    logger.info(f"Registered algorithm: {name}")

def get_algorithm(name: str) -> Optional[BaseAlgorithm]:
    """이름으로 알고리즘 인스턴스 반환"""
    if name not in _algorithm_registry:
        logger.error(f"Algorithm '{name}' not found in registry")
        logger.info(f"Available algorithms: {list(_algorithm_registry.keys())}")
        return None
    
    algorithm_class = _algorithm_registry[name]
    return algorithm_class(name)

def get_available_algorithms() -> list:
    """사용 가능한 알고리즘 목록 반환"""
    return list(_algorithm_registry.keys())

def _auto_discover_algorithms():
    """알고리즘 자동 탐지 및 등록"""
    algorithms_dir = Path(__file__).parent
    
    # 알고리즘 파일 패턴 매핑
    algorithm_files = {
        'ACO': 'aco.py',
        'ACO_CNN': 'aco_cnn.py',
        'ACO_DeepForest': 'aco_deepforest.py',
        'DQN': 'dqn.py',
        'DQN_DeepForest': 'dqn_deepforest.py',
        'PPO': 'ppo.py',
        'A_STAR': 'astar.py'
    }
    
    for algo_name, filename in algorithm_files.items():
        filepath = algorithms_dir / filename
        
        if filepath.exists():
            try:
                # 모듈 동적 임포트
                module_name = f"algorithms.{filename[:-3]}"  # .py 제거
                
                logger.debug(f"Importing module: {module_name}")
                module = importlib.import_module(module_name)
                
                # 알고리즘 클래스 찾기
                algorithm_class = None
                
                # 클래스명 패턴들 시도
                class_name_patterns = [
                    f"{algo_name}Algorithm",  # 예: ACOAlgorithm
                    f"{algo_name.replace('_', '')}Algorithm",  # 예: ACODeepForestAlgorithm
                    f"{algo_name}Solver",
                    algo_name
                ]
                
                for class_name in class_name_patterns:
                    if hasattr(module, class_name):
                        cls = getattr(module, class_name)
                        if (inspect.isclass(cls) and 
                            issubclass(cls, BaseAlgorithm) and 
                            cls != BaseAlgorithm):
                            algorithm_class = cls
                            break
                
                # 모든 클래스 검사 (위의 패턴으로 찾지 못한 경우)
                if algorithm_class is None:
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, BaseAlgorithm) and 
                            obj != BaseAlgorithm and
                            obj.__module__ == module.__name__):
                            algorithm_class = obj
                            break
                
                if algorithm_class:
                    register_algorithm(algo_name, algorithm_class)
                else:
                    logger.warning(f"No valid algorithm class found in {filename}")
                    
            except Exception as e:
                logger.warning(f"Failed to import {filename}: {e}")
        else:
            logger.debug(f"Algorithm file not found: {filepath}")

def _manual_registrations():
    """수동 알고리즘 등록 (자동 탐지 실패 시 fallback)"""
    
    # ACO_DeepForest 수동 등록
    try:
        from .aco_deepforest import ACODeepForestAlgorithm
        register_algorithm("ACO_DeepForest", ACODeepForestAlgorithm)
    except ImportError as e:
        logger.warning(f"Failed to manually register ACO_DeepForest: {e}")
    
    # PPO 수동 등록
    try:
        from .ppo import PPOAlgorithm
        register_algorithm("PPO", PPOAlgorithm)
    except ImportError as e:
        logger.warning(f"Failed to manually register PPO: {e}")
    
    # A* 수동 등록
    try:
        from .astar import AStarAlgorithm
        register_algorithm("A_STAR", AStarAlgorithm)
    except ImportError as e:
        logger.warning(f"Failed to manually register A_STAR: {e}")
    
    # DQN 등록 (특별 처리)
    try:
        from .dqn_solver import DQNSolver, create_dqn_solver
        
        # DQN용 래퍼 클래스 생성
        class DQNAlgorithm(BaseAlgorithm):
            def __init__(self, name: str = "DQN"):
                super().__init__(name)
                self.solver = None
            
            def configure(self, config: dict):
                super().configure(config)
                self.solver = create_dqn_solver(
                    learning_rate=config.get('learning_rate', 0.001),
                    epsilon=config.get('epsilon', 0.1),
                    batch_size=config.get('batch_size', 32),
                    memory_size=config.get('memory_size', 10000)
                )
            
            def solve(self, maze_array, metadata):
                if self.solver is None:
                    self.configure({})
                
                start = tuple(metadata.get('entrance', (0, 0)))
                goal = tuple(metadata.get('exit', (maze_array.shape[0]-1, maze_array.shape[1]-1)))
                
                result = self.solver.solve(maze_array, start, goal)
                
                return {
                    'success': result.get('success', False),
                    'solution_path': result.get('path', []),
                    'solution_length': len(result.get('path', [])),
                    'execution_time': result.get('execution_time', 0),
                    'additional_info': {
                        'training_episodes': result.get('training_episodes', 0),
                        'failure_reason': result.get('failure_reason', '')
                    }
                }
        
        register_algorithm("DQN", DQNAlgorithm)
        
    except ImportError as e:
        logger.warning(f"Failed to manually register DQN: {e}")

def _validate_registrations():
    """등록된 알고리즘들의 유효성 검사"""
    logger.info("Validating algorithm registrations...")
    
    for name, algorithm_class in _algorithm_registry.items():
        try:
            # 인스턴스 생성 테스트
            instance = algorithm_class(name)
            
            # 필수 메서드 존재 확인
            if not hasattr(instance, 'solve'):
                logger.error(f"Algorithm {name} missing solve method")
                continue
            
            if not hasattr(instance, 'configure'):
                logger.error(f"Algorithm {name} missing configure method")
                continue
            
            logger.debug(f"Algorithm {name} validation passed")
            
        except Exception as e:
            logger.error(f"Algorithm {name} validation failed: {e}")

# 패키지 초기화 시 자동 실행
try:
    logger.info("Initializing algorithms package...")
    
    # 자동 탐지 시도
    _auto_discover_algorithms()
    
    # 수동 등록 (fallback)
    _manual_registrations()
    
    # 등록 결과 검증
    _validate_registrations()
    
    registered_count = len(_algorithm_registry)
    logger.info(f"Algorithm package initialized: {registered_count} algorithms registered")
    logger.info(f"Available algorithms: {list(_algorithm_registry.keys())}")
    
except Exception as e:
    logger.error(f"Failed to initialize algorithms package: {e}")

# 공개 API
__all__ = [
    'BaseAlgorithm',
    'register_algorithm', 
    'get_algorithm',
    'get_available_algorithms'
]