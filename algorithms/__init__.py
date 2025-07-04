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
        'A_STAR': 'a_star.py'
    }
    
    for algo_name, filename in algorithm_files.items():
        filepath = algorithms_dir / filename
        
        if filepath.exists():
            try:
                # 모듈 동적 임포트
                module_name = f"algorithms.{filename[:-3]}"  # .py 제거
                module = importlib.import_module(module_name)
                
                # 알고리즘 클래스 찾기
                algorithm_class = None
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseAlgorithm) and 
                        obj != BaseAlgorithm):
                        algorithm_class = obj
                        break
                
                if algorithm_class:
                    register_algorithm(algo_name, algorithm_class)
                else:
                    logger.warning(f"No algorithm class found in {filename}")
                    
            except Exception as e:
                logger.error(f"Failed to import {filename}: {e}")
        else:
            logger.warning(f"Algorithm file not found: {filepath}")

# 패키지 임포트 시 자동 탐지 실행
_auto_discover_algorithms()

# 편의 함수들
def list_algorithms():
    """사용 가능한 알고리즘 목록 출력"""
    algorithms = get_available_algorithms()
    print(f"Available algorithms ({len(algorithms)}):")
    for i, algo in enumerate(algorithms, 1):
        print(f"  {i}. {algo}")
    return algorithms

def test_algorithm(name: str, maze_array=None, metadata=None):
    """알고리즘 테스트"""
    algorithm = get_algorithm(name)
    if not algorithm:
        return None
    
    # 테스트용 더미 데이터
    if maze_array is None:
        import numpy as np
        maze_array = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ])
    
    if metadata is None:
        metadata = {
            'entrance': (1, 1),
            'exit': (3, 3),
            'size': maze_array.shape
        }
    
    try:
        result = algorithm.solve(maze_array, metadata)
        print(f"Test result for {name}: {result}")
        return result
    except Exception as e:
        print(f"Test failed for {name}: {e}")
        return None

# 디버깅을 위한 레지스트리 상태 출력
if __name__ == "__main__":
    print("Algorithm Registry Debug Info:")
    print(f"Registry contents: {list(_algorithm_registry.keys())}")
    
    # 각 알고리즘 테스트
    for algo_name in get_available_algorithms():
        print(f"\nTesting {algo_name}...")
        test_algorithm(algo_name)