"""
재현성을 위한 시드 관리 유틸리티
모든 라이브러리의 랜덤 시드를 통합 관리
"""

import os
import random
import numpy as np
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# 선택적 라이브러리 import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# 기본 시드 값
DEFAULT_SEED = 42


class SeedManager:
    """시드 관리자 클래스"""
    
    def __init__(self, seed: int = DEFAULT_SEED):
        self.seed = seed
        self.is_set = False
        self.library_versions = {}
        
    def set_seed(self, seed: Optional[int] = None) -> Dict[str, str]:
        """
        모든 라이브러리의 시드를 설정
        
        Args:
            seed: 설정할 시드 값 (None이면 기본값 사용)
            
        Returns:
            Dict: 설정된 라이브러리들의 상태
        """
        if seed is not None:
            self.seed = seed
            
        results = {}
        
        # Python 기본 random 모듈
        random.seed(self.seed)
        results['python_random'] = f"seed={self.seed}"
        
        # NumPy
        np.random.seed(self.seed)
        results['numpy'] = f"seed={self.seed}, version={np.__version__}"
        
        # PyTorch
        if TORCH_AVAILABLE:
            torch.manual_seed(self.seed)
            results['torch'] = f"seed={self.seed}, version={torch.__version__}"
            
            # CUDA 시드 설정 (GPU 사용 시)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
                
                # 결정적 동작을 위한 설정
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
                results['torch_cuda'] = f"seed={self.seed}, devices={torch.cuda.device_count()}"
        
        # TensorFlow
        if TF_AVAILABLE:
            tf.random.set_seed(self.seed)
            results['tensorflow'] = f"seed={self.seed}, version={tf.__version__}"
        
        # 환경 변수 설정 (추가적인 결정적 동작)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        results['python_hash'] = f"seed={self.seed}"
        
        # CUDA 관련 환경 변수
        if TORCH_AVAILABLE and torch.cuda.is_available():
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            results['cuda_blocking'] = "enabled"
        
        self.is_set = True
        self.library_versions = results
        
        logger.info(f"시드 설정 완료: {self.seed}")
        for lib, status in results.items():
            logger.debug(f"  {lib}: {status}")
            
        return results
    
    def get_current_seed(self) -> int:
        """현재 시드 값 반환"""
        return self.seed
    
    def is_seed_set(self) -> bool:
        """시드가 설정되었는지 확인"""
        return self.is_set
    
    def get_library_status(self) -> Dict[str, str]:
        """라이브러리별 시드 설정 상태 반환"""
        return self.library_versions.copy()
    
    def reset_seed(self, new_seed: int):
        """새로운 시드로 재설정"""
        logger.info(f"시드 재설정: {self.seed} → {new_seed}")
        self.set_seed(new_seed)
    
    def create_reproducible_state(self) -> Dict:
        """재현 가능한 상태 정보 생성"""
        state = {
            'seed': self.seed,
            'python_random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'library_versions': self.library_versions,
            'environment': {
                'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED', ''),
                'CUDA_LAUNCH_BLOCKING': os.environ.get('CUDA_LAUNCH_BLOCKING', ''),
            }
        }
        
        # PyTorch 상태 추가
        if TORCH_AVAILABLE:
            state['torch_random_state'] = torch.get_rng_state()
            if torch.cuda.is_available():
                state['torch_cuda_random_state'] = torch.cuda.get_rng_state()
        
        # TensorFlow 상태 추가
        if TF_AVAILABLE:
            # TensorFlow는 상태 추출이 복잡하므로 시드 값만 저장
            state['tensorflow_seed'] = self.seed
        
        return state
    
    def validate_reproducibility(self) -> Dict[str, bool]:
        """재현성 검증"""
        validation_results = {}
        
        # Python random 검증
        old_state = random.getstate()
        random.seed(self.seed)
        test_val1 = random.random()
        random.seed(self.seed)
        test_val2 = random.random()
        validation_results['python_random'] = (test_val1 == test_val2)
        random.setstate(old_state)
        
        # NumPy 검증
        old_state = np.random.get_state()
        np.random.seed(self.seed)
        test_arr1 = np.random.random(5)
        np.random.seed(self.seed)
        test_arr2 = np.random.random(5)
        validation_results['numpy'] = np.array_equal(test_arr1, test_arr2)
        np.random.set_state(old_state)
        
        # PyTorch 검증
        if TORCH_AVAILABLE:
            old_state = torch.get_rng_state()
            torch.manual_seed(self.seed)
            test_tensor1 = torch.randn(3, 3)
            torch.manual_seed(self.seed)
            test_tensor2 = torch.randn(3, 3)
            validation_results['torch'] = torch.equal(test_tensor1, test_tensor2)
            torch.set_rng_state(old_state)
        
        return validation_results


# 전역 시드 매니저 인스턴스
_global_seed_manager: Optional[SeedManager] = None


def get_seed_manager() -> SeedManager:
    """전역 시드 매니저 인스턴스 반환"""
    global _global_seed_manager
    if _global_seed_manager is None:
        _global_seed_manager = SeedManager()
    return _global_seed_manager


def set_seed(seed: int = DEFAULT_SEED) -> Dict[str, str]:
    """
    모든 라이브러리의 시드를 설정하는 편의 함수
    
    Args:
        seed: 설정할 시드 값
        
    Returns:
        Dict: 설정된 라이브러리들의 상태
    """
    manager = get_seed_manager()
    return manager.set_seed(seed)


def get_current_seed() -> int:
    """현재 시드 값 반환"""
    manager = get_seed_manager()
    return manager.get_current_seed()


def ensure_reproducibility(seed: int = DEFAULT_SEED) -> bool:
    """
    재현성을 보장하는 함수
    
    Args:
        seed: 설정할 시드 값
        
    Returns:
        bool: 재현성 검증 통과 여부
    """
    manager = get_seed_manager()
    
    # 시드 설정
    manager.set_seed(seed)
    
    # 재현성 검증
    validation_results = manager.validate_reproducibility()
    
    # 모든 검증이 통과되었는지 확인
    all_passed = all(validation_results.values())
    
    if all_passed:
        logger.info("재현성 검증 통과")
    else:
        logger.warning("재현성 검증 실패:")
        for lib, passed in validation_results.items():
            if not passed:
                logger.warning(f"  {lib}: 실패")
    
    return all_passed


def create_experiment_config(seed: int = DEFAULT_SEED, 
                           experiment_name: str = "maze_benchmark") -> Dict:
    """
    실험 설정 생성
    
    Args:
        seed: 실험 시드
        experiment_name: 실험 이름
        
    Returns:
        Dict: 실험 설정 정보
    """
    manager = get_seed_manager()
    manager.set_seed(seed)
    
    config = {
        'experiment_name': experiment_name,
        'seed': seed,
        'reproducible_state': manager.create_reproducible_state(),
        'library_availability': {
            'torch': TORCH_AVAILABLE,
            'tensorflow': TF_AVAILABLE,
            'cuda': TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False
        },
        'hardware_info': {}
    }
    
    # 하드웨어 정보 추가
    if TORCH_AVAILABLE and torch.cuda.is_available():
        config['hardware_info']['cuda_device_count'] = torch.cuda.device_count()
        config['hardware_info']['cuda_device_name'] = torch.cuda.get_device_name(0)
        config['hardware_info']['cuda_memory_total'] = torch.cuda.get_device_properties(0).total_memory
    
    return config


def save_experiment_config(config: Dict, filepath: str):
    """실험 설정을 파일로 저장"""
    import json
    
    # NumPy 배열과 텐서는 JSON 직렬화할 수 없으므로 제외
    serializable_config = {}
    for key, value in config.items():
        if key == 'reproducible_state':
            # 상태 정보에서 직렬화 가능한 부분만 저장
            serializable_state = {}
            for state_key, state_value in value.items():
                if state_key in ['seed', 'library_versions', 'environment']:
                    serializable_state[state_key] = state_value
            serializable_config[key] = serializable_state
        else:
            serializable_config[key] = value
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"실험 설정 저장: {filepath}")


def load_experiment_config(filepath: str) -> Dict:
    """저장된 실험 설정 로드"""
    import json
    
    with open(filepath, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 시드 재설정
    if 'seed' in config:
        set_seed(config['seed'])
        logger.info(f"실험 설정 로드 및 시드 재설정: {config['seed']}")
    
    return config


if __name__ == "__main__":
    # 테스트 코드
    print("=== 시드 관리 유틸리티 테스트 ===")
    
    # 시드 설정 테스트
    print("\n1. 시드 설정 테스트:")
    results = set_seed(42)
    for lib, status in results.items():
        print(f"  {lib}: {status}")
    
    # 재현성 검증 테스트
    print("\n2. 재현성 검증 테스트:")
    is_reproducible = ensure_reproducibility(42)
    print(f"  재현성 보장: {'성공' if is_reproducible else '실패'}")
    
    # 실험 설정 생성 테스트
    print("\n3. 실험 설정 생성 테스트:")
    config = create_experiment_config(42, "test_experiment")
    print(f"  실험 이름: {config['experiment_name']}")
    print(f"  시드: {config['seed']}")
    print(f"  라이브러리 가용성: {config['library_availability']}")
    
    # 시드 변경 테스트
    print("\n4. 시드 변경 테스트:")
    manager = get_seed_manager()
    
    # 첫 번째 시드로 랜덤 값 생성
    manager.set_seed(42)
    random_val1 = random.random()
    numpy_val1 = np.random.random()
    print(f"  시드 42: random={random_val1:.6f}, numpy={numpy_val1:.6f}")
    
    # 두 번째 시드로 변경
    manager.set_seed(123)
    random_val2 = random.random()
    numpy_val2 = np.random.random()
    print(f"  시드 123: random={random_val2:.6f}, numpy={numpy_val2:.6f}")
    
    # 다시 첫 번째 시드로 변경 (재현성 확인)
    manager.set_seed(42)
    random_val3 = random.random()
    numpy_val3 = np.random.random()
    print(f"  시드 42 재설정: random={random_val3:.6f}, numpy={numpy_val3:.6f}")
    
    # 재현성 확인
    reproducible = (random_val1 == random_val3) and (numpy_val1 == numpy_val3)
    print(f"  재현성 확인: {'성공' if reproducible else '실패'}")
    
    # 설정 저장/로드 테스트
    print("\n5. 설정 저장/로드 테스트:")
    try:
        save_experiment_config(config, "test_config.json")
        loaded_config = load_experiment_config("test_config.json")
        print(f"  저장/로드 성공: {loaded_config['experiment_name']}")
        
        # 파일 정리
        import os
        if os.path.exists("test_config.json"):
            os.remove("test_config.json")
            
    except Exception as e:
        print(f"  설정 저장/로드 실패: {e}")
    
    print("\n=== 테스트 완료 ===")