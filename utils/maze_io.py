"""
미로 데이터 로딩 및 입출력 유틸리티 (수정 버전)
datasets/{train,valid,test} 구조에서 PNG, JSON, NPY 파일 로드
자료형 및 경로 불일치 문제 해결
"""

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, Optional, Union, List
import hashlib
import logging

logger = logging.getLogger(__name__)

class MazeDataLoader:
    """미로 데이터셋 로더 - 수정 버전"""
    
    def __init__(self, dataset_root: str = "datasets"):
        self.dataset_root = Path(dataset_root)
        self._validate_dataset_structure()
        
    def _validate_dataset_structure(self):
        """데이터셋 구조 검증 및 자동 수정"""
        required_subsets = ["train", "valid", "test"]
        
        for subset in required_subsets:
            subset_path = self.dataset_root / subset
            if not subset_path.exists():
                logger.warning(f"Dataset subset '{subset}' not found, creating: {subset_path}")
                subset_path.mkdir(parents=True, exist_ok=True)
            
            # 이미지 디렉토리 통일 (img vs images)
            img_dir = subset_path / "img"
            images_dir = subset_path / "images"
            
            if images_dir.exists() and not img_dir.exists():
                logger.info(f"Renaming 'images' to 'img' in {subset}")
                images_dir.rename(img_dir)
            elif not img_dir.exists():
                img_dir.mkdir(exist_ok=True)
            
            # 필수 디렉토리 생성
            for subdir in ["metadata", "arrays"]:
                subdir_path = subset_path / subdir
                if not subdir_path.exists():
                    logger.info(f"Creating missing directory: {subdir_path}")
                    subdir_path.mkdir(exist_ok=True)
    
    def load_sample(self, sample_id: str, subset: str = "train") -> Tuple[Image.Image, Dict, Optional[np.ndarray]]:
        """
        샘플 데이터 로드 (수정 버전)
        
        Args:
            sample_id: 샘플 ID (예: "000001")
            subset: 데이터셋 분할 ("train", "valid", "test")
            
        Returns:
            tuple: (PIL Image, metadata dict, numpy array or None)
        """
        if subset not in ["train", "valid", "test"]:
            raise ValueError(f"Invalid subset: {subset}")
            
        subset_path = self.dataset_root / subset
        
        # PNG 이미지 로드 (통일된 경로)
        img_path = subset_path / "img" / f"{sample_id}.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        img = Image.open(img_path)
        
        # JSON 메타데이터 로드 및 키 통일
        meta_path = subset_path / "metadata" / f"{sample_id}.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 메타데이터 키 통일 (entrance/exit -> start/goal)
        metadata = self._normalize_metadata(metadata)
        
        # NPY 배열 로드 (선택적, 자료형 통일)
        array_path = subset_path / "arrays" / f"{sample_id}.npy"
        array = None
        if array_path.exists():
            try:
                array = np.load(array_path).astype(np.float32)  # float32로 통일
            except Exception as e:
                logger.warning(f"Failed to load array {array_path}: {e}")
        
        return img, metadata, array
    
    def _normalize_metadata(self, metadata: Dict) -> Dict:
        """메타데이터 키 정규화"""
        normalized = metadata.copy()
        
        # entrance -> start 변환
        if 'entrance' in normalized and 'start' not in normalized:
            normalized['start'] = normalized['entrance']
        
        # exit -> goal 변환  
        if 'exit' in normalized and 'goal' not in normalized:
            normalized['goal'] = normalized['exit']
        elif 'exit_point' in normalized and 'goal' not in normalized:
            normalized['goal'] = normalized['exit_point']
        
        # 기본값 설정
        if 'start' not in normalized:
            normalized['start'] = (1, 1)
        if 'goal' not in normalized:
            # 미로 크기 기반 기본 목표점
            size = normalized.get('size', (50, 50))
            normalized['goal'] = (size[0]-2, size[1]-2)
        
        return normalized
    
    def load_dataset_info(self, subset: str) -> Dict:
        """데이터셋 정보 로드"""
        info_path = self.dataset_root / subset / "dataset_info.json"
        if not info_path.exists():
            logger.warning(f"Dataset info not found: {info_path}, creating default")
            return self._create_default_dataset_info(subset)
        
        with open(info_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_default_dataset_info(self, subset: str) -> Dict:
        """기본 데이터셋 정보 생성"""
        sample_ids = self.get_sample_ids(subset)
        return {
            "subset": subset,
            "total_samples": len(sample_ids),
            "maze_sizes": ["50x50", "100x100", "150x150", "200x200"],
            "algorithms": ["recursive_backtrack", "binary_tree", "eller"],
            "created_at": "auto-generated"
        }
    
    def get_sample_ids(self, subset: str) -> List[str]:
        """특정 subset의 모든 샘플 ID 목록 반환"""
        img_dir = self.dataset_root / subset / "img"
        if not img_dir.exists():
            logger.warning(f"Image directory not found: {img_dir}, returning empty list")
            return []
        
        sample_ids = []
        for img_file in img_dir.glob("*.png"):
            sample_ids.append(img_file.stem)
        
        return sorted(sample_ids)
    
    def convert_image_to_array(self, img: Image.Image) -> np.ndarray:
        """PIL 이미지를 NumPy 배열로 변환 (자료형 통일)"""
        # 흑백 이미지로 변환
        if img.mode != 'L':
            img = img.convert('L')
        
        # float32로 통일하여 자료형 불일치 방지
        array = np.array(img, dtype=np.float32)
        
        # 0 (벽) 또는 1 (통로)로 정규화
        array = (array > 128).astype(np.float32)
        
        return array
    
    def get_maze_size(self, sample_id: str, subset: str = "train") -> Tuple[int, int]:
        """미로 크기 반환"""
        try:
            _, metadata, _ = self.load_sample(sample_id, subset)
            if 'size' in metadata:
                size = metadata['size']
                if isinstance(size, (list, tuple)) and len(size) == 2:
                    return tuple(size)
            
            # 메타데이터에서 크기 정보를 찾을 수 없으면 이미지에서 추출
            img_path = self.dataset_root / subset / "img" / f"{sample_id}.png"
            if img_path.exists():
                with Image.open(img_path) as img:
                    return img.size[::-1]  # (width, height) -> (height, width)
                    
        except Exception as e:
            logger.warning(f"Could not determine size for {sample_id}: {e}")
        
        return (50, 50)  # 기본값
    
    def verify_sample(self, sample_id: str, subset: str = "train") -> bool:
        """샘플 데이터 유효성 검증"""
        try:
            img, metadata, array = self.load_sample(sample_id, subset)
            
            # 이미지 검증
            if img is None or img.size[0] == 0 or img.size[1] == 0:
                return False
            
            # 메타데이터 검증
            required_keys = ['start', 'goal']
            for key in required_keys:
                if key not in metadata:
                    logger.warning(f"Missing key '{key}' in metadata for {sample_id}")
                    return False
            
            # 배열 검증 (있는 경우)
            if array is not None:
                if array.dtype != np.float32:
                    logger.warning(f"Array dtype mismatch for {sample_id}: {array.dtype}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Sample verification failed for {sample_id}: {e}")
            return False
    
    def get_valid_sample_ids(self, subset: str, max_samples: Optional[int] = None) -> List[str]:
        """유효한 샘플 ID만 반환"""
        all_ids = self.get_sample_ids(subset)
        valid_ids = []
        
        for sample_id in all_ids:
            if self.verify_sample(sample_id, subset):
                valid_ids.append(sample_id)
            
            if max_samples and len(valid_ids) >= max_samples:
                break
        
        logger.info(f"Found {len(valid_ids)} valid samples out of {len(all_ids)} in {subset}")
        return valid_ids


# 전역 로더 인스턴스
_global_loader = None

def get_loader(dataset_root: str = "datasets") -> MazeDataLoader:
    """전역 미로 데이터 로더 인스턴스 반환"""
    global _global_loader
    if _global_loader is None:
        _global_loader = MazeDataLoader(dataset_root)
    return _global_loader


# 데이터 일관성 검사 유틸리티
def check_dataset_consistency(dataset_root: str = "datasets"):
    """데이터셋 일관성 검사 및 보고서 생성"""
    loader = MazeDataLoader(dataset_root)
    
    print("="*50)
    print("데이터셋 일관성 검사 보고서")
    print("="*50)
    
    for subset in ["train", "valid", "test"]:
        print(f"\n[{subset.upper()}]")
        
        all_ids = loader.get_sample_ids(subset)
        valid_ids = loader.get_valid_sample_ids(subset)
        
        print(f"총 샘플 수: {len(all_ids)}")
        print(f"유효 샘플 수: {len(valid_ids)}")
        print(f"유효율: {len(valid_ids)/len(all_ids)*100:.1f}%" if all_ids else "N/A")
        
        if valid_ids:
            # 크기 분포 확인
            sizes = [loader.get_maze_size(sid, subset) for sid in valid_ids[:100]]
            unique_sizes = list(set(sizes))
            print(f"미로 크기 종류: {unique_sizes}")


if __name__ == "__main__":
    # 데이터셋 일관성 검사 실행
    check_dataset_consistency()
    
    # 샘플 사용법
    loader = get_loader()
    
    # 훈련 데이터 로드 예시
    train_ids = loader.get_valid_sample_ids("train", max_samples=10)
    
    for sample_id in train_ids[:3]:
        try:
            img, metadata, array = loader.load_sample(sample_id, "train")
            print(f"\n샘플 {sample_id}:")
            print(f"  이미지 크기: {img.size}")
            print(f"  시작점: {metadata['start']}")
            print(f"  목표점: {metadata['goal']}")
            print(f"  배열 존재: {array is not None}")
            if array is not None:
                print(f"  배열 형태: {array.shape}, 타입: {array.dtype}")
        except Exception as e:
            print(f"샘플 {sample_id} 로드 실패: {e}")