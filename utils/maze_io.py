"""
미로 데이터 로딩 및 입출력 유틸리티
datasets/{train,valid,test} 구조에서 PNG, JSON, NPY 파일 로드
"""

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, Optional, Union
import hashlib
import logging

logger = logging.getLogger(__name__)

class MazeDataLoader:
    """미로 데이터셋 로더"""
    
    def __init__(self, dataset_root: str = "datasets"):
        self.dataset_root = Path(dataset_root)
        self._validate_dataset_structure()
        
    def _validate_dataset_structure(self):
        """데이터셋 구조 검증"""
        required_subsets = ["train", "valid", "test"]
        required_subdirs = ["img", "metadata", "arrays"]
        
        for subset in required_subsets:
            subset_path = self.dataset_root / subset
            if not subset_path.exists():
                raise FileNotFoundError(f"Dataset subset '{subset}' not found at {subset_path}")
            
            for subdir in required_subdirs:
                subdir_path = subset_path / subdir
                if not subdir_path.exists():
                    raise FileNotFoundError(f"Required subdirectory '{subdir}' not found at {subdir_path}")
    
    def load_sample(self, sample_id: str, subset: str = "train") -> Tuple[Image.Image, Dict, Optional[np.ndarray]]:
        """
        샘플 데이터 로드
        
        Args:
            sample_id: 샘플 ID (예: "000001")
            subset: 데이터셋 분할 ("train", "valid", "test")
            
        Returns:
            tuple: (PIL Image, metadata dict, numpy array or None)
        """
        if subset not in ["train", "valid", "test"]:
            raise ValueError(f"Invalid subset: {subset}")
            
        subset_path = self.dataset_root / subset
        
        # PNG 이미지 로드
        img_path = subset_path / "images" / f"{sample_id}.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path)
        
        # JSON 메타데이터 로드
        meta_path = subset_path / "metadata" / f"{sample_id}.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # NPY 배열 로드 (선택적)
        array_path = subset_path / "arrays" / f"{sample_id}.npy"
        array = None
        if array_path.exists():
            try:
                array = np.load(array_path)
            except Exception as e:
                logger.warning(f"Failed to load array {array_path}: {e}")
        
        return img, metadata, array
    
    def load_dataset_info(self, subset: str) -> Dict:
        """데이터셋 정보 로드"""
        info_path = self.dataset_root / subset / "dataset_info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Dataset info not found: {info_path}")
        
        with open(info_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_sample_ids(self, subset: str) -> list:
        """특정 subset의 모든 샘플 ID 목록 반환"""
        img_dir = self.dataset_root / subset / "img"
        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        
        sample_ids = []
        for img_file in img_dir.glob("*.png"):
            sample_ids.append(img_file.stem)
        
        return sorted(sample_ids)
    
    def convert_image_to_array(self, img: Image.Image) -> np.ndarray:
        """PIL 이미지를 NumPy 배열로 변환"""
        # 흑백 이미지로 변환 (미로는 일반적으로 흑백)
        if img.mode != 'L':
            img = img.convert('L')
        
        # 0 (벽) 또는 1 (통로)로 정규화
        array = np.array(img, dtype=np.uint8)
        # 일반적으로 검은색(0)이 벽, 흰색(255)이 통로
        array = (array > 128).astype(np.uint8)
        
        return array
    
    def get_maze_size(self, sample_id: str, subset: str = "train") -> Tuple[int, int]:
        """미로 크기 반환 (height, width)"""
        img, _, _ = self.load_sample(sample_id, subset)
        return img.size[::-1]  # PIL은 (width, height) 반환하므로 뒤집기
    
    def batch_load_samples(self, sample_ids: list, subset: str = "train") -> list:
        """배치로 샘플들 로드"""
        samples = []
        for sample_id in sample_ids:
            try:
                img, meta, array = self.load_sample(sample_id, subset)
                samples.append({
                    'id': sample_id,
                    'image': img,
                    'metadata': meta,
                    'array': array
                })
            except Exception as e:
                logger.error(f"Failed to load sample {sample_id}: {e}")
                continue
        
        return samples


# 전역 로더 인스턴스
_global_loader = None

def get_loader(dataset_root: str = "datasets") -> MazeDataLoader:
    """전역 데이터 로더 인스턴스 반환"""
    global _global_loader
    if _global_loader is None:
        _global_loader = MazeDataLoader(dataset_root)
    return _global_loader


def load_sample(sample_id: str, subset: str = "train") -> Tuple[Image.Image, Dict, Optional[np.ndarray]]:
    """
    편의 함수: 샘플 로드
    
    Args:
        sample_id: 샘플 ID
        subset: 데이터셋 분할
        
    Returns:
        tuple: (PIL Image, metadata dict, numpy array or None)
    """
    loader = get_loader()
    return loader.load_sample(sample_id, subset)


def load_maze_as_array(sample_id: str, subset: str = "train") -> Tuple[np.ndarray, Dict]:
    """
    편의 함수: 미로를 NumPy 배열로 로드
    
    Returns:
        tuple: (maze array, metadata dict)
    """
    loader = get_loader()
    img, metadata, array = loader.load_sample(sample_id, subset)
    
    # 미리 저장된 배열이 있으면 사용, 없으면 이미지에서 변환
    if array is not None:
        maze_array = array
    else:
        maze_array = loader.convert_image_to_array(img)
    
    return maze_array, metadata


def get_dataset_stats(subset: str = "train") -> Dict:
    """데이터셋 통계 정보 반환"""
    loader = get_loader()
    sample_ids = loader.get_sample_ids(subset)
    
    stats = {
        'total_samples': len(sample_ids),
        'sample_ids': sample_ids[:10],  # 처음 10개만 표시
        'sizes': {}
    }
    
    # 크기 분포 계산 (처음 100개 샘플만)
    size_counts = {}
    for sample_id in sample_ids[:100]:
        try:
            size = loader.get_maze_size(sample_id, subset)
            size_str = f"{size[0]}x{size[1]}"
            size_counts[size_str] = size_counts.get(size_str, 0) + 1
        except Exception as e:
            logger.warning(f"Failed to get size for sample {sample_id}: {e}")
    
    stats['sizes'] = size_counts
    return stats


if __name__ == "__main__":
    # 테스트 코드
    try:
        # 데이터셋 통계 출력
        print("=== 데이터셋 통계 ===")
        for subset in ["train", "valid", "test"]:
            stats = get_dataset_stats(subset)
            print(f"\n{subset.upper()} 데이터셋:")
            print(f"  총 샘플 수: {stats['total_samples']}")
            print(f"  크기 분포: {stats['sizes']}")
        
        # 샘플 로드 테스트
        print("\n=== 샘플 로드 테스트 ===")
        loader = get_loader()
        sample_ids = loader.get_sample_ids("train")
        
        if sample_ids:
            sample_id = sample_ids[0]
            img, meta, array = load_sample(sample_id, "train")
            print(f"샘플 {sample_id} 로드 성공:")
            print(f"  이미지 크기: {img.size}")
            print(f"  메타데이터 키: {list(meta.keys())}")
            print(f"  배열 형태: {array.shape if array is not None else 'None'}")
            
            # 배열 변환 테스트
            maze_array, _ = load_maze_as_array(sample_id, "train")
            print(f"  미로 배열 형태: {maze_array.shape}")
            print(f"  미로 배열 값 범위: {maze_array.min()} ~ {maze_array.max()}")
        
    except Exception as e:
        print(f"테스트 실패: {e}")