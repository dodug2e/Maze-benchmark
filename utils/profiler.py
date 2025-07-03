"""
RTX 3060 최적화된 성능 프로파일링 유틸리티
GPU/CPU 사용률, VRAM, 전력 소비 등을 모니터링
"""

import time
import psutil
import threading
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# GPU 모니터링을 위한 선택적 import
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("pynvml not available. GPU monitoring disabled.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. CUDA memory monitoring disabled.")


@dataclass
class PerformanceMetrics:
    """성능 메트릭 데이터 클래스"""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    gpu_percent: float = 0.0
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    power_watts: float = 0.0
    temperature_c: float = 0.0
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            'timestamp': self.timestamp,
            'cpu_percent': self.cpu_percent,
            'memory_mb': self.memory_mb,
            'gpu_percent': self.gpu_percent,
            'vram_used_mb': self.vram_used_mb,
            'vram_total_mb': self.vram_total_mb,
            'power_watts': self.power_watts,
            'temperature_c': self.temperature_c
        }


class PerformanceProfiler:
    """RTX 3060 최적화된 성능 프로파일러"""
    
    def __init__(self, gpu_index: int = 0, monitoring_interval: float = 0.5):
        self.gpu_index = gpu_index
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.metrics_history: List[PerformanceMetrics] = []
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # GPU 핸들 초기화
        self.gpu_handle = None
        if GPU_AVAILABLE:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle)
                logger.info(f"GPU 감지: {gpu_name}")
            except Exception as e:
                logger.warning(f"GPU 초기화 실패: {e}")
                self.gpu_handle = None
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """현재 성능 메트릭 수집"""
        timestamp = time.time()
        
        # CPU 및 메모리 사용률
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        memory_mb = memory_info.used / (1024 * 1024)
        
        # GPU 메트릭 초기화
        gpu_percent = 0.0
        vram_used_mb = 0.0
        vram_total_mb = 0.0
        power_watts = 0.0
        temperature_c = 0.0
        
        # GPU 메트릭 수집
        if self.gpu_handle:
            try:
                # GPU 사용률
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_percent = gpu_util.gpu
                
                # VRAM 사용률
                vram_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                vram_used_mb = vram_info.used / (1024 * 1024)
                vram_total_mb = vram_info.total / (1024 * 1024)
                
                # 전력 소비
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                    power_watts = power_mw / 1000.0
                except:
                    pass  # 일부 GPU는 전력 모니터링 미지원
                
                # 온도
                try:
                    temperature_c = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    pass  # 일부 GPU는 온도 모니터링 미지원
                    
            except Exception as e:
                logger.warning(f"GPU 메트릭 수집 실패: {e}")
        
        # PyTorch CUDA 메모리 사용률 (가능한 경우)
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch_vram_used = torch.cuda.memory_allocated() / (1024 * 1024)
                torch_vram_cached = torch.cuda.memory_reserved() / (1024 * 1024)
                # PyTorch 메모리 정보가 더 정확할 수 있음
                if torch_vram_used > 0:
                    vram_used_mb = max(vram_used_mb, torch_vram_used)
            except Exception as e:
                logger.warning(f"PyTorch CUDA 메모리 수집 실패: {e}")
        
        return PerformanceMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            gpu_percent=gpu_percent,
            vram_used_mb=vram_used_mb,
            vram_total_mb=vram_total_mb,
            power_watts=power_watts,
            temperature_c=temperature_c
        )
    
    def start_monitoring(self):
        """백그라운드 모니터링 시작"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.metrics_history.clear()
        
        def monitor_loop():
            while self.is_monitoring:
                try:
                    metrics = self.get_current_metrics()
                    self.metrics_history.append(metrics)
                    time.sleep(self.monitoring_interval)
                except Exception as e:
                    logger.error(f"모니터링 에러: {e}")
                    
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("성능 모니터링 시작")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        if not self.is_monitoring:
            return
            
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        logger.info("성능 모니터링 중지")
    
    def get_summary_stats(self) -> Dict:
        """수집된 메트릭의 요약 통계"""
        if not self.metrics_history:
            return {}
        
        cpu_values = [m.cpu_percent for m in self.metrics_history]
        memory_values = [m.memory_mb for m in self.metrics_history]
        gpu_values = [m.gpu_percent for m in self.metrics_history]
        vram_values = [m.vram_used_mb for m in self.metrics_history]
        power_values = [m.power_watts for m in self.metrics_history if m.power_watts > 0]
        temp_values = [m.temperature_c for m in self.metrics_history if m.temperature_c > 0]
        
        def get_stats(values: List[float]) -> Dict:
            if not values:
                return {'min': 0, 'max': 0, 'avg': 0, 'peak': 0}
            return {
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'peak': max(values)
            }
        
        return {
            'duration_seconds': len(self.metrics_history) * self.monitoring_interval,
            'total_samples': len(self.metrics_history),
            'cpu_percent': get_stats(cpu_values),
            'memory_mb': get_stats(memory_values),
            'gpu_percent': get_stats(gpu_values),
            'vram_used_mb': get_stats(vram_values),
            'vram_total_mb': self.metrics_history[-1].vram_total_mb if self.metrics_history else 0,
            'power_watts': get_stats(power_values),
            'temperature_c': get_stats(temp_values)
        }
    
    def clear_history(self):
        """메트릭 히스토리 초기화"""
        self.metrics_history.clear()
    
    def export_metrics(self, filepath: str):
        """메트릭을 JSON 파일로 저장"""
        metrics_data = [m.to_dict() for m in self.metrics_history]
        summary = self.get_summary_stats()
        
        export_data = {
            'summary': summary,
            'metrics': metrics_data,
            'gpu_available': GPU_AVAILABLE,
            'torch_available': TORCH_AVAILABLE
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"메트릭 저장 완료: {filepath}")
    
    @contextmanager
    def measure(self, description: str = ""):
        """컨텍스트 매니저로 성능 측정"""
        logger.info(f"성능 측정 시작: {description}")
        
        # 초기 메트릭 수집
        start_metrics = self.get_current_metrics()
        start_time = time.time()
        
        # 모니터링 시작
        was_monitoring = self.is_monitoring
        if not was_monitoring:
            self.start_monitoring()
        
        try:
            yield self
        finally:
            # 최종 메트릭 수집
            end_time = time.time()
            end_metrics = self.get_current_metrics()
            
            # 모니터링 중지 (원래 중지 상태였다면)
            if not was_monitoring:
                self.stop_monitoring()
            
            # 결과 출력
            duration = end_time - start_time
            vram_peak = max(m.vram_used_mb for m in self.metrics_history) if self.metrics_history else 0
            
            logger.info(f"성능 측정 완료: {description}")
            logger.info(f"  실행 시간: {duration:.2f}초")
            logger.info(f"  VRAM 피크: {vram_peak:.1f}MB")
            logger.info(f"  최종 VRAM: {end_metrics.vram_used_mb:.1f}MB")
    
    def check_rtx3060_limits(self) -> Dict:
        """RTX 3060 한계 체크"""
        current = self.get_current_metrics()
        
        # RTX 3060 제한사항
        MAX_VRAM_MB = 6144  # 6GB
        MAX_TEMP_C = 83     # 일반적인 온도 한계
        MAX_POWER_W = 170   # RTX 3060 TDP
        
        warnings = []
        
        if current.vram_used_mb > MAX_VRAM_MB * 0.9:
            warnings.append(f"VRAM 사용량 높음: {current.vram_used_mb:.1f}MB / {MAX_VRAM_MB}MB")
        
        if current.temperature_c > MAX_TEMP_C * 0.9:
            warnings.append(f"GPU 온도 높음: {current.temperature_c:.1f}°C")
        
        if current.power_watts > MAX_POWER_W * 0.9:
            warnings.append(f"전력 소비 높음: {current.power_watts:.1f}W")
        
        return {
            'current_metrics': current.to_dict(),
            'limits': {
                'max_vram_mb': MAX_VRAM_MB,
                'max_temp_c': MAX_TEMP_C,
                'max_power_w': MAX_POWER_W
            },
            'warnings': warnings,
            'vram_utilization_percent': (current.vram_used_mb / MAX_VRAM_MB) * 100 if MAX_VRAM_MB > 0 else 0
        }


# 전역 프로파일러 인스턴스
_global_profiler: Optional[PerformanceProfiler] = None

def get_profiler() -> PerformanceProfiler:
    """전역 프로파일러 인스턴스 반환"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def measure_vram_usage() -> float:
    """현재 VRAM 사용량 반환 (MB)"""
    profiler = get_profiler()
    metrics = profiler.get_current_metrics()
    return metrics.vram_used_mb


def measure_gpu_utilization() -> float:
    """현재 GPU 사용률 반환 (%)"""
    profiler = get_profiler()
    metrics = profiler.get_current_metrics()
    return metrics.gpu_percent


def measure_cpu_utilization() -> float:
    """현재 CPU 사용률 반환 (%)"""
    profiler = get_profiler()
    metrics = profiler.get_current_metrics()
    return metrics.cpu_percent


def measure_power_consumption() -> float:
    """현재 전력 소비량 반환 (W)"""
    profiler = get_profiler()
    metrics = profiler.get_current_metrics()
    return metrics.power_watts


@contextmanager
def profile_execution(description: str = ""):
    """실행 성능 프로파일링 컨텍스트 매니저"""
    profiler = get_profiler()
    with profiler.measure(description):
        yield profiler


if __name__ == "__main__":
    # 테스트 코드
    print("=== 성능 프로파일러 테스트 ===")
    
    profiler = get_profiler()
    
    # 현재 메트릭 출력
    print("\n현재 시스템 상태:")
    current = profiler.get_current_metrics()
    print(f"  CPU: {current.cpu_percent:.1f}%")
    print(f"  메모리: {current.memory_mb:.1f}MB")
    print(f"  GPU: {current.gpu_percent:.1f}%")
    print(f"  VRAM: {current.vram_used_mb:.1f}MB / {current.vram_total_mb:.1f}MB")
    print(f"  전력: {current.power_watts:.1f}W")
    print(f"  온도: {current.temperature_c:.1f}°C")
    
    # RTX 3060 한계 체크
    print("\nRTX 3060 한계 체크:")
    limits_check = profiler.check_rtx3060_limits()
    print(f"  VRAM 사용률: {limits_check['vram_utilization_percent']:.1f}%")
    if limits_check['warnings']:
        print("  경고:")
        for warning in limits_check['warnings']:
            print(f"    - {warning}")
    else:
        print("  모든 지표가 정상 범위입니다.")
    
    # 성능 측정 테스트
    print("\n성능 측정 테스트 (5초):")
    with profile_execution("테스트 실행"):
        # 간단한 계산 작업
        import numpy as np
        for i in range(100):
            arr = np.random.randn(1000, 1000)
            result = np.dot(arr, arr.T)
            time.sleep(0.05)
    
    # 요약 통계 출력
    summary = profiler.get_summary_stats()
    print(f"\n요약 통계:")
    print(f"  실행 시간: {summary.get('duration_seconds', 0):.2f}초")
    print(f"  CPU 평균: {summary.get('cpu_percent', {}).get('avg', 0):.1f}%")
    print(f"  VRAM 피크: {summary.get('vram_used_mb', {}).get('peak', 0):.1f}MB")