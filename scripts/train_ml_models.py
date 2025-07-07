#!/usr/bin/env python3
"""
CNN과 Deep Forest 통합 학습 스크립트
RTX 3060 최적화 버전
"""

import argparse
import logging
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List
import torch

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 로컬 모듈 import
from utils.maze_io import get_loader
from utils.profiler import get_profiler, profile_execution
from algorithms.cnn_model import MazePathCNN, CNNTrainer, MazeDataset
from algorithms.deep_forest_model import DeepForestTrainer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MLModelTrainer:
    """ML 모델 통합 학습 관리자"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.maze_loader = get_loader(config.get('dataset_path', 'datasets'))
        self.profiler = get_profiler()
        self.results = {}
        
        # 결과 저장 디렉터리 생성
        self.output_dir = Path(config.get('output_dir', 'output'))
        self.output_dir.mkdir(exist_ok=True)
        
        # 모델 저장 디렉터리
        self.models_dir = self.output_dir / 'models'
        self.models_dir.mkdir(exist_ok=True)
        
        # 로그 디렉터리
        self.logs_dir = self.output_dir / 'logs'
        self.logs_dir.mkdir(exist_ok=True)
    
    def get_sample_splits(self) -> Dict[str, List[str]]:
        """데이터셋 분할 정보 가져오기"""
        splits = {}
        
        for subset in ['train', 'valid', 'test']:
            sample_ids = self.maze_loader.get_sample_ids(subset)
            
            # RTX 3060 메모리 제한을 고려한 샘플 수 조정
            max_samples = self.config.get(f'max_{subset}_samples', {
                'train': 10000,
                'valid': 8000,
                'test': 2000
            }[subset])
            
            if len(sample_ids) > max_samples:
                sample_ids = sample_ids[:max_samples]
                logger.info(f"{subset} 샘플 수 제한: {len(sample_ids)}")
            
            splits[subset] = sample_ids
        
        logger.info(f"데이터 분할: train={len(splits['train'])}, "
                   f"valid={len(splits['valid'])}, test={len(splits['test'])}")
        
        return splits
    
    def train_cnn_model(self, splits: Dict[str, List[str]]) -> Dict:
        """CNN 모델 학습"""
        logger.info("\n" + "="*50)
        logger.info("CNN 모델 학습 시작")
        logger.info("="*50)
        
        with profile_execution("CNN 학습") as profiler:
            # CNN 설정
            cnn_config = self.config.get('cnn', {})
            model = MazePathCNN(
                input_size=cnn_config.get('input_size', 200),
                num_classes=cnn_config.get('num_classes', 4),
                dropout_rate=cnn_config.get('dropout_rate', 0.2)
            )
            
            # 모델 정보 출력
            model_info = model.get_model_size()
            logger.info(f"CNN 파라미터: {model_info['total_parameters']:,}")
            logger.info(f"예상 메모리: {model_info['estimated_memory_mb']:.1f}MB")
            
            # 학습기 생성
            trainer = CNNTrainer(
                model=model,
                learning_rate=cnn_config.get('learning_rate', 0.001)
            )
            
            # 데이터셋 생성
            train_dataset = MazeDataset(
                self.maze_loader, splits['train'], 'train'
            )
            val_dataset = MazeDataset(
                self.maze_loader, splits['valid'], 'valid'
            )
            
            # 데이터 로더 생성
            batch_size = cnn_config.get('batch_size', 16)  # RTX 3060에 맞춰 작은 배치
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )
            
            val_loader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
            
            # 학습 루프
            epochs = cnn_config.get('epochs', 10)
            best_val_acc = 0
            train_history = []
            
            start_time = time.time()
            
            for epoch in range(epochs):
                # 학습
                train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
                
                # 검증
                val_loss, val_acc = trainer.validate(val_loader)
                
                # 기록
                epoch_result = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'vram_usage': profiler.get_current_metrics().vram_used_mb
                }
                train_history.append(epoch_result)
                
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                           f"Train: {train_loss:.4f}({train_acc:.2f}%) "
                           f"Val: {val_loss:.4f}({val_acc:.2f}%) "
                           f"VRAM: {epoch_result['vram_usage']:.1f}MB")
                
                # 최고 성능 모델 저장
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    model_path = self.models_dir / 'best_cnn_model.pth'
                    trainer.save_model(str(model_path), epoch, val_loss)
            
            training_time = time.time() - start_time
            
            # 최종 모델 저장
            final_model_path = self.models_dir / 'final_cnn_model.pth'
            trainer.save_model(str(final_model_path), epochs-1, val_loss)
            
            # 결과 정리
            cnn_result = {
                'model_type': 'CNN',
                'training_time': training_time,
                'best_val_accuracy': best_val_acc,
                'final_val_accuracy': val_acc,
                'total_epochs': epochs,
                'model_info': model_info,
                'train_history': train_history,
                'model_paths': {
                    'best': str(model_path),
                    'final': str(final_model_path)
                }
            }
            
            # 성능 요약 저장
            performance_summary = profiler.get_summary_stats()
            cnn_result['performance'] = performance_summary
            
            logger.info(f"CNN 학습 완료 - 최고 검증 정확도: {best_val_acc:.4f}")
            
            return cnn_result
    
    def train_deep_forest_model(self, splits: Dict[str, List[str]]) -> Dict:
        """Deep Forest 모델 학습"""
        logger.info("\n" + "="*50)
        logger.info("Deep Forest 모델 학습 시작")
        logger.info("="*50)
        
        with profile_execution("Deep Forest 학습") as profiler:
            # Deep Forest 설정
            df_config = self.config.get('deep_forest', {})
            
            # RTX 3060에 맞춘 기본 설정
            model_config = {
                'n_layers': df_config.get('n_layers', 2),
                'n_estimators': df_config.get('n_estimators', 50),
                'max_depth': df_config.get('max_depth', 10),
                'min_improvement': df_config.get('min_improvement', 0.005),
                'patience': df_config.get('patience', 5),
                'random_state': 42
            }
            
            # 학습기 생성
            trainer = DeepForestTrainer(
                model_config=model_config,
                save_dir=str(self.models_dir)
            )
            
            # 학습 실행
            start_time = time.time()
            train_result = trainer.train(
                self.maze_loader, 
                splits['train'], 
                splits['valid']
            )
            
            # 테스트 평가
            test_result = trainer.evaluate(
                self.maze_loader, 
                splits['test']
            )
            
            training_time = time.time() - start_time
            
            # 결과 정리
            df_result = {
                'model_type': 'Deep Forest',
                'training_time': training_time,
                'train_accuracy': train_result['train_accuracy'],
                'val_accuracy': train_result['val_accuracy'],
                'test_accuracy': test_result['test_accuracy'],
                'best_layer_count': train_result['best_layer_count'],
                'model_path': train_result['model_path'],
                'model_info': trainer.model.get_model_info() if trainer.model else {},
                'feature_importance': trainer.model.get_feature_importance() if trainer.model else {}
            }
            
            # 성능 요약 저장
            performance_summary = profiler.get_summary_stats()
            df_result['performance'] = performance_summary
            
            logger.info(f"Deep Forest 학습 완료 - 테스트 정확도: {test_result['test_accuracy']:.4f}")
            
            return df_result
    
    def run_training(self) -> Dict:
        """전체 학습 파이프라인 실행"""
        logger.info("ML 모델 학습 파이프라인 시작")
        
        # 시스템 상태 확인
        rtx_check = self.profiler.check_rtx3060_limits()
        logger.info(f"초기 VRAM 사용률: {rtx_check['vram_utilization_percent']:.1f}%")
        
        if rtx_check['warnings']:
            logger.warning("초기 상태 경고:")
            for warning in rtx_check['warnings']:
                logger.warning(f"  - {warning}")
        
        # 데이터셋 분할
        splits = self.get_sample_splits()
        
        # 학습할 모델 목록
        models_to_train = self.config.get('models', ['cnn', 'deep_forest'])
        
        results = {}
        
        # CNN 학습
        if 'cnn' in models_to_train:
            try:
                import torch
                results['cnn'] = self.train_cnn_model(splits)
                
                # GPU 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"CNN 학습 실패: {e}")
                results['cnn'] = {'error': str(e)}
        
        # Deep Forest 학습
        if 'deep_forest' in models_to_train:
            try:
                results['deep_forest'] = self.train_deep_forest_model(splits)
            except Exception as e:
                logger.error(f"Deep Forest 학습 실패: {e}")
                results['deep_forest'] = {'error': str(e)}
        
        # 전체 결과 저장
        results_path = self.output_dir / 'training_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 최종 시스템 상태
        final_check = self.profiler.check_rtx3060_limits()
        logger.info(f"최종 VRAM 사용률: {final_check['vram_utilization_percent']:.1f}%")
        
        logger.info(f"학습 완료 - 결과 저장: {results_path}")
        
        return results
    
    def generate_report(self, results: Dict):
        """학습 결과 리포트 생성"""
        report_path = self.output_dir / 'training_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# ML 모델 학습 결과 리포트\n\n")
            f.write(f"생성일시: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # CNN 결과
            if 'cnn' in results and 'error' not in results['cnn']:
                cnn = results['cnn']
                f.write("## CNN 모델\n\n")
                f.write(f"- 학습 시간: {cnn['training_time']:.2f}초\n")
                f.write(f"- 최고 검증 정확도: {cnn['best_val_accuracy']:.4f}\n")
                f.write(f"- 총 파라미터: {cnn['model_info']['total_parameters']:,}\n")
                f.write(f"- 피크 VRAM: {cnn['performance']['vram_used_mb']['peak']:.1f}MB\n\n")
            
            # Deep Forest 결과
            if 'deep_forest' in results and 'error' not in results['deep_forest']:
                df = results['deep_forest']
                f.write("## Deep Forest 모델\n\n")
                f.write(f"- 학습 시간: {df['training_time']:.2f}초\n")
                f.write(f"- 테스트 정확도: {df['test_accuracy']:.4f}\n")
                f.write(f"- 최적 레이어 수: {df['best_layer_count']}\n")
                f.write(f"- 총 트리 수: {df['model_info'].get('total_trees', 'N/A')}\n\n")
                
                # 특성 중요도 상위 5개
                if 'feature_importance' in df:
                    f.write("### 주요 특성 중요도\n\n")
                    for i, (feature, importance) in enumerate(list(df['feature_importance'].items())[:5]):
                        f.write(f"{i+1}. {feature}: {importance:.4f}\n")
                    f.write("\n")
        
        logger.info(f"리포트 생성 완료: {report_path}")


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description="ML 모델 학습 스크립트")
    
    parser.add_argument('--config', type=str, default='configs/ml_training.json',
                       help='설정 파일 경로')
    parser.add_argument('--models', nargs='+', choices=['cnn', 'deep_forest'],
                       default=['cnn', 'deep_forest'],
                       help='학습할 모델 선택')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='결과 저장 디렉터리')
    parser.add_argument('--dry-run', action='store_true',
                       help='빠른 테스트 실행 (적은 데이터)')
    parser.add_argument('--gpu-check', action='store_true',
                       help='GPU 상태만 확인하고 종료')
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """설정 파일 로드"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # 기본 설정
        logger.warning(f"설정 파일을 찾을 수 없음: {config_path}, 기본 설정 사용")
        return {
            "dataset_path": "datasets",
            "output_dir": "output",
            "models": ["cnn", "deep_forest"],
            "max_train_samples": 1000,
            "max_valid_samples": 200,
            "max_test_samples": 100,
            "cnn": {
                "input_size": 200,
                "num_classes": 4,
                "dropout_rate": 0.2,
                "learning_rate": 0.001,
                "batch_size": 8,
                "epochs": 5
            },
            "deep_forest": {
                "n_layers": 2,
                "n_estimators": 30,
                "max_depth": 8,
                "min_improvement": 0.01,
                "patience": 1
            }
        }


def main():
    """메인 함수"""
    args = parse_arguments()
    
    # GPU 상태 확인
    profiler = get_profiler()
    gpu_status = profiler.check_rtx3060_limits()
    
    print(f"=== GPU 상태 확인 ===")
    print(f"VRAM 사용률: {gpu_status['vram_utilization_percent']:.1f}%")
    print(f"현재 VRAM: {gpu_status['current_metrics']['vram_used_mb']:.1f}MB")
    
    if gpu_status['warnings']:
        print("경고 사항:")
        for warning in gpu_status['warnings']:
            print(f"  - {warning}")
    
    if args.gpu_check:
        return
    
    # 설정 로드
    config = load_config(args.config)
    
    # 명령행 인수로 설정 오버라이드
    if args.models:
        config['models'] = args.models
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # Dry run 모드
    if args.dry_run:
        logger.info("=== DRY RUN 모드 ===")
        config['max_train_samples'] = 50
        config['max_valid_samples'] = 20
        config['max_test_samples'] = 10
        config['cnn']['epochs'] = 2
        config['cnn']['batch_size'] = 4
        config['deep_forest']['n_estimators'] = 10
        config['deep_forest']['n_layers'] = 1
    
    try:
        # 학습 실행
        trainer = MLModelTrainer(config)
        results = trainer.run_training()
        
        # 리포트 생성
        trainer.generate_report(results)
        
        # 요약 출력
        print("\n" + "="*60)
        print("학습 완료 요약")
        print("="*60)
        
        for model_name, result in results.items():
            if 'error' in result:
                print(f"{model_name.upper()}: 실패 - {result['error']}")
            else:
                print(f"{model_name.upper()}: 성공")
                if model_name == 'cnn':
                    print(f"  최고 검증 정확도: {result['best_val_accuracy']:.4f}")
                    print(f"  학습 시간: {result['training_time']:.1f}초")
                elif model_name == 'deep_forest':
                    print(f"  테스트 정확도: {result['test_accuracy']:.4f}")
                    print(f"  학습 시간: {result['training_time']:.1f}초")
        
        # 최종 GPU 상태
        final_status = profiler.check_rtx3060_limits()
        print(f"\n최종 VRAM 사용률: {final_status['vram_utilization_percent']:.1f}%")
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()