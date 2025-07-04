#!/usr/bin/env python3
"""
Maze Benchmark CLI 진입점
Usage: 
    python maze_runner.py train algo=DQN subset=train --dry-run
    python maze_runner.py profile algo=ACO metric=vram subset=test
    python maze_runner.py benchmark --all-algorithms
"""

import argparse
import sys
import subprocess
from pathlib import Path
import os

def run_training(args):
    """학습 실행"""
    cmd = [
        sys.executable, 
        "scripts/train.py",
        args.algo
    ]
    
    if args.subset:
        cmd.extend(["--subset", args.subset])
    if args.dry_run:
        cmd.append("--dry-run")
    if args.max_samples:
        cmd.extend(["--max-samples", str(args.max_samples)])
    if args.seed:
        cmd.extend(["--seed", str(args.seed)])
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd)

def run_profiling(args):
    """프로파일링 실행"""
    cmd = [
        sys.executable,
        "scripts/profile.py",
        args.algo
    ]
    
    if args.metric:
        cmd.extend(["--metric", args.metric])
    if args.subset:
        cmd.extend(["--subset", args.subset])
    if args.duration:
        cmd.extend(["--duration", str(args.duration)])
    
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd)

def run_benchmark(args):
    """벤치마크 실행"""
    if args.all_algorithms:
        algorithms = ['ACO', 'ACO_CNN', 'ACO_DeepForest', 'DQN', 'DQN_DeepForest', 'PPO', 'A_STAR']
        
        for algo in algorithms:
            print(f"\n{'='*50}")
            print(f"Running benchmark for {algo}")
            print(f"{'='*50}")
            
            cmd = [
                sys.executable,
                "scripts/train.py",
                algo,
                "--subset", "test",
                "--max-samples", "100"  # 벤치마크용 샘플 수
            ]
            
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"Warning: {algo} benchmark failed")
                continue
    else:
        print("Individual algorithm benchmarking not implemented yet")
        return 1
    
    return 0

def generate_dataset(args):
    """데이터셋 생성"""
    cmd = [
        sys.executable,
        "-c",
        f"""
from maze_generator import MazeDatasetGenerator
generator = MazeDatasetGenerator('datasets')
generator.generate_dataset(
    total_samples={args.total_samples},
    size_range=({args.min_size}, {args.max_size}),
    train_ratio={args.train_ratio},
    val_ratio={args.val_ratio},
    test_ratio={args.test_ratio}
)
"""
    ]
    
    print(f"Generating dataset with {args.total_samples} samples...")
    return subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description='Maze Benchmark CLI')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train 서브커맨드
    train_parser = subparsers.add_parser('train', help='Train algorithms')
    train_parser.add_argument('algo', choices=['ACO', 'ACO_CNN', 'ACO_DeepForest', 
                                             'DQN', 'DQN_DeepForest', 'PPO', 'A_STAR'],
                             help='Algorithm to train')
    train_parser.add_argument('--subset', choices=['train', 'valid', 'test'], 
                             default='train', help='Dataset subset')
    train_parser.add_argument('--dry-run', action='store_true', 
                             help='Run with limited samples for testing')
    train_parser.add_argument('--max-samples', type=int, 
                             help='Maximum number of samples to process')
    train_parser.add_argument('--seed', type=int, default=42, 
                             help='Random seed for reproducibility')
    train_parser.add_argument('--output-dir', default='results', 
                             help='Output directory for results')
    
    # Profile 서브커맨드
    profile_parser = subparsers.add_parser('profile', help='Profile algorithms')
    profile_parser.add_argument('algo', choices=['ACO', 'ACO_CNN', 'ACO_DeepForest', 
                                               'DQN', 'DQN_DeepForest', 'PPO', 'A_STAR'],
                               help='Algorithm to profile')
    profile_parser.add_argument('--metric', choices=['vram', 'gpu', 'cpu', 'power'], 
                               default='vram', help='Metric to profile')
    profile_parser.add_argument('--subset', choices=['train', 'valid', 'test'], 
                               default='test', help='Dataset subset')
    profile_parser.add_argument('--duration', type=int, default=60, 
                               help='Profiling duration in seconds')
    
    # Benchmark 서브커맨드
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    benchmark_parser.add_argument('--all-algorithms', action='store_true',
                                 help='Run benchmark for all algorithms')
    
    # Generate 서브커맨드
    generate_parser = subparsers.add_parser('generate', help='Generate dataset')
    generate_parser.add_argument('--total-samples', type=int, default=12000,
                                help='Total number of samples to generate')
    generate_parser.add_argument('--min-size', type=int, default=50,
                                help='Minimum maze size')
    generate_parser.add_argument('--max-size', type=int, default=200,
                                help='Maximum maze size')
    generate_parser.add_argument('--train-ratio', type=float, default=0.7,
                                help='Training set ratio')
    generate_parser.add_argument('--val-ratio', type=float, default=0.15,
                                help='Validation set ratio')
    generate_parser.add_argument('--test-ratio', type=float, default=0.15,
                                help='Test set ratio')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # 프로젝트 루트 디렉토리로 이동
    os.chdir(Path(__file__).parent)
    
    if args.command == 'train':
        return run_training(args).returncode
    elif args.command == 'profile':
        return run_profiling(args).returncode
    elif args.command == 'benchmark':
        return run_benchmark(args)
    elif args.command == 'generate':
        return generate_dataset(args).returncode
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())