#!/usr/bin/env python3
"""
발표용 실시간 시연 스크립트
랩 세미나에서 DQN, PPO 알고리즘을 실시간으로 시연하는 도구
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import sys
import time
import json

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from algorithms.dqn_solver import DQNSolver
from algorithms.ppo_solver import PPOSolver
from utils.profiler import get_profiler

class PresentationDemo:
    """발표용 실시간 시연"""
    
    def __init__(self):
        self.demo_maze = self._create_demo_maze()
        self.start = (1, 1)
        self.goal = (8, 8)
        self.results = {}
        
    def _create_demo_maze(self):
        """시연용 미로 생성 (10x10, 해결 가능)"""
        maze = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])
        return maze
    
    def visualize_maze(self, path=None, title="미로"):
        """미로 시각화"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 미로 표시 (0: 검은색 벽, 1: 흰색 통로)
        display_maze = self.demo_maze.copy().astype(float)
        
        # 경로 표시
        if path:
            for i, (x, y) in enumerate(path):
                if 0 <= x < self.demo_maze.shape[0] and 0 <= y < self.demo_maze.shape[1]:
                    intensity = 0.3 + 0.4 * (i / len(path))  # 경로가 진행될수록 밝게
                    display_maze[x, y] = intensity
        
        # 시작점과 목표점 표시
        display_maze[self.start[0], self.start[1]] = 0.8  # 시작점 (밝은 회색)
        display_maze[self.goal[0], self.goal[1]] = 0.2   # 목표점 (어두운 회색)
        
        ax.imshow(display_maze, cmap='RdYlBu_r', vmin=0, vmax=1)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 격자 표시
        for i in range(self.demo_maze.shape[0] + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5)
        for j in range(self.demo_maze.shape[1] + 1):
            ax.axvline(j - 0.5, color='gray', linewidth=0.5)
        
        # 범례
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkblue', label='벽'),
            Patch(facecolor='yellow', label='통로'),
            Patch(facecolor='orange', label='해결 경로'),
            Patch(facecolor='lightgray', label='시작점'),
            Patch(facecolor='darkgray', label='목표점')
        ]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        return fig, ax
    
    def demo_dqn_training(self, episodes=100):
        """DQN 학습 시연"""
        print("\n🤖 DQN 학습 시연 시작...")
        print("=" * 50)
        
        solver = DQNSolver(
            episodes=episodes,
            max_steps=200,
            learning_rate=1e-2,
            batch_size=16,
            memory_size=1000,
            device='auto'
        )
        
        start_time = time.time()
        
        print(f"학습 설정:")
        print(f"  - 에피소드: {episodes}")
        print(f"  - 최대 스텝: 200")
        print(f"  - 미로 크기: {self.demo_maze.shape}")
        print(f"  - 시작점: {self.start}, 목표점: {self.goal}")
        
        # 학습 실행
        training_result = solver.train(self.demo_maze, self.start, self.goal)
        training_time = time.time() - start_time
        
        print(f"\n📊 DQN 학습 결과:")
        print(f"  - 학습 시간: {training_time:.1f}초")
        print(f"  - 최종 성공률: {training_result['final_success_rate']:.2%}")
        print(f"  - 평균 보상: {training_result['average_reward']:.2f}")
        
        # 해결 시연
        print(f"\n🎯 DQN 미로 해결 시연...")
        solve_start = time.time()
        path, solve_result = solver.solve(self.demo_maze, self.start, self.goal)
        solve_time = time.time() - solve_start
        
        print(f"  - 해결 시간: {solve_time:.2f}초")
        print(f"  - 성공 여부: {'✅ 성공' if solve_result['success'] else '❌ 실패'}")
        print(f"  - 경로 길이: {len(path) - 1}")
        print(f"  - 총 스텝: {solve_result['steps']}")
        
        self.results['DQN'] = {
            'training_time': training_time,
            'success_rate': training_result['final_success_rate'],
            'solve_time': solve_time,
            'path': path,
            'success': solve_result['success'],
            'path_length': len(path) - 1
        }
        
        return path, solve_result['success']
    
    def demo_ppo_training(self, timesteps=20000):
        """PPO 학습 시연"""
        print("\n🎯 PPO 학습 시연 시작...")
        print("=" * 50)
        
        solver = PPOSolver(
            total_timesteps=timesteps,
            max_episode_steps=200,
            learning_rate=1e-2,
            buffer_size=512,
            batch_size=64,
            device='auto'
        )
        
        start_time = time.time()
        
        print(f"학습 설정:")
        print(f"  - 총 타임스텝: {timesteps}")
        print(f"  - 최대 에피소드 스텝: 200")
        print(f"  - 미로 크기: {self.demo_maze.shape}")
        print(f"  - 시작점: {self.start}, 목표점: {self.goal}")
        
        # 학습 실행
        training_result = solver.train(self.demo_maze, self.start, self.goal)
        training_time = time.time() - start_time
        
        print(f"\n📊 PPO 학습 결과:")
        print(f"  - 학습 시간: {training_time:.1f}초")
        print(f"  - 최종 성공률: {training_result['final_success_rate']:.2%}")
        print(f"  - 총 에피소드: {training_result['total_episodes']}")
        print(f"  - 평균 보상: {training_result['average_reward']:.2f}")
        
        # 해결 시연
        print(f"\n🎯 PPO 미로 해결 시연...")
        solve_start = time.time()
        path, solve_result = solver.solve(self.demo_maze, self.start, self.goal)
        solve_time = time.time() - solve_start
        
        print(f"  - 해결 시간: {solve_time:.2f}초")
        print(f"  - 성공 여부: {'✅ 성공' if solve_result['success'] else '❌ 실패'}")
        print(f"  - 경로 길이: {len(path) - 1}")
        print(f"  - 총 스텝: {solve_result['steps']}")
        
        self.results['PPO'] = {
            'training_time': training_time,
            'success_rate': training_result['final_success_rate'],
            'solve_time': solve_time,
            'path': path,
            'success': solve_result['success'],
            'path_length': len(path) - 1
        }
        
        return path, solve_result['success']
    
    def demo_comparison(self):
        """알고리즘 비교 시연"""
        print("\n📊 알고리즘 성능 비교")
        print("=" * 50)
        
        if len(self.results) < 2:
            print("❌ 비교할 결과가 부족합니다.")
            return
        
        # 비교 표 출력
        print(f"{'알고리즘':<8} {'학습시간':<10} {'성공률':<10} {'해결시간':<10} {'경로길이':<10}")
        print("-" * 50)
        
        for algorithm, result in self.results.items():
            print(f"{algorithm:<8} {result['training_time']:<10.1f} "
                  f"{result['success_rate']:<10.2%} {result['solve_time']:<10.2f} "
                  f"{result['path_length']:<10}")
        
        # 시각화
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        algorithms = list(self.results.keys())
        colors = ['#FF6B6B', '#4ECDC4']
        
        # 성공률 비교
        success_rates = [self.results[alg]['success_rate'] for alg in algorithms]
        ax1.bar(algorithms, success_rates, color=colors[:len(algorithms)])
        ax1.set_title('성공률 비교')
        ax1.set_ylabel('성공률')
        ax1.set_ylim(0, 1)
        
        # 학습 시간 비교
        training_times = [self.results[alg]['training_time'] for alg in algorithms]
        ax2.bar(algorithms, training_times, color=colors[:len(algorithms)])
        ax2.set_title('학습 시간 비교')
        ax2.set_ylabel('시간 (초)')
        
        # 해결 시간 비교
        solve_times = [self.results[alg]['solve_time'] for alg in algorithms]
        ax3.bar(algorithms, solve_times, color=colors[:len(algorithms)])
        ax3.set_title('해결 시간 비교')
        ax3.set_ylabel('시간 (초)')
        
        # 경로 길이 비교
        path_lengths = [self.results[alg]['path_length'] for alg in algorithms]
        ax4.bar(algorithms, path_lengths, color=colors[:len(algorithms)])
        ax4.set_title('경로 길이 비교')
        ax4.set_ylabel('스텝 수')
        
        plt.tight_layout()
        plt.savefig('docs/demo_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # 최고 성능 알고리즘 발표
        best_success = max(self.results.items(), key=lambda x: x[1]['success_rate'])
        fastest_training = min(self.results.items(), key=lambda x: x[1]['training_time'])
        fastest_solve = min([item for item in self.results.items() if item[1]['success']], 
                           key=lambda x: x[1]['solve_time'])
        
        print(f"\n🏆 성능 우수상:")
        print(f"  - 최고 성공률: {best_success[0]} ({best_success[1]['success_rate']:.2%})")
        print(f"  - 최빠른 학습: {fastest_training[0]} ({fastest_training[1]['training_time']:.1f}초)")
        print(f"  - 최빠른 해결: {fastest_solve[0]} ({fastest_solve[1]['solve_time']:.2f}초)")
    
    def show_solution_paths(self):
        """해결 경로 시각화"""
        if not self.results:
            print("❌ 표시할 결과가 없습니다.")
            return
        
        n_algorithms = len(self.results)
        fig, axes = plt.subplots(1, n_algorithms + 1, figsize=(5 * (n_algorithms + 1), 5))
        
        if n_algorithms == 1:
            axes = [axes]
        
        # 원본 미로
        ax = axes[0]
        ax.imshow(self.demo_maze, cmap='RdYlBu_r')
        ax.set_title('원본 미로', fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 시작점과 목표점 표시
        ax.plot(self.start[1], self.start[0], 'go', markersize=10, label='시작점')
        ax.plot(self.goal[1], self.goal[0], 'ro', markersize=10, label='목표점')
        ax.legend()
        
        # 각 알고리즘의 해결 경로
        for i, (algorithm, result) in enumerate(self.results.items()):
            ax = axes[i + 1]
            
            if result['success'] and result['path']:
                self.visualize_maze_with_path(ax, result['path'], 
                                            f"{algorithm} 해결 경로\n길이: {result['path_length']}")
            else:
                self.visualize_maze_with_path(ax, [], f"{algorithm}\n해결 실패")
        
        plt.tight_layout()
        plt.savefig('docs/demo_solution_paths.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def visualize_maze_with_path(self, ax, path, title):
        """특정 축에 미로와 경로 시각화"""
        display_maze = self.demo_maze.copy().astype(float)
        
        # 경로 표시
        if path:
            for i, (x, y) in enumerate(path):
                if 0 <= x < self.demo_maze.shape[0] and 0 <= y < self.demo_maze.shape[1]:
                    intensity = 0.3 + 0.4 * (i / len(path))
                    display_maze[x, y] = intensity
        
        # 시작점과 목표점
        display_maze[self.start[0], self.start[1]] = 0.8
        display_maze[self.goal[0], self.goal[1]] = 0.2
        
        ax.imshow(display_maze, cmap='RdYlBu_r', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    def run_full_demo(self, quick_mode=False):
        """전체 시연 실행"""
        print("🎬 발표용 실시간 시연 시작!")
        print("=" * 60)
        
        # 시스템 상태 확인
        profiler = get_profiler()
        current_metrics = profiler.get_current_metrics()
        
        print(f"💻 시스템 상태:")
        print(f"  - VRAM: {current_metrics.vram_used_mb:.1f}MB")
        print(f"  - GPU 사용률: {current_metrics.gpu_percent:.1f}%")
        print(f"  - CPU 사용률: {current_metrics.cpu_percent:.1f}%")
        
        # 미로 표시
        print(f"\n🏷️  시연용 미로 ({self.demo_maze.shape[0]}x{self.demo_maze.shape[1]})")
        self.visualize_maze(title="시연용 미로")
        plt.savefig('docs/demo_maze.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # 알고리즘 시연
        if quick_mode:
            print("\n⚡ 빠른 모드로 실행합니다...")
            self.demo_dqn_training(episodes=50)
            self.demo_ppo_training(timesteps=10000)
        else:
            self.demo_dqn_training(episodes=100)
            self.demo_ppo_training(timesteps=20000)
        
        # 비교 및 결과
        self.demo_comparison()
        self.show_solution_paths()
        
        print("\n🎉 시연 완료!")
        print("생성된 파일들:")
        print("  - docs/demo_maze.png")
        print("  - docs/demo_comparison.png") 
        print("  - docs/demo_solution_paths.png")

def main():
    parser = argparse.ArgumentParser(description='발표용 실시간 시연')
    
    parser.add_argument('--quick', action='store_true',
                       help='빠른 모드 (짧은 학습)')
    parser.add_argument('--algorithm', type=str, choices=['dqn', 'ppo', 'both'],
                       default='both', help='시연할 알고리즘')
    
    args = parser.parse_args()
    
    demo = PresentationDemo()
    
    print("🎭 발표용 시연 도구")
    print("이 스크립트는 랩 세미나에서 실시간으로 알고리즘을 시연할 때 사용합니다.")
    
    if args.algorithm == 'dqn':
        demo.demo_dqn_training(50 if args.quick else 100)
        demo.show_solution_paths()
    elif args.algorithm == 'ppo':
        demo.demo_ppo_training(10000 if args.quick else 20000)
        demo.show_solution_paths()
    else:
        demo.run_full_demo(args.quick)

if __name__ == "__main__":
    main()