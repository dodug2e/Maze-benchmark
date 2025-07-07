#!/usr/bin/env python3
"""
A* Baseline 실험 스크립트
A* 알고리즘으로 모든 테스트 미로의 최적해를 구하고 baseline 성능을 측정
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.astar import AStarSolver, AStarResult
from utils.profiler import get_profiler, profile_execution
from utils.maze_io import get_loader, load_maze_as_array

class AStarBaseline:
    """A* 알고리즘 baseline 실험 클래스"""
    
    def __init__(self, results_dir: str = "results/baseline"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # A* 설정
        self.astar_solver = AStarSolver(diagonal_movement=False)
        self.profiler = get_profiler()
        
        # 결과 저장용
        self.results = {
            'optimal_solutions': {},
            'performance_metrics': {},
            'size_scaling': {},
            'memory_usage': {}
        }
    
    def run_single_maze(self, maze_data: Dict, maze_id: str) -> AStarResult:
        """단일 미로에 대한 A* 실행"""
        
        maze_array = maze_data['maze']
        metadata = maze_data['metadata']
        
        # 미로 데이터 변환: 생성기에서 0=wall, 1=path이므로 A*용으로 반전
        # A*에서는 0=path, 1=wall을 기대함
        converted_maze = 1 - maze_array  # 0과 1을 뒤바꿈
        
        # 시작점과 끝점 추출
        start = tuple(metadata.get('entrance', (0, 0)))
        goal = tuple(metadata.get('exit', (maze_array.shape[0]-1, maze_array.shape[1]-1)))
        
        # 좌표 순서 확인 (x,y vs row,col)
        # 메타데이터가 (x,y) 형식이라면 (row,col)로 변환
        if 'entrance' in metadata:
            entrance = metadata['entrance']
            start = (entrance[1], entrance[0]) if isinstance(entrance, list) else (entrance[0], entrance[1])
        
        if 'exit' in metadata:
            exit_point = metadata['exit']
            goal = (exit_point[1], exit_point[0]) if isinstance(exit_point, list) else (exit_point[0], exit_point[1])
        
        print(f"미로 {maze_id} 실행 중... (크기: {maze_array.shape}, 시작: {start}, 끝: {goal})")
        
        # 경계 체크
        rows, cols = converted_maze.shape
        if not (0 <= start[0] < rows and 0 <= start[1] < cols):
            print(f"❌ 시작점 {start}이 범위를 벗어남")
            result = AStarResult()
            result.solution_found = False
            result.failure_reason = f"Start point {start} out of bounds"
            return result
            
        if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
            print(f"❌ 끝점 {goal}이 범위를 벗어남")
            result = AStarResult()
            result.solution_found = False
            result.failure_reason = f"Goal point {goal} out of bounds"
            return result
        
        # 시작점과 끝점이 통로인지 확인
        if converted_maze[start[0], start[1]] != 0:
            print(f"❌ 시작점 {start}이 벽입니다 (값: {converted_maze[start[0], start[1]]})")
            # 시작점을 강제로 통로로 만들기
            converted_maze[start[0], start[1]] = 0
            print(f"✅ 시작점을 통로로 변경")
            
        if converted_maze[goal[0], goal[1]] != 0:
            print(f"❌ 끝점 {goal}이 벽입니다 (값: {converted_maze[goal[0], goal[1]]})")
            # 끝점을 강제로 통로로 만들기
            converted_maze[goal[0], goal[1]] = 0
            print(f"✅ 끝점을 통로로 변경")
        
        # 성능 모니터링 시작
        self.profiler.start_monitoring()
        start_time = time.time()
        
        try:
            # A* 실행 (변환된 미로 사용)
            result = self.astar_solver.solve(converted_maze, start, goal)
        except Exception as e:
            print(f"❌ A* 실행 중 오류: {e}")
            result = AStarResult()
            result.solution_found = False
            result.failure_reason = str(e)
        
        # 성능 측정 완료
        execution_time = time.time() - start_time
        self.profiler.stop_monitoring()
        performance_data = self.profiler.get_summary_stats()
        
        # 결과 보완
        result.maze_id = maze_id
        result.maze_size = maze_array.shape
        result.execution_time = execution_time
        result.vram_usage = performance_data.get('vram_used_mb', {}).get('peak', 0)
        result.cpu_utilization = performance_data.get('cpu_percent', {}).get('avg', 0)
        result.gpu_utilization = performance_data.get('gpu_percent', {}).get('avg', 0)
        
        if result.solution_found:
            print(f"✅ 성공! 경로 길이: {result.solution_length}, 실행시간: {execution_time:.3f}초")
        else:
            print(f"❌ 실패: {result.failure_reason}")
        
        return result
    
    def run_size_scaling_test(self, mazes_by_size: Dict[str, List]):
        """미로 크기별 성능 스케일링 테스트"""
        
        print("\n=== 미로 크기별 스케일링 테스트 ===")
        
        size_results = {}
        
        for size_category, maze_list in mazes_by_size.items():
            print(f"\n크기 카테고리: {size_category}")
            
            category_results = []
            
            for maze_data, maze_id in maze_list[:5]:  # 각 크기별로 5개씩만 테스트
                result = self.run_single_maze(maze_data, maze_id)
                category_results.append({
                    'maze_id': maze_id,
                    'execution_time': result.execution_time,
                    'solution_length': result.solution_length,
                    'success': result.solution_found,
                    'total_cells': result.maze_size[0] * result.maze_size[1]
                })
            
            # 카테고리별 통계 계산
            successful_results = [r for r in category_results if r['success']]
            
            if successful_results:
                size_results[size_category] = {
                    'avg_execution_time': np.mean([r['execution_time'] for r in successful_results]),
                    'avg_solution_length': np.mean([r['solution_length'] for r in successful_results]),
                    'success_rate': len(successful_results) / len(category_results),
                    'avg_cells': np.mean([r['total_cells'] for r in category_results]),
                    'sample_count': len(category_results)
                }
        
        self.results['size_scaling'] = size_results
        return size_results
    
    def run_baseline_experiment(self, dataset_subset: str = "test", sample_limit: int = 100):
        """전체 baseline 실험 실행"""
        
        print("=== A* Baseline 실험 시작 ===")
        
        # 데이터셋 로더 초기화
        print(f"데이터셋 로더 초기화 중...")
        loader = get_loader()
        
        # 테스트 샘플 ID 목록 가져오기
        sample_ids = loader.get_sample_ids(dataset_subset)
        
        if not sample_ids:
            print("❌ 샘플 ID를 찾을 수 없습니다")
            return
        
        # 샘플 수 제한
        if len(sample_ids) > sample_limit:
            sample_ids = sample_ids[:sample_limit]
        
        print(f"✅ {len(sample_ids)} 개 미로 로드 완료")
        
        # 미로 데이터 변환
        mazes = {}
        for sample_id in sample_ids:
            try:
                maze_array, metadata = load_maze_as_array(sample_id, dataset_subset)
                mazes[sample_id] = {
                    'maze': maze_array,
                    'metadata': metadata
                }
            except Exception as e:
                print(f"❌ 미로 {sample_id} 로드 실패: {e}")
                continue
        
        # 크기별 분류
        mazes_by_size = self._categorize_by_size(mazes)
        
        # 1. 전체 미로 실행
        print("\n=== 전체 미로 A* 실행 ===")
        all_results = []
        optimal_solutions = {}
        
        for i, (maze_id, maze_data) in enumerate(mazes.items()):
            if i >= sample_limit:
                break
                
            result = self.run_single_maze(maze_data, maze_id)
            all_results.append(result)
            
            # 최적해 저장
            if result.solution_found:
                optimal_solutions[maze_id] = {
                    'optimal_length': result.solution_length,
                    'optimal_path': result.path,
                    'execution_time': result.execution_time
                }
            
            # 진행상황 출력
            if (i + 1) % 10 == 0:
                success_count = sum(1 for r in all_results if r.solution_found)
                print(f"진행: {i+1}/{min(len(mazes), sample_limit)} "
                      f"(성공률: {success_count/(i+1)*100:.1f}%)")
        
        # 2. 크기별 스케일링 테스트
        size_scaling = self.run_size_scaling_test(mazes_by_size)
        
        # 3. 전체 통계 계산
        self._calculate_overall_statistics(all_results)
        
        # 4. 결과 저장
        self._save_results(optimal_solutions)
        
        # 5. 시각화
        self._create_visualizations()
        
        print("\n=== A* Baseline 실험 완료 ===")
        self._print_summary()
    
    def _categorize_by_size(self, mazes: Dict) -> Dict[str, List]:
        """미로를 크기별로 분류"""
        categories = {
            'small': [],    # 50x50 이하
            'medium': [],   # 51x51 ~ 100x100
            'large': [],    # 101x101 ~ 150x150
            'xlarge': []    # 151x151 이상
        }
        
        for maze_id, maze_data in mazes.items():
            size = maze_data['maze'].shape[0]
            
            if size <= 50:
                categories['small'].append((maze_data, maze_id))
            elif size <= 100:
                categories['medium'].append((maze_data, maze_id))
            elif size <= 150:
                categories['large'].append((maze_data, maze_id))
            else:
                categories['xlarge'].append((maze_data, maze_id))
        
        return categories
    
    def _calculate_overall_statistics(self, results: List[AStarResult]):
        """전체 통계 계산"""
        
        successful_results = [r for r in results if r.solution_found]
        
        if not successful_results:
            print("❌ 성공한 결과가 없습니다.")
            return
        
        stats = {
            'total_mazes': len(results),
            'successful_mazes': len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'avg_execution_time': np.mean([r.execution_time for r in successful_results]),
            'std_execution_time': np.std([r.execution_time for r in successful_results]),
            'avg_solution_length': np.mean([r.solution_length for r in successful_results]),
            'std_solution_length': np.std([r.solution_length for r in successful_results]),
            'min_execution_time': min(r.execution_time for r in successful_results),
            'max_execution_time': max(r.execution_time for r in successful_results),
            'avg_vram_usage': np.mean([r.vram_usage for r in results if r.vram_usage > 0]),
            'avg_cpu_usage': np.mean([r.cpu_utilization for r in results if r.cpu_utilization > 0])
        }
        
        self.results['performance_metrics'] = stats
    
    def _save_results(self, optimal_solutions: Dict):
        """결과 저장"""
        
        # 최적해 저장
        optimal_path = self.results_dir / "optimal_solutions.json"
        with open(optimal_path, 'w') as f:
            # NumPy 배열을 리스트로 변환
            serializable_solutions = {}
            for maze_id, solution in optimal_solutions.items():
                serializable_solutions[maze_id] = {
                    'optimal_length': solution['optimal_length'],
                    'optimal_path': [list(point) for point in solution['optimal_path']],
                    'execution_time': solution['execution_time']
                }
            json.dump(serializable_solutions, f, indent=2)
        
        # 전체 결과 저장
        results_path = self.results_dir / "astar_baseline_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"✅ 결과 저장: {results_path}")
        print(f"✅ 최적해 저장: {optimal_path}")
    
    def _create_visualizations(self):
        """결과 시각화"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 크기별 실행시간
        if 'size_scaling' in self.results:
            size_data = self.results['size_scaling']
            sizes = list(size_data.keys())
            times = [size_data[s]['avg_execution_time'] for s in sizes]
            
            axes[0,0].bar(sizes, times)
            axes[0,0].set_title('미로 크기별 평균 실행시간')
            axes[0,0].set_ylabel('실행시간 (초)')
        
        # 2. 크기별 성공률
        if 'size_scaling' in self.results:
            success_rates = [size_data[s]['success_rate'] * 100 for s in sizes]
            
            axes[0,1].bar(sizes, success_rates)
            axes[0,1].set_title('미로 크기별 성공률')
            axes[0,1].set_ylabel('성공률 (%)')
            axes[0,1].set_ylim(0, 100)
        
        # 3. 셀 개수 vs 실행시간 상관관계
        if 'size_scaling' in self.results:
            cell_counts = [size_data[s]['avg_cells'] for s in sizes]
            
            axes[1,0].scatter(cell_counts, times)
            axes[1,0].set_title('미로 크기 vs 실행시간')
            axes[1,0].set_xlabel('총 셀 개수')
            axes[1,0].set_ylabel('실행시간 (초)')
        
        # 4. A* 성능 요약
        if 'performance_metrics' in self.results:
            metrics = self.results['performance_metrics']
            labels = ['성공률', '평균 실행시간', '평균 경로길이']
            values = [
                metrics['success_rate'] * 100,
                metrics['avg_execution_time'],
                metrics['avg_solution_length'] / 100  # 스케일 조정
            ]
            
            axes[1,1].bar(labels, values)
            axes[1,1].set_title('A* 전체 성능 요약')
        
        plt.tight_layout()
        
        # 저장
        viz_path = self.results_dir / "astar_baseline_analysis.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 시각화 저장: {viz_path}")
    
    def _print_summary(self):
        """결과 요약 출력"""
        
        if 'performance_metrics' not in self.results:
            return
        
        metrics = self.results['performance_metrics']
        
        print(f"""
🎯 A* Baseline 실험 결과 요약

📊 전체 성능:
   • 총 미로 수: {metrics['total_mazes']}
   • 성공 미로 수: {metrics['successful_mazes']}
   • 성공률: {metrics['success_rate']*100:.1f}%

⏱️ 실행 성능:
   • 평균 실행시간: {metrics['avg_execution_time']:.3f}초 (±{metrics['std_execution_time']:.3f})
   • 최단 실행시간: {metrics['min_execution_time']:.3f}초
   • 최장 실행시간: {metrics['max_execution_time']:.3f}초

🛤️ 경로 품질:
   • 평균 경로 길이: {metrics['avg_solution_length']:.1f} (±{metrics['std_solution_length']:.1f})

💻 시스템 자원:
   • 평균 VRAM 사용량: {metrics.get('avg_vram_usage', 0):.1f}MB
   • 평균 CPU 사용률: {metrics.get('avg_cpu_usage', 0):.1f}%

🚀 이제 강화학습 모델들을 이 baseline과 비교해보세요!
   python scripts/train.py algo=DQN --baseline=astar
   python scripts/train.py algo=PPO --baseline=astar
        """)


def main():
    """메인 실행 함수"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='A* Baseline 실험')
    parser.add_argument('--subset', type=str, default='test', 
                       help='데이터셋 분할 (train/valid/test)')
    parser.add_argument('--output', type=str, default='results/baseline',
                       help='결과 저장 경로')
    parser.add_argument('--samples', type=int, default=100,
                       help='테스트할 미로 개수')
    
    args = parser.parse_args()
    
    # Baseline 실험 실행
    baseline = AStarBaseline(results_dir=args.output)
    baseline.run_baseline_experiment(args.subset, args.samples)


if __name__ == "__main__":
    main()