#!/usr/bin/env python3
"""
벤치마크 결과 시각화 스크립트
Usage: python scripts/visualize.py --results-dir results --output-dir docs
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_results(results_dir: str) -> Dict[str, List[Dict]]:
    """결과 파일들 로드"""
    results = {}
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return results
    
    # JSON 파일들 찾기
    json_files = list(results_path.glob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # 파일명에서 알고리즘 이름 추출
            filename = json_file.stem
            parts = filename.split('_')
            algo_name = parts[0] if parts else 'unknown'
            
            if algo_name not in results:
                results[algo_name] = []
            
            results[algo_name].append(data)
            
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return results

def create_performance_comparison(results: Dict[str, List[Dict]], output_dir: str):
    """성능 비교 차트 생성"""
    algorithms = []
    success_rates = []
    avg_times = []
    avg_solution_lengths = []
    
    for algo_name, algo_results in results.items():
        if not algo_results:
            continue
            
        # 최신 결과 사용
        latest_result = max(algo_results, key=lambda x: x.get('timestamp', 0))
        
        algorithms.append(algo_name)
        
        # 성공률 계산
        total_samples = latest_result.get('total_samples', 0)
        successful_runs = latest_result.get('successful_runs', 0)
        success_rate = (successful_runs / total_samples * 100) if total_samples > 0 else 0
        success_rates.append(success_rate)
        
        # 평균 실행 시간
        avg_time = latest_result.get('average_execution_time', 0)
        avg_times.append(avg_time)
        
        # 평균 해결 경로 길이
        avg_length = latest_result.get('average_solution_length', 0)
        avg_solution_lengths.append(avg_length)
    
    # 3개 서브플롯 생성
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 성공률 차트
    bars1 = axes[0].bar(algorithms, success_rates, color='skyblue', alpha=0.7)
    axes[0].set_title('Algorithm Success Rate (%)')
    axes[0].set_ylabel('Success Rate (%)')
    axes[0].set_ylim(0, 100)
    
    # 값 표시
    for bar, rate in zip(bars1, success_rates):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
    
    # 평균 실행 시간 차트
    bars2 = axes[1].bar(algorithms, avg_times, color='lightgreen', alpha=0.7)
    axes[1].set_title('Average Execution Time (seconds)')
    axes[1].set_ylabel('Time (s)')
    
    # 값 표시
    for bar, time_val in zip(bars2, avg_times):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{time_val:.2f}s', ha='center', va='bottom')
    
    # 평균 해결 경로 길이 차트
    bars3 = axes[2].bar(algorithms, avg_solution_lengths, color='lightcoral', alpha=0.7)
    axes[2].set_title('Average Solution Path Length')
    axes[2].set_ylabel('Path Length')
    
    # 값 표시
    for bar, length in zip(bars3, avg_solution_lengths):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{length:.1f}', ha='center', va='bottom')
    
    # X축 라벨 회전
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance comparison saved to {output_dir}/performance_comparison.png")

def create_resource_usage_chart(results: Dict[str, List[Dict]], output_dir: str):
    """리소스 사용량 차트 생성"""
    # 프로파일링 결과가 있는 경우만 처리
    profiling_results = {}
    
    for algo_name, algo_results in results.items():
        for result in algo_results:
            if 'performance_metrics' in result:
                profiling_results[algo_name] = result['performance_metrics']
                break
    
    if not profiling_results:
        print("No profiling results found for resource usage chart")
        return
    
    algorithms = list(profiling_results.keys())
    vram_peak = []
    gpu_avg = []
    cpu_avg = []
    
    for algo_name in algorithms:
        metrics = profiling_results[algo_name]
        
        # VRAM 피크 사용량
        vram_stats = metrics.get('vram_used_mb', {})
        vram_peak.append(vram_stats.get('peak', 0))
        
        # GPU 평균 사용률
        gpu_stats = metrics.get('gpu_percent', {})
        gpu_avg.append(gpu_stats.get('avg', 0))
        
        # CPU 평균 사용률
        cpu_stats = metrics.get('cpu_percent', {})
        cpu_avg.append(cpu_stats.get('avg', 0))
    
    # 2개 서브플롯 생성
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # VRAM 사용량 차트
    bars1 = axes[0].bar(algorithms, vram_peak, color='orange', alpha=0.7)
    axes[0].set_title('Peak VRAM Usage (MB)')
    axes[0].set_ylabel('VRAM (MB)')
    axes[0].axhline(y=6144, color='red', linestyle='--', label='RTX 3060 Limit (6GB)')
    axes[0].legend()
    
    # 값 표시
    for bar, vram in zip(bars1, vram_peak):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{vram:.0f}MB', ha='center', va='bottom')
    
    # GPU/CPU 사용률 차트
    x = np.arange(len(algorithms))
    width = 0.35
    
    bars2 = axes[1].bar(x - width/2, gpu_avg, width, label='GPU (%)', color='green', alpha=0.7)
    bars3 = axes[1].bar(x + width/2, cpu_avg, width, label='CPU (%)', color='blue', alpha=0.7)
    
    axes[1].set_title('Average GPU/CPU Utilization (%)')
    axes[1].set_ylabel('Utilization (%)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(algorithms)
    axes[1].legend()
    
    # 값 표시
    for bar, val in zip(bars2, gpu_avg):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom')
    
    for bar, val in zip(bars3, cpu_avg):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom')
    
    # X축 라벨 회전
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/resource_usage.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Resource usage chart saved to {output_dir}/resource_usage.png")

def create_algorithm_comparison_table(results: Dict[str, List[Dict]], output_dir: str):
    """알고리즘 비교 테이블 생성"""
    table_data = []
    
    for algo_name, algo_results in results.items():
        if not algo_results:
            continue
            
        # 최신 결과 사용
        latest_result = max(algo_results, key=lambda x: x.get('timestamp', 0))
        
        # 성능 메트릭 추출
        total_samples = latest_result.get('total_samples', 0)
        successful_runs = latest_result.get('successful_runs', 0)
        success_rate = (successful_runs / total_samples * 100) if total_samples > 0 else 0
        
        avg_time = latest_result.get('average_execution_time', 0)
        avg_length = latest_result.get('average_solution_length', 0)
        
        # 프로파일링 데이터
        perf_metrics = latest_result.get('performance_metrics', {})
        vram_peak = perf_metrics.get('vram_used_mb', {}).get('peak', 0)
        gpu_avg = perf_metrics.get('gpu_percent', {}).get('avg', 0)
        
        table_data.append({
            'Algorithm': algo_name,
            'Success Rate (%)': f"{success_rate:.1f}",
            'Avg Time (s)': f"{avg_time:.2f}",
            'Avg Path Length': f"{avg_length:.1f}",
            'Peak VRAM (MB)': f"{vram_peak:.0f}",
            'Avg GPU (%)': f"{gpu_avg:.1f}"
        })
    
    # 테이블을 HTML로 생성
    html_content = """
    <html>
    <head>
        <title>Algorithm Comparison Results</title>
        <style>
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: center; }
            th { background-color: #f2f2f2; font-weight: bold; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .best { background-color: #d4edda; font-weight: bold; }
            .worst { background-color: #f8d7da; }
        </style>
    </head>
    <body>
        <h1>Maze Solving Algorithm Benchmark Results</h1>
        <p>Generated on: {timestamp}</p>
        
        <h2>Performance Comparison</h2>
        <table>
            <tr>
                <th>Algorithm</th>
                <th>Success Rate (%)</th>
                <th>Avg Time (s)</th>
                <th>Avg Path Length</th>
                <th>Peak VRAM (MB)</th>
                <th>Avg GPU (%)</th>
            </tr>
    """.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    for row in table_data:
        html_content += "<tr>"
        for key, value in row.items():
            html_content += f"<td>{value}</td>"
        html_content += "</tr>"
    
    html_content += """
        </table>
        
        <h2>Notes</h2>
        <ul>
            <li>Tests performed on RTX 3060 (6GB VRAM)</li>
            <li>Higher success rate and lower execution time are better</li>
            <li>Lower path length indicates more efficient solutions</li>
            <li>VRAM usage should stay below 6GB limit</li>
        </ul>
    </body>
    </html>
    """
    
    # HTML 파일 저장
    with open(f"{output_dir}/comparison_table.html", 'w') as f:
        f.write(html_content)
    
    print(f"Comparison table saved to {output_dir}/comparison_table.html")

def generate_summary_report(results: Dict[str, List[Dict]], output_dir: str):
    """요약 보고서 생성"""
    report_content = f"""
# Maze Solving Algorithm Benchmark Report

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Test Environment
- Hardware: RTX 3060 (6GB VRAM)
- Framework: Python + PyTorch
- Dataset: Procedurally generated mazes (50x50 - 200x200)

## Algorithm Results Summary

"""
    
    for algo_name, algo_results in results.items():
        if not algo_results:
            continue
            
        latest_result = max(algo_results, key=lambda x: x.get('timestamp', 0))
        
        total_samples = latest_result.get('total_samples', 0)
        successful_runs = latest_result.get('successful_runs', 0)
        success_rate = (successful_runs / total_samples * 100) if total_samples > 0 else 0
        
        report_content += f"""
### {algo_name}
- Success Rate: {success_rate:.1f}%
- Average Execution Time: {latest_result.get('average_execution_time', 0):.2f}s
- Average Solution Length: {latest_result.get('average_solution_length', 0):.1f}
- Samples Processed: {total_samples}

"""
    
    report_content += """
## Performance Analysis

The benchmark results show the comparative performance of different maze-solving algorithms:

1. **Success Rate**: Percentage of mazes successfully solved
2. **Execution Time**: Average time to solve a maze
3. **Solution Quality**: Average path length (shorter is better)
4. **Resource Usage**: VRAM and GPU utilization

## Future Work

- Extension to Isaac Sim for 3D maze environments
- Additional algorithm implementations
- Larger dataset generation and testing
- Real-time performance optimization

---
*Generated by Maze Benchmark Framework v2.0-Lite*
"""
    
    # 마크다운 파일 저장
    with open(f"{output_dir}/benchmark_report.md", 'w') as f:
        f.write(report_content)
    
    print(f"Summary report saved to {output_dir}/benchmark_report.md")

def main():
    parser = argparse.ArgumentParser(description='Visualize benchmark results')
    parser.add_argument('--results-dir', default='results', 
                       help='Directory containing result JSON files')
    parser.add_argument('--output-dir', default='docs', 
                       help='Output directory for visualizations')
    parser.add_argument('--format', choices=['png', 'pdf', 'svg'], default='png',
                       help='Output format for charts')
    
    args = parser.parse_args()
    
    # 출력 디렉터리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 결과 로드
    results = load_results(args.results_dir)
    
    if not results:
        print("No results found to visualize")
        return
    
    print(f"Found results for {len(results)} algorithms")
    
    # 시각화 생성
    create_performance_comparison(results, args.output_dir)
    create_resource_usage_chart(results, args.output_dir)
    create_algorithm_comparison_table(results, args.output_dir)
    generate_summary_report(results, args.output_dir)
    
    print(f"\nAll visualizations saved to {args.output_dir}/")
    print("Files generated:")
    print("- performance_comparison.png")
    print("- resource_usage.png") 
    print("- comparison_table.html")
    print("- benchmark_report.md")

if __name__ == "__main__":
    main()