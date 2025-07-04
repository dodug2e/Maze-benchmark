#!/usr/bin/env python3
"""
발표용 결과 분석 스크립트
DQN, PPO 실험 결과를 종합 분석하여 발표 자료 생성
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Any
from datetime import datetime

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Malgun Gothic']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PresentationAnalyzer:
    """발표용 결과 분석기"""
    
    def __init__(self, output_dir: str = "docs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 분석 결과 저장용
        self.analysis_results = {
            'summary': {},
            'detailed_comparison': {},
            'performance_metrics': {},
            'recommendations': []
        }
        
        # 시각화 스타일 설정
        sns.set_style("whitegrid")
        plt.style.use('seaborn-v0_8')
    
    def load_experiment_results(self, results_dirs: Dict[str, str]) -> Dict[str, List[Dict]]:
        """실험 결과 로드"""
        all_results = {}
        
        for algorithm, results_dir in results_dirs.items():
            results_path = Path(results_dir)
            algorithm_results = []
            
            if results_path.exists():
                # JSON 파일들 수집
                for json_file in results_path.glob("*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            data['source_file'] = str(json_file)
                            algorithm_results.append(data)
                    except Exception as e:
                        logger.warning(f"파일 로드 실패 {json_file}: {e}")
                
                logger.info(f"{algorithm}: {len(algorithm_results)}개 결과 로드")
            else:
                logger.warning(f"{algorithm} 결과 디렉터리 없음: {results_path}")
            
            all_results[algorithm] = algorithm_results
        
        return all_results
    
    def analyze_success_rates(self, results: Dict[str, List[Dict]]) -> Dict:
        """성공률 분석"""
        success_analysis = {}
        
        for algorithm, data_list in results.items():
            if not data_list:
                continue
                
            successes = []
            
            for data in data_list:
                if 'success' in data:
                    successes.append(data['success'])
                elif 'summary' in data and 'success_rate' in data['summary']:
                    # 배치 결과인 경우
                    successes.append(data['summary']['success_rate'])
                elif 'individual_results' in data:
                    # 개별 결과들 추출
                    for result in data['individual_results']:
                        successes.append(result.get('success', False))
            
            if successes:
                success_analysis[algorithm] = {
                    'total_trials': len(successes),
                    'successful_trials': sum(successes),
                    'success_rate': np.mean(successes),
                    'std_deviation': np.std(successes),
                    'success_list': successes
                }
        
        return success_analysis
    
    def analyze_performance_metrics(self, results: Dict[str, List[Dict]]) -> Dict:
        """성능 지표 분석"""
        performance_analysis = {}
        
        for algorithm, data_list in results.items():
            if not data_list:
                continue
            
            metrics = {
                'execution_times': [],
                'solution_lengths': [],
                'vram_usage': [],
                'gpu_utilization': [],
                'training_times': []
            }
            
            for data in data_list:
                # 단일 결과
                if 'execution_time' in data:
                    metrics['execution_times'].append(data['execution_time'])
                
                if 'solution_length' in data:
                    metrics['solution_lengths'].append(data['solution_length'])
                
                if 'performance' in data:
                    perf = data['performance']
                    metrics['vram_usage'].append(perf.get('vram_usage', 0))
                    metrics['gpu_utilization'].append(perf.get('gpu_utilization', 0))
                
                # 배치 결과
                if 'individual_results' in data:
                    for result in data['individual_results']:
                        metrics['execution_times'].append(result.get('execution_time', 0))
                        metrics['solution_lengths'].append(result.get('solution_length', 0))
                        metrics['vram_usage'].append(result.get('vram_usage', 0))
            
            # 통계 계산
            performance_analysis[algorithm] = {}
            for metric_name, values in metrics.items():
                if values:
                    performance_analysis[algorithm][metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values),
                        'values': values
                    }
        
        return performance_analysis
    
    def create_success_rate_comparison(self, success_analysis: Dict) -> str:
        """성공률 비교 차트 생성"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        algorithms = list(success_analysis.keys())
        success_rates = [success_analysis[alg]['success_rate'] for alg in algorithms]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        # 성공률 막대 그래프
        bars = ax1.bar(algorithms, success_rates, color=colors[:len(algorithms)], alpha=0.8)
        ax1.set_title('알고리즘별 성공률 비교', fontsize=14, fontweight='bold')
        ax1.set_ylabel('성공률')
        ax1.set_ylim(0, 1)
        
        # 값 표시
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.2%}', ha='center', va='bottom', fontweight='bold')
        
        # 성공률 분포 박스플롯
        success_distributions = []
        labels = []
        
        for alg in algorithms:
            if success_analysis[alg]['success_list']:
                success_distributions.append(success_analysis[alg]['success_list'])
                labels.append(alg)
        
        if success_distributions:
            ax2.boxplot(success_distributions, labels=labels)
            ax2.set_title('성공률 분포', fontsize=14, fontweight='bold')
            ax2.set_ylabel('성공률')
        
        plt.tight_layout()
        save_path = self.output_dir / "success_rate_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"성공률 비교 차트 저장: {save_path}")
        return str(save_path)
    
    def create_performance_comparison(self, performance_analysis: Dict) -> str:
        """성능 지표 비교 차트 생성"""
        algorithms = list(performance_analysis.keys())
        
        if len(algorithms) < 2:
            logger.warning("비교할 알고리즘이 부족합니다.")
            return ""
        
        # 메트릭별 비교
        metrics_to_plot = ['execution_times', 'solution_lengths', 'vram_usage']
        metric_labels = ['실행 시간 (초)', '해결 경로 길이', 'VRAM 사용량 (MB)']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
            ax = axes[i]
            
            means = []
            stds = []
            alg_names = []
            
            for alg in algorithms:
                if metric in performance_analysis[alg]:
                    means.append(performance_analysis[alg][metric]['mean'])
                    stds.append(performance_analysis[alg][metric]['std'])
                    alg_names.append(alg)
            
            if means:
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(alg_names)]
                bars = ax.bar(alg_names, means, yerr=stds, capsize=5, 
                             color=colors, alpha=0.8)
                
                ax.set_title(f'{label} 비교', fontsize=12, fontweight='bold')
                ax.set_ylabel(label)
                
                # 값 표시
                for bar, mean, std in zip(bars, means, stds):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                           f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        save_path = self.output_dir / "performance_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"성능 비교 차트 저장: {save_path}")
        return str(save_path)
    
    def create_learning_curves_comparison(self, results: Dict[str, List[Dict]]) -> str:
        """학습 곡선 비교"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        colors = {'DQN': '#FF6B6B', 'PPO': '#4ECDC4', 'ACO': '#45B7D1'}
        
        for algorithm, data_list in results.items():
            color = colors.get(algorithm, '#666666')
            
            for data in data_list:
                learning_curves = data.get('learning_curves', {})
                
                # 에피소드 보상
                if 'rewards' in learning_curves:
                    episodes = learning_curves.get('episodes', range(len(learning_curves['rewards'])))
                    rewards = learning_curves['rewards']
                    
                    # 이동 평균 계산
                    window = min(50, len(rewards) // 10)
                    if window > 1:
                        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                        ax1.plot(episodes[window-1:], moving_avg, 
                               label=f'{algorithm} (이동평균)', color=color, linewidth=2)
                    else:
                        ax1.plot(episodes, rewards, label=algorithm, color=color, alpha=0.7)
                
                # 성공률
                if 'success_rates' in learning_curves:
                    episodes = range(len(learning_curves['success_rates']))
                    ax2.plot(episodes, learning_curves['success_rates'], 
                           label=algorithm, color=color, linewidth=2)
                
                # 정책 손실 (PPO만)
                if algorithm == 'PPO' and 'policy_losses' in learning_curves:
                    updates = range(len(learning_curves['policy_losses']))
                    ax3.plot(updates, learning_curves['policy_losses'], 
                           label='정책 손실', color=color, linewidth=2)
                
                # 가치 손실
                if 'value_losses' in learning_curves:
                    updates = range(len(learning_curves['value_losses']))
                    ax4.plot(updates, learning_curves['value_losses'], 
                           label=f'{algorithm} 가치 손실', color=color, linewidth=2)
        
        # 축 설정
        ax1.set_title('에피소드 보상 변화', fontsize=12, fontweight='bold')
        ax1.set_xlabel('에피소드')
        ax1.set_ylabel('보상')
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_title('성공률 변화', fontsize=12, fontweight='bold')
        ax2.set_xlabel('에피소드')
        ax2.set_ylabel('성공률')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True)
        
        ax3.set_title('정책 손실 변화 (PPO)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('업데이트')
        ax3.set_ylabel('정책 손실')
        ax3.legend()
        ax3.grid(True)
        
        ax4.set_title('가치 손실 변화', fontsize=12, fontweight='bold')
        ax4.set_xlabel('업데이트')
        ax4.set_ylabel('가치 손실')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        save_path = self.output_dir / "learning_curves_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"학습 곡선 비교 저장: {save_path}")
        return str(save_path)
    
    def generate_presentation_summary(self, success_analysis: Dict, 
                                    performance_analysis: Dict) -> str:
        """발표용 요약 보고서 생성"""
        
        summary_content = f"""# 🎯 강화학습 미로 탐색 알고리즘 비교 연구

> **발표자**: 기계공학과 3학년  
> **일시**: {datetime.now().strftime('%Y년 %m월 %d일')}  
> **주제**: DQN vs PPO 미로 해결 성능 비교

---

## 📊 핵심 결과 요약

### 성공률 비교
"""
        
        # 성공률 표
        for algorithm, analysis in success_analysis.items():
            summary_content += f"- **{algorithm}**: {analysis['success_rate']:.1%} ({analysis['successful_trials']}/{analysis['total_trials']})\n"
        
        summary_content += "\n### 성능 지표 비교\n\n"
        
        # 성능 지표 표
        summary_content += "| 알고리즘 | 평균 실행시간 | 평균 경로길이 | 평균 VRAM |\n"
        summary_content += "|---------|-------------|-------------|----------|\n"
        
        for algorithm, metrics in performance_analysis.items():
            exec_time = metrics.get('execution_times', {}).get('mean', 0)
            path_length = metrics.get('solution_lengths', {}).get('mean', 0)
            vram = metrics.get('vram_usage', {}).get('mean', 0)
            
            summary_content += f"| {algorithm} | {exec_time:.2f}초 | {path_length:.1f} | {vram:.0f}MB |\n"
        
        # 권장사항 및 결론
        summary_content += f"""

---

## 🎯 주요 발견사항

### 1. 알고리즘 특성 비교
- **DQN**: Value-based 강화학습, Experience Replay 활용
- **PPO**: Policy-based 강화학습, 안정적인 정책 개선

### 2. 성능 분석
"""
        
        # 최고 성능 알고리즘 찾기
        best_success = max(success_analysis.items(), key=lambda x: x[1]['success_rate'])
        summary_content += f"- **성공률 최고**: {best_success[0]} ({best_success[1]['success_rate']:.1%})\n"
        
        if performance_analysis:
            fastest_alg = min(performance_analysis.items(), 
                            key=lambda x: x[1].get('execution_times', {}).get('mean', float('inf')))
            summary_content += f"- **실행속도 최고**: {fastest_alg[0]} ({fastest_alg[1].get('execution_times', {}).get('mean', 0):.2f}초)\n"
        
        summary_content += """

### 3. RTX 3060 최적화 효과
- 배치 크기 조정으로 VRAM 효율성 개선
- 미로 크기별 동적 설정으로 안정성 확보

---

## 🚀 향후 계획

### 단기 목표
1. **하이퍼파라미터 최적화**: Grid Search 적용
2. **더 복잡한 미로**: 동적 장애물, 다중 목표
3. **앙상블 방법**: DQN + PPO 하이브리드

### 장기 목표
1. **Isaac Sim 통합**: 3D 시뮬레이션 환경 확장
2. **실제 로봇 적용**: 네비게이션 시스템 개발
3. **산업 응용**: 창고 자동화, AGV 경로 계획

---

## 📋 발표 체크리스트

### 기술적 내용
- [ ] DQN, PPO 알고리즘 원리 설명 (3분)
- [ ] 실험 설정 및 환경 소개 (2분)
- [ ] 성능 비교 결과 발표 (4분)
- [ ] 결론 및 향후 계획 (1분)

### 시연 준비
- [ ] 학습 과정 동영상
- [ ] 미로 해결 시연
- [ ] 성능 그래프 실시간 표시

### 질문 대비
- [ ] 왜 강화학습을 선택했는가?
- [ ] DQN과 PPO의 차이점은?
- [ ] 실제 로봇에 어떻게 적용할 것인가?
- [ ] Isaac Sim 확장 계획의 구체적 내용?

---

**🎉 성공적인 발표를 위해 화이팅!**
"""
        
        # 파일 저장
        summary_path = self.output_dir / "presentation_summary.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        logger.info(f"발표 요약 저장: {summary_path}")
        return str(summary_path)
    
    def create_comparison_table(self, success_analysis: Dict, 
                              performance_analysis: Dict) -> str:
        """상세 비교 표 생성"""
        
        # 데이터프레임 생성
        comparison_data = []
        
        for algorithm in success_analysis.keys():
            row = {'알고리즘': algorithm}
            
            # 성공률 데이터
            success_data = success_analysis[algorithm]
            row['성공률'] = f"{success_data['success_rate']:.1%}"
            row['성공_시행'] = f"{success_data['successful_trials']}/{success_data['total_trials']}"
            
            # 성능 데이터
            if algorithm in performance_analysis:
                perf_data = performance_analysis[algorithm]
                
                if 'execution_times' in perf_data:
                    row['평균_실행시간'] = f"{perf_data['execution_times']['mean']:.2f}초"
                
                if 'solution_lengths' in perf_data:
                    row['평균_경로길이'] = f"{perf_data['solution_lengths']['mean']:.1f}"
                
                if 'vram_usage' in perf_data:
                    row['평균_VRAM'] = f"{perf_data['vram_usage']['mean']:.0f}MB"
                
                if 'gpu_utilization' in perf_data:
                    row['평균_GPU사용률'] = f"{perf_data['gpu_utilization']['mean']:.1f}%"
            
            comparison_data.append(row)
        
        # CSV 저장
        df = pd.DataFrame(comparison_data)
        csv_path = self.output_dir / "detailed_comparison.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"상세 비교표 저장: {csv_path}")
        return str(csv_path)
    
    def run_complete_analysis(self, results_dirs: Dict[str, str]) -> Dict[str, str]:
        """완전 분석 실행"""
        logger.info("발표용 결과 분석 시작...")
        
        # 결과 로드
        results = self.load_experiment_results(results_dirs)
        
        if not any(results.values()):
            logger.error("분석할 결과가 없습니다.")
            return {}
        
        # 분석 실행
        success_analysis = self.analyze_success_rates(results)
        performance_analysis = self.analyze_performance_metrics(results)
        
        # 시각화 생성
        output_files = {}
        
        if success_analysis:
            output_files['success_chart'] = self.create_success_rate_comparison(success_analysis)
        
        if performance_analysis:
            output_files['performance_chart'] = self.create_performance_comparison(performance_analysis)
        
        output_files['learning_curves'] = self.create_learning_curves_comparison(results)
        output_files['summary'] = self.generate_presentation_summary(success_analysis, performance_analysis)
        output_files['comparison_table'] = self.create_comparison_table(success_analysis, performance_analysis)
        
        # 종합 결과 저장
        analysis_summary = {
            'analysis_date': datetime.now().isoformat(),
            'algorithms_analyzed': list(results.keys()),
            'success_analysis': success_analysis,
            'performance_analysis': performance_analysis,
            'output_files': output_files
        }
        
        summary_path = self.output_dir / "complete_analysis.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"종합 분석 완료. 결과 저장: {self.output_dir}")
        return output_files

def main():
    parser = argparse.ArgumentParser(description='발표용 결과 분석')
    
    parser.add_argument('--dqn-results', type=str, default='results/dqn',
                       help='DQN 결과 디렉터리')
    parser.add_argument('--ppo-results', type=str, default='results/ppo',
                       help='PPO 결과 디렉터리')
    parser.add_argument('--aco-results', type=str, default='results/aco',
                       help='ACO 결과 디렉터리 (선택사항)')
    parser.add_argument('--output', type=str, default='docs',
                       help='출력 디렉터리')
    
    args = parser.parse_args()
    
    # 결과 디렉터리 설정
    results_dirs = {}
    
    if Path(args.dqn_results).exists():
        results_dirs['DQN'] = args.dqn_results
    
    if Path(args.ppo_results).exists():
        results_dirs['PPO'] = args.ppo_results
    
    if args.aco_results and Path(args.aco_results).exists():
        results_dirs['ACO'] = args.aco_results
    
    if not results_dirs:
        print("❌ 분석할 결과 디렉터리가 없습니다.")
        print("다음 명령어로 실험을 먼저 실행하세요:")
        print("  python scripts/train_dqn.py --maze-id 000001 --output results/dqn/test.json")
        print("  python scripts/train_ppo.py --maze-id 000001 --output results/ppo/test.json")
        return
    
    print(f"📊 분석 시작: {list(results_dirs.keys())}")
    
    # 분석 실행
    analyzer = PresentationAnalyzer(args.output)
    output_files = analyzer.run_complete_analysis(results_dirs)
    
    # 결과 출력
    print("\n🎉 분석 완료!")
    print(f"📁 출력 디렉터리: {args.output}")
    print("\n📊 생성된 파일들:")
    
    for file_type, file_path in output_files.items():
        if file_path:
            print(f"  📄 {file_type}: {file_path}")
    
    print(f"\n🎯 발표 준비:")
    print(f"  1. 요약 문서: {output_files.get('summary', 'N/A')}")
    print(f"  2. 성공률 차트: {output_files.get('success_chart', 'N/A')}")
    print(f"  3. 성능 비교: {output_files.get('performance_chart', 'N/A')}")
    print(f"  4. 학습 곡선: {output_files.get('learning_curves', 'N/A')}")

if __name__ == "__main__":
    main()