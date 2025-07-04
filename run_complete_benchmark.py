#!/usr/bin/env python3
"""
완전 벤치마크 실행 스크립트
DQN, PPO 학습부터 결과 분석까지 원클릭 실행
랩 세미나 발표용 완전 자동화 파이프라인
"""

import argparse
import json
import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('complete_benchmark.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class CompleteBenchmarkRunner:
    """완전 벤치마크 실행기"""
    
    def __init__(self, 
                 mode: str = "quick",
                 output_dir: str = "results",
                 presentation_dir: str = "docs"):
        
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.presentation_dir = Path(presentation_dir)
        
        # 디렉터리 생성
        for subdir in ['dqn', 'ppo', 'comparison']:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        self.presentation_dir.mkdir(parents=True, exist_ok=True)
        
        # 실행 모드별 설정
        self.mode_configs = {
            'quick': {
                'description': '빠른 테스트 (20분)',
                'dqn_episodes': 200,
                'ppo_timesteps': 50000,
                'max_steps': 300,
                'test_mazes': ['000001', '000002']
            },
            'standard': {
                'description': '표준 실험 (2시간)', 
                'dqn_episodes': 1000,
                'ppo_timesteps': 500000,
                'max_steps': 1000,
                'test_mazes': ['000001', '000010', '000020', '000030', '000040']
            },
            'complete': {
                'description': '완전 실험 (6시간)',
                'dqn_episodes': 2000,
                'ppo_timesteps': 1000000,
                'max_steps': 1000,
                'test_mazes': ['000001', '000005', '000010', '000015', '000020', 
                              '000025', '000030', '000035', '000040', '000045']
            }
        }
        
        self.config = self.mode_configs[mode]
        self.execution_log = []
        
        logger.info(f"벤치마크 모드: {mode} - {self.config['description']}")
    
    def check_prerequisites(self) -> bool:
        """실행 전 요구사항 확인"""
        logger.info("실행 전 요구사항 확인 중...")
        
        checks = []
        
        # Python 패키지 확인
        required_packages = ['torch', 'numpy', 'matplotlib', 'pandas']
        for package in required_packages:
            try:
                __import__(package)
                checks.append(f"✅ {package}")
            except ImportError:
                checks.append(f"❌ {package} 설치 필요")
                logger.error(f"{package} 패키지가 설치되지 않았습니다.")
                return False
        
        # GPU 확인
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                checks.append(f"✅ GPU: {gpu_name}")
            else:
                checks.append("⚠️  GPU 없음 (CPU 모드로 실행)")
        except:
            checks.append("❌ PyTorch GPU 확인 실패")
        
        # 데이터셋 확인
        dataset_path = Path("datasets")
        if dataset_path.exists():
            checks.append(f"✅ 데이터셋: {dataset_path}")
        else:
            checks.append(f"❌ 데이터셋 없음: {dataset_path}")
            logger.error("데이터셋이 없습니다. maze_generator.py를 먼저 실행하세요.")
            return False
        
        # 결과 출력
        print("\n📋 사전 확인 결과:")
        for check in checks:
            print(f"  {check}")
        
        return True
    
    def run_dqn_experiments(self) -> bool:
        """DQN 실험 실행"""
        logger.info("🤖 DQN 실험 시작...")
        
        start_time = time.time()
        
        for i, maze_id in enumerate(self.config['test_mazes']):
            logger.info(f"DQN 실험 {i+1}/{len(self.config['test_mazes'])}: 미로 {maze_id}")
            
            cmd = [
                sys.executable, "scripts/train_dqn.py",
                "--maze-id", maze_id,
                "--episodes", str(self.config['dqn_episodes']),
                "--max-steps", str(self.config['max_steps']),
                "--output", str(self.output_dir / "dqn" / f"maze_{maze_id}.json")
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                
                if result.returncode == 0:
                    logger.info(f"✅ DQN 미로 {maze_id} 완료")
                    self.execution_log.append({
                        'algorithm': 'DQN',
                        'maze_id': maze_id,
                        'status': 'success',
                        'output_file': str(self.output_dir / "dqn" / f"maze_{maze_id}.json")
                    })
                else:
                    logger.error(f"❌ DQN 미로 {maze_id} 실패: {result.stderr}")
                    self.execution_log.append({
                        'algorithm': 'DQN',
                        'maze_id': maze_id,
                        'status': 'failed',
                        'error': result.stderr
                    })
                    
            except subprocess.TimeoutExpired:
                logger.error(f"❌ DQN 미로 {maze_id} 타임아웃")
                self.execution_log.append({
                    'algorithm': 'DQN',
                    'maze_id': maze_id,
                    'status': 'timeout'
                })
            
            except Exception as e:
                logger.error(f"❌ DQN 미로 {maze_id} 오류: {e}")
                self.execution_log.append({
                    'algorithm': 'DQN',
                    'maze_id': maze_id,
                    'status': 'error',
                    'error': str(e)
                })
        
        dqn_time = time.time() - start_time
        logger.info(f"🤖 DQN 실험 완료 ({dqn_time/60:.1f}분)")
        
        return True
    
    def run_ppo_experiments(self) -> bool:
        """PPO 실험 실행"""
        logger.info("🎯 PPO 실험 시작...")
        
        start_time = time.time()
        
        for i, maze_id in enumerate(self.config['test_mazes']):
            logger.info(f"PPO 실험 {i+1}/{len(self.config['test_mazes'])}: 미로 {maze_id}")
            
            cmd = [
                sys.executable, "scripts/train_ppo.py",
                "--maze-id", maze_id,
                "--timesteps", str(self.config['ppo_timesteps']),
                "--max-steps", str(self.config['max_steps']),
                "--plot-curves",
                "--output", str(self.output_dir / "ppo" / f"maze_{maze_id}.json")
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                
                if result.returncode == 0:
                    logger.info(f"✅ PPO 미로 {maze_id} 완료")
                    self.execution_log.append({
                        'algorithm': 'PPO',
                        'maze_id': maze_id,
                        'status': 'success',
                        'output_file': str(self.output_dir / "ppo" / f"maze_{maze_id}.json")
                    })
                else:
                    logger.error(f"❌ PPO 미로 {maze_id} 실패: {result.stderr}")
                    self.execution_log.append({
                        'algorithm': 'PPO',
                        'maze_id': maze_id,
                        'status': 'failed',
                        'error': result.stderr
                    })
                    
            except subprocess.TimeoutExpired:
                logger.error(f"❌ PPO 미로 {maze_id} 타임아웃")
                self.execution_log.append({
                    'algorithm': 'PPO',
                    'maze_id': maze_id,
                    'status': 'timeout'
                })
            
            except Exception as e:
                logger.error(f"❌ PPO 미로 {maze_id} 오류: {e}")
                self.execution_log.append({
                    'algorithm': 'PPO',
                    'maze_id': maze_id,
                    'status': 'error',
                    'error': str(e)
                })
        
        ppo_time = time.time() - start_time
        logger.info(f"🎯 PPO 실험 완료 ({ppo_time/60:.1f}분)")
        
        return True
    
    def run_comparison_analysis(self) -> bool:
        """비교 분석 실행"""
        logger.info("📊 결과 분석 시작...")
        
        cmd = [
            sys.executable, "scripts/analyze_results.py",
            "--dqn-results", str(self.output_dir / "dqn"),
            "--ppo-results", str(self.output_dir / "ppo"),
            "--output", str(self.presentation_dir)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info("✅ 결과 분석 완료")
                return True
            else:
                logger.error(f"❌ 결과 분석 실패: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 결과 분석 오류: {e}")
            return False
    
    def generate_final_report(self) -> str:
        """최종 보고서 생성"""
        logger.info("📋 최종 보고서 생성...")
        
        # 실행 통계 계산
        total_experiments = len(self.execution_log)
        successful = len([log for log in self.execution_log if log['status'] == 'success'])
        failed = len([log for log in self.execution_log if log['status'] == 'failed'])
        timeout = len([log for log in self.execution_log if log['status'] == 'timeout'])
        
        # 알고리즘별 성공률
        dqn_logs = [log for log in self.execution_log if log['algorithm'] == 'DQN']
        ppo_logs = [log for log in self.execution_log if log['algorithm'] == 'PPO']
        
        dqn_success = len([log for log in dqn_logs if log['status'] == 'success'])
        ppo_success = len([log for log in ppo_logs if log['status'] == 'success'])
        
        # 보고서 내용
        report = f"""# 🎯 완전 벤치마크 실행 보고서

## 📋 실행 정보
- **실행 모드**: {self.mode} ({self.config['description']})
- **실행 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **테스트 미로**: {len(self.config['test_mazes'])}개
- **총 실험**: {total_experiments}개

## 📊 실행 결과 요약
- ✅ **성공**: {successful}개 ({successful/total_experiments*100:.1f}%)
- ❌ **실패**: {failed}개 ({failed/total_experiments*100:.1f}%)
- ⏰ **타임아웃**: {timeout}개 ({timeout/total_experiments*100:.1f}%)

## 🤖 알고리즘별 성공률
- **DQN**: {dqn_success}/{len(dqn_logs)} ({dqn_success/len(dqn_logs)*100:.1f}% 성공)
- **PPO**: {ppo_success}/{len(ppo_logs)} ({ppo_success/len(ppo_logs)*100:.1f}% 성공)

## 📁 생성된 파일들
- 결과 디렉터리: `{self.output_dir}/`
- 발표 자료: `{self.presentation_dir}/`
- 실행 로그: `complete_benchmark.log`

## 🎯 다음 단계
1. **발표 자료 확인**: `{self.presentation_dir}/presentation_summary.md`
2. **성능 그래프**: `{self.presentation_dir}/*.png`
3. **상세 결과**: `{self.presentation_dir}/complete_analysis.json`

## 📝 개별 실험 결과
"""
        
        # 개별 실험 결과 추가
        for log in self.execution_log:
            status_emoji = "✅" if log['status'] == 'success' else "❌"
            report += f"- {status_emoji} {log['algorithm']} 미로 {log['maze_id']}: {log['status']}\n"
        
        report += f"""
---

## 🚀 발표 준비 체크리스트
- [ ] 결과 그래프 확인
- [ ] 발표 요약 문서 검토
- [ ] 시연 준비 (성공한 실험 위주)
- [ ] 질의응답 준비

**🎉 벤치마크 실행 완료! 좋은 발표 되세요!**
"""
        
        # 보고서 저장
        report_path = self.presentation_dir / "final_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 실행 로그 JSON 저장
        log_path = self.presentation_dir / "execution_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.execution_log, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 최종 보고서 저장: {report_path}")
        return str(report_path)
    
    def run_complete_pipeline(self) -> bool:
        """완전 파이프라인 실행"""
        print(f"""
🚀 완전 벤치마크 시작!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
모드: {self.mode} - {self.config['description']}
테스트 미로: {len(self.config['test_mazes'])}개
예상 소요 시간: {self._estimate_time()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """)
        
        start_time = time.time()
        
        # 1. 사전 확인
        if not self.check_prerequisites():
            logger.error("❌ 사전 확인 실패")
            return False
        
        # 2. DQN 실험
        if not self.run_dqn_experiments():
            logger.error("❌ DQN 실험 실패")
            return False
        
        # 3. PPO 실험  
        if not self.run_ppo_experiments():
            logger.error("❌ PPO 실험 실패")
            return False
        
        # 4. 결과 분석
        if not self.run_comparison_analysis():
            logger.warning("⚠️  결과 분석 실패 (실험 결과는 사용 가능)")
        
        # 5. 최종 보고서
        report_path = self.generate_final_report()
        
        total_time = time.time() - start_time
        
        print(f"""
🎉 완전 벤치마크 완료!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
총 실행 시간: {total_time/60:.1f}분
최종 보고서: {report_path}
발표 자료: {self.presentation_dir}/
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """)
        
        return True
    
    def _estimate_time(self) -> str:
        """예상 소요 시간 계산"""
        if self.mode == 'quick':
            return "약 20분"
        elif self.mode == 'standard':
            return "약 2시간"
        else:
            return "약 6시간"

def main():
    parser = argparse.ArgumentParser(description='완전 벤치마크 실행')
    
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'standard', 'complete'],
                       help='실행 모드 선택')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='결과 저장 디렉터리')
    parser.add_argument('--presentation-dir', type=str, default='docs',
                       help='발표 자료 저장 디렉터리')
    parser.add_argument('--dry-run', action='store_true',
                       help='실제 실행 없이 계획만 확인')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 드라이 런 모드: 실행 계획만 확인합니다.")
        
        runner = CompleteBenchmarkRunner(args.mode, args.output_dir, args.presentation_dir)
        
        print(f"\n📋 실행 계획:")
        print(f"  모드: {args.mode}")
        print(f"  테스트 미로: {runner.config['test_mazes']}")
        print(f"  DQN 에피소드: {runner.config['dqn_episodes']}")
        print(f"  PPO 타임스텝: {runner.config['ppo_timesteps']}")
        print(f"  예상 시간: {runner._estimate_time()}")
        
        return
    
    # 실제 실행
    runner = CompleteBenchmarkRunner(args.mode, args.output_dir, args.presentation_dir)
    success = runner.run_complete_pipeline()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()