#!/usr/bin/env python3
"""
A* 데이터 로딩 및 구조 확인 테스트
실제 데이터 구조를 확인하고 A* 알고리즘과의 호환성 체크
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.maze_io import get_loader, load_maze_as_array

def test_data_loading():
    """데이터 로딩 테스트"""
    print("=== 데이터 로딩 테스트 ===")
    
    try:
        # 데이터 로더 초기화
        loader = get_loader()
        print("✅ 데이터 로더 초기화 성공")
        
        # 샘플 ID 목록 가져오기
        sample_ids = loader.get_sample_ids("test")
        print(f"✅ {len(sample_ids)} 개 test 샘플 발견")
        
        if sample_ids:
            # 첫 번째 샘플 테스트
            sample_id = sample_ids[0]
            print(f"\n샘플 {sample_id} 테스트:")
            
            # 직접 로드 방식
            maze_array, metadata = load_maze_as_array(sample_id, "test")
            print(f"  ✅ 미로 배열 형태: {maze_array.shape}")
            print(f"  ✅ 미로 값 범위: {maze_array.min()} ~ {maze_array.max()}")
            print(f"  ✅ 메타데이터 키: {list(metadata.keys())}")
            
            # 시작점과 끝점 확인
            start = tuple(metadata.get('entrance', (0, 0)))
            goal = tuple(metadata.get('exit', (maze_array.shape[0]-1, maze_array.shape[1]-1)))
            print(f"  시작점: {start}")
            print(f"  끝점: {goal}")
            
            # 미로 구조 분석
            print(f"  벽(1) 개수: {np.sum(maze_array == 1)}")
            print(f"  통로(0) 개수: {np.sum(maze_array == 0)}")
            
            return maze_array, metadata, start, goal
            
    except Exception as e:
        print(f"❌ 데이터 로딩 실패: {e}")
        return None, None, None, None

def test_astar_import():
    """A* 알고리즘 import 테스트"""
    print("\n=== A* 알고리즘 Import 테스트 ===")
    
    try:
        # A* 알고리즘 import 시도
        from algorithms.astar import AStarSolver, AStarResult
        print("✅ AStarSolver, AStarResult import 성공")
        
        # 인스턴스 생성 테스트
        solver = AStarSolver(diagonal_movement=False)
        print("✅ AStarSolver 인스턴스 생성 성공")
        
        return solver
        
    except ImportError as e:
        print(f"❌ A* 알고리즘 import 실패: {e}")
        print("   algorithms/astar.py 파일이 존재하는지 확인하세요.")
        return None
    except Exception as e:
        print(f"❌ A* 알고리즘 초기화 실패: {e}")
        return None

def test_astar_solve(solver, maze_array, start, goal):
    """A* 해결 테스트"""
    print("\n=== A* 해결 테스트 ===")
    
    if solver is None or maze_array is None:
        print("❌ 이전 단계에서 실패하여 테스트 불가")
        return None
    
    try:
        print(f"미로 크기: {maze_array.shape}")
        print(f"시작점: {start}, 끝점: {goal}")
        
        # A* 실행
        result = solver.solve(maze_array, start, goal)
        print(f"✅ A* 실행 완료")
        print(f"  결과 타입: {type(result)}")
        print(f"  결과 내용: {result}")
        
        return result
        
    except Exception as e:
        print(f"❌ A* 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import numpy as np
    
    # 1. 데이터 로딩 테스트
    maze_array, metadata, start, goal = test_data_loading()
    
    # 2. A* import 테스트
    solver = test_astar_import()
    
    # 3. A* 해결 테스트
    if solver and maze_array is not None:
        result = test_astar_solve(solver, maze_array, start, goal)
    
    print("\n=== 테스트 완료 ===")
    if maze_array is not None and solver is not None:
        print("✅ 모든 기본 컴포넌트가 작동 중")
        print("이제 baseline_astar.py를 실행할 수 있습니다!")
    else:
        print("❌ 일부 컴포넌트에 문제가 있습니다. 위 오류를 확인해주세요.")