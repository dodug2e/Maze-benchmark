#!/usr/bin/env python3
"""
간단한 A* 테스트 - 실제 반환값 확인
"""

import sys
import numpy as np
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.astar import AStarSolver

def test_simple_maze():
    """간단한 5x5 미로로 A* 테스트"""
    
    # 간단한 5x5 미로 생성 (0=통로, 1=벽)
    simple_maze = np.array([
        [0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.uint8)
    
    print("테스트 미로:")
    print(simple_maze)
    print("0: 통로, 1: 벽")
    
    start = (0, 0)
    goal = (4, 4)
    
    print(f"\n시작점: {start}")
    print(f"끝점: {goal}")
    
    # A* 실행
    solver = AStarSolver(diagonal_movement=False)
    result = solver.solve(simple_maze, start, goal)
    
    print(f"\n결과 타입: {type(result)}")
    print(f"결과 내용:")
    
    # 결과 분석
    if hasattr(result, '__dict__'):
        for key, value in result.__dict__.items():
            print(f"  {key}: {value}")
    else:
        print(f"  {result}")
    
    # 성공 여부 확인
    if hasattr(result, 'solution_found'):
        print(f"\n해결 성공: {result.solution_found}")
        if result.solution_found and hasattr(result, 'path'):
            print(f"경로: {result.path}")
            print(f"경로 길이: {len(result.path) if result.path else 0}")
    
    return result

if __name__ == "__main__":
    print("=== 간단한 A* 테스트 ===")
    test_simple_maze()