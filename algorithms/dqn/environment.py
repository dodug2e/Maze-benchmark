"""
DQN용 미로 환경 클래스
기존 maze_io.py와 호환되도록 설계
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional, List
from utils.maze_io import load_maze_as_array
import logging

logger = logging.getLogger(__name__)

class MazeEnvironment:
    """DQN을 위한 미로 환경"""
    
    def __init__(self, maze_id: str, subset: str = "train"):
        """
        Args:
            maze_id: 미로 ID (예: "000001")
            subset: 데이터셋 분할 ("train", "valid", "test")
        """
        self.maze_id = maze_id
        self.subset = subset
        
        # 미로 로드
        self.maze_array, self.metadata = load_maze_as_array(maze_id, subset)
        self.height, self.width = self.maze_array.shape
        
        # 시작점과 목표점 설정
        self.start_pos = tuple(self.metadata.get('entrance', [1, 1]))
        self.goal_pos = tuple(self.metadata.get('exit', [self.width-2, self.height-2]))
        
        # 현재 위치
        self.current_pos = None
        
        # 행동 공간: 상, 하, 좌, 우
        self.action_space = 4
        self.actions = {
            0: (-1, 0),  # 위
            1: (1, 0),   # 아래  
            2: (0, -1),  # 왼쪽
            3: (0, 1)    # 오른쪽
        }
        
        # 상태 공간: 위치 + 목표까지의 거리 정보
        self.state_size = 4  # [x, y, goal_x, goal_y]
        
        # 에피소드 관련
        self.max_steps = self.width * self.height * 2  # 충분한 스텝 수
        self.current_step = 0
        
        # 방문 기록 (순환 방지)
        self.visited_positions = set()
        self.position_visit_count = {}
        
    def reset(self) -> np.ndarray:
        """환경 초기화"""
        self.current_pos = self.start_pos
        self.current_step = 0
        self.visited_positions = {self.start_pos}
        self.position_visit_count = {self.start_pos: 1}
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """현재 상태 반환"""
        # 정규화된 위치 정보
        state = np.array([
            self.current_pos[0] / self.height,      # 현재 x 위치 (정규화)
            self.current_pos[1] / self.width,       # 현재 y 위치 (정규화)
            self.goal_pos[0] / self.height,         # 목표 x 위치 (정규화)
            self.goal_pos[1] / self.width,          # 목표 y 위치 (정규화)
        ], dtype=np.float32)
        
        return state
    
    def _get_manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """맨하탄 거리 계산"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        행동 실행
        
        Returns:
            next_state, reward, done, info
        """
        self.current_step += 1
        
        # 행동 실행
        dx, dy = self.actions[action]
        new_x = self.current_pos[0] + dx
        new_y = self.current_pos[1] + dy
        new_pos = (new_x, new_y)
        
        # 보상 계산
        reward = self._calculate_reward(new_pos, action)
        
        # 유효한 움직임인지 확인
        if self._is_valid_position(new_pos):
            old_distance = self._get_manhattan_distance(self.current_pos, self.goal_pos)
            self.current_pos = new_pos
            new_distance = self._get_manhattan_distance(self.current_pos, self.goal_pos)
            
            # 방문 기록 업데이트
            if new_pos in self.position_visit_count:
                self.position_visit_count[new_pos] += 1
            else:
                self.position_visit_count[new_pos] = 1
            self.visited_positions.add(new_pos)
            
        # 종료 조건 확인
        done, info = self._check_done()
        
        next_state = self._get_state()
        
        return next_state, reward, done, info
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """위치가 유효한지 확인"""
        x, y = pos
        
        # 경계 확인
        if x < 0 or x >= self.height or y < 0 or y >= self.width:
            return False
        
        # 벽 확인 (0은 벽, 1은 통로)
        if self.maze_array[x, y] == 0:
            return False
        
        return True
    
    def _calculate_reward(self, new_pos: Tuple[int, int], action: int) -> float:
        """보상 계산"""
        reward = 0.0
        
        # 목표 도달
        if new_pos == self.goal_pos:
            return 100.0
        
        # 유효하지 않은 움직임 (벽 충돌)
        if not self._is_valid_position(new_pos):
            return -10.0
        
        # 거리 기반 보상
        old_distance = self._get_manhattan_distance(self.current_pos, self.goal_pos)
        new_distance = self._get_manhattan_distance(new_pos, self.goal_pos)
        
        if new_distance < old_distance:
            reward += 1.0  # 목표에 가까워짐
        elif new_distance > old_distance:
            reward -= 0.5  # 목표에서 멀어짐
        
        # 반복 방문 패널티
        visit_count = self.position_visit_count.get(new_pos, 0)
        if visit_count > 3:
            reward -= visit_count * 0.5
        
        # 시간 패널티 (빠른 해결 유도)
        reward -= 0.01
        
        return reward
    
    def _check_done(self) -> Tuple[bool, Dict]:
        """에피소드 종료 조건 확인"""
        info = {
            'success': False,
            'timeout': False,
            'current_step': self.current_step,
            'max_steps': self.max_steps
        }
        
        # 목표 도달
        if self.current_pos == self.goal_pos:
            info['success'] = True
            return True, info
        
        # 시간 초과
        if self.current_step >= self.max_steps:
            info['timeout'] = True
            return True, info
        
        return False, info
    
    def get_valid_actions(self) -> List[int]:
        """현재 위치에서 유효한 행동 목록"""
        valid_actions = []
        
        for action, (dx, dy) in self.actions.items():
            new_x = self.current_pos[0] + dx
            new_y = self.current_pos[1] + dy
            
            if self._is_valid_position((new_x, new_y)):
                valid_actions.append(action)
        
        return valid_actions
    
    def render(self) -> np.ndarray:
        """현재 상태 시각화 (디버깅용)"""
        # 미로 복사
        display_maze = self.maze_array.copy().astype(float)
        
        # 현재 위치 표시
        if self.current_pos:
            display_maze[self.current_pos] = 0.5
        
        # 목표 위치 표시
        display_maze[self.goal_pos] = 0.8
        
        # 시작 위치 표시
        display_maze[self.start_pos] = 0.3
        
        return display_maze
    
    def get_info(self) -> Dict:
        """환경 정보 반환"""
        return {
            'maze_id': self.maze_id,
            'maze_size': (self.height, self.width),
            'start_pos': self.start_pos,
            'goal_pos': self.goal_pos,
            'current_pos': self.current_pos,
            'current_step': self.current_step,
            'max_steps': self.max_steps,
            'visited_positions': len(self.visited_positions),
            'state_size': self.state_size,
            'action_space': self.action_space
        }


class MultiMazeEnvironment:
    """여러 미로를 순차적으로 학습하기 위한 환경"""
    
    def __init__(self, maze_ids: List[str], subset: str = "train"):
        """
        Args:
            maze_ids: 미로 ID 목록
            subset: 데이터셋 분할
        """
        self.maze_ids = maze_ids
        self.subset = subset
        self.current_maze_idx = 0
        self.current_env = None
        
        # 첫 번째 미로로 초기화
        self._switch_maze()
    
    def _switch_maze(self):
        """다음 미로로 전환"""
        maze_id = self.maze_ids[self.current_maze_idx]
        self.current_env = MazeEnvironment(maze_id, self.subset)
        logger.info(f"미로 전환: {maze_id}")
    
    def reset(self) -> np.ndarray:
        """환경 리셋 (필요시 다음 미로로 전환)"""
        return self.current_env.reset()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """행동 실행"""
        state, reward, done, info = self.current_env.step(action)
        
        # 에피소드 완료시 다음 미로로 전환 (선택적)
        if done:
            self.current_maze_idx = (self.current_maze_idx + 1) % len(self.maze_ids)
            # 다음 리셋에서 새 미로 사용
        
        return state, reward, done, info
    
    def switch_to_next_maze(self):
        """수동으로 다음 미로로 전환"""
        self.current_maze_idx = (self.current_maze_idx + 1) % len(self.maze_ids)
        self._switch_maze()
    
    def get_current_maze_info(self) -> Dict:
        """현재 미로 정보"""
        return self.current_env.get_info()
    
    @property
    def state_size(self):
        return self.current_env.state_size
    
    @property
    def action_space(self):
        return self.current_env.action_space


if __name__ == "__main__":
    # 테스트 코드
    print("=== DQN 미로 환경 테스트 ===")
    
    try:
        # 단일 미로 환경 테스트
        env = MazeEnvironment("000001", "train")
        
        print(f"환경 정보: {env.get_info()}")
        
        # 몇 스텝 실행
        state = env.reset()
        print(f"초기 상태: {state}")
        
        for step in range(10):
            valid_actions = env.get_valid_actions()
            if valid_actions:
                action = np.random.choice(valid_actions)
                next_state, reward, done, info = env.step(action)
                
                print(f"스텝 {step+1}: 행동={action}, 보상={reward:.2f}, 완료={done}")
                
                if done:
                    print(f"에피소드 완료: {info}")
                    break
        
        # 다중 미로 환경 테스트
        print("\n=== 다중 미로 환경 테스트 ===")
        multi_env = MultiMazeEnvironment(["000001", "000002"], "train")
        
        state = multi_env.reset()
        print(f"다중 환경 초기 상태: {state}")
        print(f"현재 미로: {multi_env.get_current_maze_info()['maze_id']}")
        
    except Exception as e:
        print(f"테스트 실패: {e}")
        import traceback
        traceback.print_exc()