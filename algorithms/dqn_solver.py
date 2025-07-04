"""
DQN 기반 미로 탐색 알고리즘
RTX 3060 최적화 및 기존 벤치마크 시스템 호환
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
from typing import Tuple, List, Optional, Dict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 경험 재생을 위한 Transition 구조체
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class MazeEnvironment:
    """미로 탐색을 위한 강화학습 환경"""
    
    def __init__(self, maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], max_steps: int = 1000):
        self.maze = maze.copy()  # 0: 벽, 1: 통로
        self.height, self.width = maze.shape
        self.start = start
        self.goal = goal
        self.max_steps = max_steps
        
        # 현재 상태
        self.current_pos = None
        self.step_count = 0
        self.visited = set()
        
        # 행동 공간: 상, 하, 좌, 우
        self.actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.action_size = len(self.actions)
        
        # 상태 공간 설계: 위치 + 목표 상대 위치 + 방문 정보
        self.state_size = self.height * self.width + 4  # 위치 one-hot + 목표 거리 정보
        
    def reset(self) -> np.ndarray:
        """환경 초기화"""
        self.current_pos = self.start
        self.step_count = 0
        self.visited = set()
        self.visited.add(self.current_pos)
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """행동 실행"""
        if self.current_pos is None:
            raise ValueError("Environment not reset")
            
        # 행동 실행
        dx, dy = self.actions[action]
        new_x = self.current_pos[0] + dx
        new_y = self.current_pos[1] + dy
        
        # 경계 및 벽 충돌 체크
        valid_move = (0 <= new_x < self.height and 
                     0 <= new_y < self.width and 
                     self.maze[new_x, new_y] == 1)
        
        reward = 0.0
        done = False
        info = {}
        
        if valid_move:
            self.current_pos = (new_x, new_y)
            
            # 보상 계산
            if self.current_pos == self.goal:
                reward = 100.0  # 목표 도달
                done = True
                info['success'] = True
            elif self.current_pos in self.visited:
                reward = -0.5  # 이미 방문한 곳 (순환 방지)
            else:
                # 목표에 가까워질수록 높은 보상
                old_dist = self._manhattan_distance(self.start, self.goal)
                new_dist = self._manhattan_distance(self.current_pos, self.goal)
                reward = (old_dist - new_dist) * 0.1
                
            self.visited.add(self.current_pos)
        else:
            reward = -1.0  # 벽 충돌 또는 경계 벗어남
        
        # 스텝 수 증가
        self.step_count += 1
        
        # 최대 스텝 도달 시 종료
        if self.step_count >= self.max_steps:
            done = True
            if self.current_pos != self.goal:
                reward -= 10.0  # 시간 초과 페널티
                info['timeout'] = True
        
        # 시간에 따른 작은 페널티 (효율성 장려)
        reward -= 0.01
        
        return self._get_state(), reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """현재 상태를 벡터로 변환"""
        state = np.zeros(self.state_size, dtype=np.float32)
        
        # 현재 위치를 one-hot 인코딩
        pos_index = self.current_pos[0] * self.width + self.current_pos[1]
        state[pos_index] = 1.0
        
        # 목표까지의 거리 정보 (정규화)
        goal_dx = (self.goal[0] - self.current_pos[0]) / self.height
        goal_dy = (self.goal[1] - self.current_pos[1]) / self.width
        manhattan_dist = self._manhattan_distance(self.current_pos, self.goal) / (self.height + self.width)
        euclidean_dist = self._euclidean_distance(self.current_pos, self.goal) / np.sqrt(self.height**2 + self.width**2)
        
        state[-4:] = [goal_dx, goal_dy, manhattan_dist, euclidean_dist]
        
        return state
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """맨하탄 거리 계산"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """유클리드 거리 계산"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


class DQN(nn.Module):
    """Deep Q-Network 모델 (RTX 3060 최적화)"""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [512, 256, 128]):
        super(DQN, self).__init__()
        
        layers = []
        input_size = state_size
        
        # 은닉층 생성
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # 과적합 방지
            input_size = hidden_size
        
        # 출력층
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """가중치 초기화"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """경험 재생 버퍼 (메모리 효율성 최적화)"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """경험 저장"""
        self.buffer.append(Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Transition]:
        """배치 샘플링"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN 에이전트 (RTX 3060 VRAM 최적화)"""
    
    def __init__(self, 
                 state_size: int, 
                 action_size: int,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 32,
                 target_update: int = 1000,
                 device: str = 'auto'):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # 디바이스 설정
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"DQN 에이전트 초기화: {self.device}")
        
        # 네트워크 초기화
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 타겟 네트워크 동기화
        self.update_target_network()
        
        # 경험 재생 버퍼
        self.memory = ReplayBuffer(memory_size)
        
        # 학습 카운터
        self.learn_step = 0
        
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """행동 선택 (ε-greedy)"""
        if training and random.random() < self.epsilon:
            return random.choice(range(self.action_size))
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.cpu().data.numpy().argmax()
    
    def remember(self, state, action, reward, next_state, done):
        """경험 저장"""
        self.memory.push(state, action, reward, next_state, done)
    
    def learn(self):
        """경험 재생을 통한 학습"""
        if len(self.memory) < self.batch_size:
            return
        
        # 배치 샘플링
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # 텐서 변환
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.BoolTensor(batch.done).to(self.device)
        
        # 현재 Q값 계산
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # 다음 상태의 최대 Q값 계산 (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_state_batch).max(1)[1].unsqueeze(1)
            next_q_values = self.target_network(next_state_batch).gather(1, next_actions).squeeze(1)
            target_q_values = reward_batch + (self.gamma * next_q_values * ~done_batch)
        
        # 손실 계산 및 역전파
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # 그래디언트 클리핑 (안정성 향상)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # ε 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 타겟 네트워크 업데이트
        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.update_target_network()
    
    def update_target_network(self):
        """타겟 네트워크 가중치 업데이트"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath: str):
        """모델 저장"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'learn_step': self.learn_step
        }, filepath)
        logger.info(f"DQN 모델 저장: {filepath}")
    
    def load(self, filepath: str):
        """모델 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.learn_step = checkpoint['learn_step']
        logger.info(f"DQN 모델 로드: {filepath}")


class DQNSolver:
    """DQN 미로 해결사 (벤치마크 시스템 호환)"""
    
    def __init__(self, 
                 episodes: int = 2000,
                 max_steps: int = 1000,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 32,
                 target_update: int = 1000,
                 save_interval: int = 500,
                 device: str = 'auto'):
        
        self.episodes = episodes
        self.max_steps = max_steps
        self.save_interval = save_interval
        
        # 에이전트는 환경에 따라 동적 생성
        self.agent = None
        self.env = None
        
        # 하이퍼파라미터 저장
        self.config = {
            'learning_rate': learning_rate,
            'gamma': gamma,
            'epsilon_start': epsilon_start,
            'epsilon_end': epsilon_end,
            'epsilon_decay': epsilon_decay,
            'memory_size': memory_size,
            'batch_size': batch_size,
            'target_update': target_update,
            'device': device
        }
        
        # 학습 기록
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rate': [],
            'epsilon_history': []
        }
    
    def solve(self, maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], Dict]:
        """미로 해결 (추론 모드)"""
        if self.agent is None:
            raise ValueError("에이전트가 학습되지 않았습니다. train() 메서드를 먼저 실행하세요.")
        
        # 환경 설정
        env = MazeEnvironment(maze, start, goal, self.max_steps)
        state = env.reset()
        
        path = [start]
        total_reward = 0
        
        for step in range(self.max_steps):
            # 행동 선택 (탐색 없음)
            action = self.agent.act(state, training=False)
            
            # 행동 실행
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            path.append(env.current_pos)
            
            if done:
                if env.current_pos == goal:
                    logger.info(f"DQN: 목표 도달! 스텝: {step+1}, 총 보상: {total_reward:.2f}")
                    return path, {
                        'success': True, 
                        'steps': step + 1, 
                        'reward': total_reward,
                        'reason': 'goal_reached'
                    }
                else:
                    logger.warning(f"DQN: 시간 초과! 스텝: {step+1}")
                    return path, {
                        'success': False, 
                        'steps': step + 1, 
                        'reward': total_reward,
                        'reason': 'timeout'
                    }
            
            state = next_state
        
        # 최대 스텝 도달
        return path, {
            'success': False, 
            'steps': self.max_steps, 
            'reward': total_reward,
            'reason': 'max_steps'
        }
    
    def train(self, maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], 
              save_path: Optional[str] = None) -> Dict:
        """DQN 학습"""
        # 환경 초기화
        self.env = MazeEnvironment(maze, start, goal, self.max_steps)
        state_size = self.env.state_size
        action_size = self.env.action_size
        
        # 에이전트 초기화
        self.agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            **self.config
        )
        
        logger.info(f"DQN 학습 시작: {self.episodes} 에피소드, 상태 크기: {state_size}")
        
        # 성공률 추적을 위한 변수
        recent_successes = deque(maxlen=100)
        
        for episode in range(self.episodes):
            state = self.env.reset()
            total_reward = 0
            
            for step in range(self.max_steps):
                # 행동 선택
                action = self.agent.act(state, training=True)
                
                # 행동 실행
                next_state, reward, done, info = self.env.step(action)
                total_reward += reward
                
                # 경험 저장
                self.agent.remember(state, action, reward, next_state, done)
                
                # 학습
                self.agent.learn()
                
                if done:
                    success = info.get('success', False)
                    recent_successes.append(success)
                    
                    # 기록 저장
                    self.training_history['episode_rewards'].append(total_reward)
                    self.training_history['episode_lengths'].append(step + 1)
                    self.training_history['epsilon_history'].append(self.agent.epsilon)
                    
                    if len(recent_successes) > 0:
                        success_rate = sum(recent_successes) / len(recent_successes)
                        self.training_history['success_rate'].append(success_rate)
                    
                    if episode % 100 == 0:
                        logger.info(f"에피소드 {episode}: 보상={total_reward:.2f}, "
                                  f"스텝={step+1}, 성공률={success_rate:.2f}, "
                                  f"ε={self.agent.epsilon:.3f}")
                    
                    break
                
                state = next_state
            
            # 모델 저장
            if save_path and (episode + 1) % self.save_interval == 0:
                model_path = f"{save_path}_episode_{episode+1}.pth"
                self.agent.save(model_path)
        
        # 최종 모델 저장
        if save_path:
            final_path = f"{save_path}_final.pth"
            self.agent.save(final_path)
        
        logger.info("DQN 학습 완료")
        
        return {
            'final_success_rate': success_rate if recent_successes else 0.0,
            'average_reward': np.mean(self.training_history['episode_rewards'][-100:]),
            'training_history': self.training_history
        }
    
    def load_model(self, filepath: str):
        """사전 학습된 모델 로드"""
        if self.agent is None:
            # 기본 에이전트 생성 (실제 상태/행동 크기는 나중에 조정)
            self.agent = DQNAgent(100, 4, **self.config)
        
        self.agent.load(filepath)
        logger.info(f"DQN 모델 로드 완료: {filepath}")


# 벤치마크 시스템과의 호환성을 위한 팩토리 함수
def create_dqn_solver(**kwargs) -> DQNSolver:
    """DQN 해결사 생성"""
    return DQNSolver(**kwargs)


if __name__ == "__main__":
    # 간단한 테스트
    print("DQN 솔버 테스트 시작...")
    
    # 간단한 5x5 미로 생성
    test_maze = np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ])
    
    start = (0, 0)
    goal = (4, 4)
    
    # DQN 솔버 생성 및 학습
    solver = DQNSolver(episodes=100, max_steps=50)
    
    print("학습 시작...")
    training_result = solver.train(test_maze, start, goal)
    print(f"학습 완료: 성공률 {training_result['final_success_rate']:.2f}")
    
    # 해결 테스트
    print("해결 테스트...")
    path, result = solver.solve(test_maze, start, goal)
    print(f"결과: {result}")
    print(f"경로 길이: {len(path)}")