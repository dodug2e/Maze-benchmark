"""
PPO (Proximal Policy Optimization) 기반 미로 탐색 알고리즘
Actor-Critic 구조로 DQN보다 안정적인 학습 제공
RTX 3060 최적화 및 기존 벤치마크 시스템 호환
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import random
from collections import deque
from typing import Tuple, List, Optional, Dict
import logging
from pathlib import Path
import math

logger = logging.getLogger(__name__)

class PPONetwork(nn.Module):
    """PPO Actor-Critic 네트워크 (RTX 3060 최적화)"""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [512, 256]):
        super(PPONetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # 공통 특징 추출 레이어
        self.shared_layers = nn.ModuleList()
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            self.shared_layers.append(nn.Linear(input_size, hidden_size))
            self.shared_layers.append(nn.ReLU())
            self.shared_layers.append(nn.Dropout(0.1))
            input_size = hidden_size
        
        # Actor 헤드 (정책 네트워크)
        self.actor_head = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1] // 2, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Critic 헤드 (가치 네트워크)
        self.critic_head = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1] // 2, 1)
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """가중치 초기화"""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, np.sqrt(2))
            module.bias.data.fill_(0.0)
    
    def forward(self, state):
        """순전파"""
        x = state
        
        # 공통 특징 추출
        for layer in self.shared_layers:
            x = layer(x)
        
        # Actor와 Critic 출력
        action_probs = self.actor_head(x)
        state_value = self.critic_head(x)
        
        return action_probs, state_value
    
    def get_action_and_value(self, state, action=None):
        """행동 선택 및 가치 추정"""
        action_probs, state_value = self.forward(state)
        
        # 행동 분포 생성
        dist = Categorical(action_probs)
        
        if action is None:
            # 새로운 행동 샘플링
            action = dist.sample()
        
        # 로그 확률과 엔트로피 계산
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, state_value.squeeze()


class PPOBuffer:
    """PPO 경험 버퍼 (GAE 지원)"""
    
    def __init__(self, buffer_size: int, state_size: int, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # 버퍼 초기화
        self.states = np.zeros((buffer_size, state_size), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        
        # 현재 위치
        self.ptr = 0
        self.size = 0
    
    def store(self, state, action, reward, value, log_prob, done):
        """경험 저장"""
        idx = self.ptr % self.buffer_size
        
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.dones[idx] = done
        
        self.ptr += 1
        self.size = min(self.size + 1, self.buffer_size)
    
    def get(self):
        """GAE를 사용하여 어드밴티지 계산 후 버퍼 데이터 반환"""
        if self.size == 0:
            return None
        
        # 실제 저장된 데이터만 사용
        valid_size = min(self.size, self.buffer_size)
        
        states = self.states[:valid_size]
        actions = self.actions[:valid_size]
        rewards = self.rewards[:valid_size]
        values = self.values[:valid_size]
        log_probs = self.log_probs[:valid_size]
        dones = self.dones[:valid_size]
        
        # GAE 계산
        advantages = self._compute_gae(rewards, values, dones)
        returns = advantages + values
        
        # 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return {
            'states': torch.FloatTensor(states),
            'actions': torch.LongTensor(actions),
            'old_log_probs': torch.FloatTensor(log_probs),
            'advantages': torch.FloatTensor(advantages),
            'returns': torch.FloatTensor(returns),
            'values': torch.FloatTensor(values)
        }
    
    def _compute_gae(self, rewards, values, dones):
        """Generalized Advantage Estimation (GAE) 계산"""
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0  # 마지막 스텝
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        
        return advantages
    
    def clear(self):
        """버퍼 초기화"""
        self.ptr = 0
        self.size = 0


class PPOAgent:
    """PPO 에이전트 (RTX 3060 VRAM 최적화)"""
    
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_coef: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 buffer_size: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 device: str = 'auto'):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        # 디바이스 설정
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"PPO 에이전트 초기화: {self.device}")
        
        # 네트워크 초기화
        self.network = PPONetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, eps=1e-5)
        
        # 경험 버퍼
        self.buffer = PPOBuffer(buffer_size, state_size, gamma, gae_lambda)
        
        # 학습 통계
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'explained_variance': []
        }
    
    def get_action(self, state: np.ndarray, training: bool = True):
        """행동 선택"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, entropy, value = self.network.get_action_and_value(state_tensor)
        
        return action.cpu().item(), log_prob.cpu().item(), value.cpu().item()
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """경험 저장"""
        self.buffer.store(state, action, reward, value, log_prob, done)
    
    def update(self):
        """PPO 업데이트"""
        # 버퍼에서 데이터 가져오기
        batch = self.buffer.get()
        if batch is None:
            return
        
        # GPU로 이동
        for key in batch:
            batch[key] = batch[key].to(self.device)
        
        # 여러 에포크 학습
        for epoch in range(self.n_epochs):
            # 미니배치로 나누어 학습
            indices = np.arange(len(batch['states']))
            np.random.shuffle(indices)
            
            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                mb_indices = indices[start:end]
                
                # 미니배치 데이터
                mb_states = batch['states'][mb_indices]
                mb_actions = batch['actions'][mb_indices]
                mb_old_log_probs = batch['old_log_probs'][mb_indices]
                mb_advantages = batch['advantages'][mb_indices]
                mb_returns = batch['returns'][mb_indices]
                
                # 현재 정책으로 평가
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    mb_states, mb_actions
                )
                
                # PPO 손실 계산
                policy_loss, value_loss, entropy_loss = self._compute_losses(
                    new_log_probs, mb_old_log_probs, mb_advantages, 
                    new_values, mb_returns, entropy
                )
                
                total_loss = (policy_loss + 
                            self.value_loss_coef * value_loss - 
                            self.entropy_coef * entropy_loss)
                
                # 역전파
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # 통계 저장
                self.training_stats['policy_loss'].append(policy_loss.item())
                self.training_stats['value_loss'].append(value_loss.item())
                self.training_stats['entropy_loss'].append(entropy_loss.item())
                self.training_stats['total_loss'].append(total_loss.item())
        
        # 설명 분산 계산
        y_pred = batch['values'].cpu().numpy()
        y_true = batch['returns'].cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        self.training_stats['explained_variance'].append(explained_var)
        
        # 버퍼 클리어
        self.buffer.clear()
    
    def _compute_losses(self, new_log_probs, old_log_probs, advantages, 
                       new_values, returns, entropy):
        """PPO 손실 함수들 계산"""
        
        # 정책 손실 (Clipped Surrogate Loss)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 가치 손실 (Clipped Value Loss)
        value_loss = F.mse_loss(new_values, returns)
        
        # 엔트로피 손실
        entropy_loss = entropy.mean()
        
        return policy_loss, value_loss, entropy_loss
    
    def save(self, filepath: str):
        """모델 저장"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, filepath)
        logger.info(f"PPO 모델 저장: {filepath}")
    
    def load(self, filepath: str):
        """모델 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
        logger.info(f"PPO 모델 로드: {filepath}")


class PPOSolver:
    """PPO 미로 해결사 (벤치마크 시스템 호환)"""
    
    def __init__(self,
                 n_envs: int = 1,  # 병렬 환경 수 (단일 환경으로 시작)
                 total_timesteps: int = 1_000_000,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_coef: float = 0.2,
                 buffer_size: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 max_episode_steps: int = 1000,
                 save_interval: int = 50000,
                 device: str = 'auto'):
        
        self.n_envs = n_envs
        self.total_timesteps = total_timesteps
        self.max_episode_steps = max_episode_steps
        self.save_interval = save_interval
        
        # 에이전트는 환경에 따라 동적 생성
        self.agent = None
        self.env = None
        
        # 하이퍼파라미터 저장
        self.config = {
            'learning_rate': learning_rate,
            'gamma': gamma,
            'gae_lambda': gae_lambda,
            'clip_coef': clip_coef,
            'buffer_size': buffer_size,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'device': device
        }
        
        # 학습 기록
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rate': [],
            'policy_losses': [],
            'value_losses': []
        }
    
    def solve(self, maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], Dict]:
        """미로 해결 (추론 모드)"""
        if self.agent is None:
            raise ValueError("에이전트가 학습되지 않았습니다. train() 메서드를 먼저 실행하세요.")
        
        # 환경 설정 (DQN의 MazeEnvironment 재사용)
        from .dqn_solver import MazeEnvironment
        env = MazeEnvironment(maze, start, goal, self.max_episode_steps)
        state = env.reset()
        
        path = [start]
        total_reward = 0
        
        for step in range(self.max_episode_steps):
            # 행동 선택 (탐색 없음)
            action, _, _ = self.agent.get_action(state, training=False)
            
            # 행동 실행
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            path.append(env.current_pos)
            
            if done:
                if env.current_pos == goal:
                    logger.info(f"PPO: 목표 도달! 스텝: {step+1}, 총 보상: {total_reward:.2f}")
                    return path, {
                        'success': True,
                        'steps': step + 1,
                        'reward': total_reward,
                        'reason': 'goal_reached'
                    }
                else:
                    logger.warning(f"PPO: 시간 초과! 스텝: {step+1}")
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
            'steps': self.max_episode_steps,
            'reward': total_reward,
            'reason': 'max_steps'
        }
    
    def train(self, maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
              save_path: Optional[str] = None) -> Dict:
        """PPO 학습"""
        
        # 환경 초기화
        from .dqn_solver import MazeEnvironment
        self.env = MazeEnvironment(maze, start, goal, self.max_episode_steps)
        state_size = self.env.state_size
        action_size = self.env.action_size
        
        # 에이전트 초기화
        self.agent = PPOAgent(
            state_size=state_size,
            action_size=action_size,
            **self.config
        )
        
        logger.info(f"PPO 학습 시작: {self.total_timesteps} 타임스텝, 상태 크기: {state_size}")
        
        # 학습 변수
        global_step = 0
        episode_count = 0
        recent_successes = deque(maxlen=100)
        
        while global_step < self.total_timesteps:
            # 에피소드 시작
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(self.max_episode_steps):
                # 행동 선택
                action, log_prob, value = self.agent.get_action(state, training=True)
                
                # 행동 실행
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                global_step += 1
                
                # 경험 저장
                self.agent.store_transition(state, action, reward, value, log_prob, done)
                
                if done:
                    success = info.get('success', False)
                    recent_successes.append(success)
                    
                    # 기록 저장
                    self.training_history['episode_rewards'].append(episode_reward)
                    self.training_history['episode_lengths'].append(episode_length)
                    
                    if len(recent_successes) > 0:
                        success_rate = sum(recent_successes) / len(recent_successes)
                        self.training_history['success_rate'].append(success_rate)
                    
                    if episode_count % 100 == 0:
                        logger.info(f"에피소드 {episode_count}: 보상={episode_reward:.2f}, "
                                  f"스텝={episode_length}, 성공률={success_rate:.2f}, "
                                  f"글로벌 스텝={global_step}")
                    
                    break
                
                state = next_state
                
                # PPO 업데이트 (버퍼가 가득 찼을 때)
                if global_step % self.agent.buffer.buffer_size == 0:
                    self.agent.update()
                    
                    # 학습 통계 저장
                    if self.agent.training_stats['policy_loss']:
                        self.training_history['policy_losses'].append(
                            np.mean(self.agent.training_stats['policy_loss'][-10:])
                        )
                        self.training_history['value_losses'].append(
                            np.mean(self.agent.training_stats['value_loss'][-10:])
                        )
                
                # 모델 저장
                if save_path and global_step % self.save_interval == 0:
                    model_path = f"{save_path}_step_{global_step}.pth"
                    self.agent.save(model_path)
            
            episode_count += 1
        
        # 최종 업데이트
        self.agent.update()
        
        # 최종 모델 저장
        if save_path:
            final_path = f"{save_path}_final.pth"
            self.agent.save(final_path)
        
        logger.info("PPO 학습 완료")
        
        final_success_rate = sum(recent_successes) / len(recent_successes) if recent_successes else 0.0
        
        return {
            'final_success_rate': final_success_rate,
            'total_episodes': episode_count,
            'total_timesteps': global_step,
            'average_reward': np.mean(self.training_history['episode_rewards'][-100:]) if self.training_history['episode_rewards'] else 0,
            'training_history': self.training_history
        }
    
    def load_model(self, filepath: str):
        """사전 학습된 모델 로드"""
        if self.agent is None:
            # 기본 에이전트 생성
            self.agent = PPOAgent(100, 4, **self.config)
        
        self.agent.load(filepath)
        logger.info(f"PPO 모델 로드 완료: {filepath}")


# 벤치마크 시스템과의 호환성을 위한 팩토리 함수
def create_ppo_solver(**kwargs) -> PPOSolver:
    """PPO 해결사 생성"""
    return PPOSolver(**kwargs)


if __name__ == "__main__":
    # 간단한 테스트
    print("PPO 솔버 테스트 시작...")
    
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
    
    # PPO 솔버 생성 및 학습
    solver = PPOSolver(
        total_timesteps=50000,  # 테스트용 단축
        max_episode_steps=50,
        buffer_size=512,
        batch_size=32
    )
    
    print("학습 시작...")
    training_result = solver.train(test_maze, start, goal)
    print(f"학습 완료: 성공률 {training_result['final_success_rate']:.2f}")
    
    # 해결 테스트
    print("해결 테스트...")
    path, result = solver.solve(test_maze, start, goal)
    print(f"결과: {result}")
    print(f"경로 길이: {len(path)}")