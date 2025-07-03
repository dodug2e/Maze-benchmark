"""
PPO (Proximal Policy Optimization) 알고리즘 구현 - 최적화 버전
미로 벤치마크용 - 랩 세미나 버전

주요 개선사항:
1. 모델 저장/로드 지원
2. 벡터화된 환경 지원
3. 효율적인 메모리 사용
4. 프로파일러 통합
5. 조기 종료 및 커리큘럼 학습
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class PPOResult:
    """PPO 알고리즘 실행 결과"""
    algorithm: str = "PPO"
    maze_id: str = ""
    maze_size: Tuple[int, int] = (0, 0)
    execution_time: float = 0.0
    power_consumption: float = 0.0
    vram_usage: float = 0.0
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    solution_found: bool = False
    solution_length: int = 0
    total_steps: int = 0
    max_steps: int = 0
    failure_reason: str = ""
    path: List[Tuple[int, int]] = field(default_factory=list)
    training_episodes: int = 0
    training_time: float = 0.0
    average_reward: float = 0.0
    convergence_episode: int = 0
    policy_loss_history: List[float] = field(default_factory=list)
    value_loss_history: List[float] = field(default_factory=list)
    entropy_history: List[float] = field(default_factory=list)
    inference_only: bool = False

class ActorCriticNetwork(nn.Module):
    """Actor-Critic 네트워크 - 최적화 버전"""
    
    def __init__(self, input_channels: int = 3, action_size: int = 4,
                 hidden_size: int = 256, use_lstm: bool = False):
        super(ActorCriticNetwork, self).__init__()
        
        self.use_lstm = use_lstm
        
        # Efficient CNN backbone
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d(4)
        
        feature_size = 64 * 4 * 4
        
        # LSTM for temporal dependencies (optional)
        if use_lstm:
            self.lstm = nn.LSTM(feature_size, hidden_size, batch_first=True)
            self.hidden_size = hidden_size
        else:
            self.fc_features = nn.Linear(feature_size, hidden_size)
            
        # Separate heads
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, hidden=None):
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Pooling and flatten
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Process features
        if self.use_lstm:
            x = x.unsqueeze(1)  # Add sequence dimension
            x, hidden = self.lstm(x, hidden)
            x = x.squeeze(1)
        else:
            x = F.relu(self.fc_features(x))
            hidden = None
        
        # Actor and critic outputs
        action_logits = self.actor_head(x)
        action_probs = F.softmax(action_logits, dim=-1)
        state_value = self.critic_head(x)
        
        return action_probs, state_value, hidden

class RolloutBuffer:
    """Optimized rollout buffer with GAE"""
    
    def __init__(self, buffer_size: int = 2048):
        self.buffer_size = buffer_size
        self.reset()
        
    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = None
        self.returns = None
        self.position = 0
        
    def add(self, state, action, reward, value, log_prob, done):
        if len(self.states) < self.buffer_size:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.values.append(value)
            self.log_probs.append(log_prob)
            self.dones.append(done)
        else:
            # Circular buffer
            self.states[self.position] = state
            self.actions[self.position] = action
            self.rewards[self.position] = reward
            self.values[self.position] = value
            self.log_probs[self.position] = log_prob
            self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.buffer_size
        
    def compute_returns_and_advantages(self, last_value, gamma=0.99, lam=0.95):
        """Compute GAE and returns"""
        rewards = np.array(self.rewards)
        values = np.array([v.cpu().numpy() for v in self.values])
        dones = np.array(self.dones)
        
        # Add last value for bootstrapping
        values = np.append(values, last_value)
        
        # GAE computation
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        
        returns = advantages + values[:-1]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        self.advantages = advantages
        self.returns = returns
        
    def get_batch(self, batch_size: int = 64):
        """Get minibatch for training"""
        n_samples = len(self.states)
        indices = np.random.permutation(n_samples)
        
        for start_idx in range(0, n_samples, batch_size):
            batch_idx = indices[start_idx:start_idx + batch_size]
            
            yield {
                'states': torch.stack([self.states[i] for i in batch_idx]),
                'actions': torch.tensor([self.actions[i] for i in batch_idx], dtype=torch.long),
                'old_log_probs': torch.stack([self.log_probs[i] for i in batch_idx]),
                'advantages': torch.tensor(self.advantages[batch_idx], dtype=torch.float32),
                'returns': torch.tensor(self.returns[batch_idx], dtype=torch.float32)
            }

class MazeEnvironment:
    """미로 환경 - 최적화 버전"""
    
    def __init__(self, maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.current_pos = start
        self.rows, self.cols = maze.shape
        
        # Action space
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.action_size = len(self.actions)
        
        # State buffer for efficiency
        self._state_buffer = np.zeros((3, self.rows, self.cols), dtype=np.float32)
        
        # Tracking
        self.visited = set()
        self.steps = 0
        self.max_steps = int(self.rows * self.cols * 1.5)
        
        # Precompute distance map
        self._compute_distance_map()
        
    def _compute_distance_map(self):
        """Compute Manhattan distance to goal"""
        self.distance_map = np.zeros((self.rows, self.cols))
        for r in range(self.rows):
            for c in range(self.cols):
                if self.maze[r, c] == 0:  # Not a wall
                    self.distance_map[r, c] = abs(r - self.goal[0]) + abs(c - self.goal[1])
                else:
                    self.distance_map[r, c] = float('inf')
    
    def reset(self):
        self.current_pos = self.start
        self.visited = {self.start}
        self.steps = 0
        return self.get_state()
    
    def get_state(self):
        """Get current state efficiently"""
        self._state_buffer.fill(0)
        self._state_buffer[0] = self.maze
        self._state_buffer[1, self.current_pos[0], self.current_pos[1]] = 1.0
        self._state_buffer[2, self.goal[0], self.goal[1]] = 1.0
        
        return torch.from_numpy(self._state_buffer.copy()).unsqueeze(0)
    
    def step(self, action: int):
        self.steps += 1
        
        dx, dy = self.actions[action]
        new_pos = (self.current_pos[0] + dx, self.current_pos[1] + dy)
        
        # Calculate reward
        if new_pos == self.goal:
            reward = 100.0
            done = True
        elif not self.is_valid_position(new_pos):
            reward = -5.0
            done = False
        else:
            # Distance-based reward
            old_dist = self.distance_map[self.current_pos]
            new_dist = self.distance_map[new_pos]
            
            if new_pos in self.visited:
                reward = -1.0
            else:
                reward = 2.0 * (old_dist - new_dist) - 0.1
            
            self.current_pos = new_pos
            self.visited.add(new_pos)
            done = False
        
        if self.steps >= self.max_steps:
            done = True
            
        return self.get_state(), reward, done
    
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return (0 <= r < self.rows and 0 <= c < self.cols and self.maze[r, c] == 0)

class PPOAgent:
    """PPO Agent - Optimized version"""
    
    def __init__(self,
                 action_size: int = 4,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 clip_ratio: float = 0.2,
                 value_coeff: float = 0.5,
                 entropy_coeff: float = 0.01,
                 max_grad_norm: float = 0.5,
                 use_lstm: bool = False,
                 device: str = None):
        
        self.action_size = action_size
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Network
        self.network = ActorCriticNetwork(
            action_size=action_size, 
            use_lstm=use_lstm
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.scaler = GradScaler()  # For mixed precision
        
        # LSTM hidden state
        self.hidden = None
        
        # Training stats
        self.training_step = 0
        self.policy_losses = deque(maxlen=100)
        self.value_losses = deque(maxlen=100)
        self.entropies = deque(maxlen=100)
        
    def act(self, state, deterministic: bool = False):
        """Select action"""
        state = state.to(self.device)
        
        with torch.no_grad():
            action_probs, value, self.hidden = self.network(state, self.hidden)
            
        if deterministic:
            action = action_probs.argmax(dim=-1)
            log_prob = torch.log(action_probs.gather(1, action.unsqueeze(-1)))
        else:
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob, value.squeeze()
    
    def update(self, rollout_buffer: RolloutBuffer, n_epochs: int = 4):
        """PPO update"""
        for epoch in range(n_epochs):
            for batch in rollout_buffer.get_batch():
                states = batch['states'].to(self.device)
                actions = batch['actions'].to(self.device)
                old_log_probs = batch['old_log_probs'].to(self.device)
                advantages = batch['advantages'].to(self.device)
                returns = batch['returns'].to(self.device)
                
                # Forward pass with mixed precision
                with autocast():
                    action_probs, values, _ = self.network(states)
                    dist = Categorical(action_probs)
                    new_log_probs = dist.log_prob(actions)
                    entropy = dist.entropy().mean()
                    
                    # Ratio for PPO
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    
                    # Clipped objective
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss
                    value_loss = F.mse_loss(values.squeeze(), returns)
                    
                    # Total loss
                    loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Record stats
                self.policy_losses.append(policy_loss.item())
                self.value_losses.append(value_loss.item())
                self.entropies.append(entropy.item())
        
        self.training_step += 1
        
    def save(self, path: str):
        """Save model"""
        torch.save({
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'stats': {
                'policy_losses': list(self.policy_losses),
                'value_losses': list(self.value_losses),
                'entropies': list(self.entropies)
            }
        }, path)
        logger.info(f"Model saved to {path}")
        
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.training_step = checkpoint.get('training_step', 0)
        logger.info(f"Model loaded from {path}")

class PPOSolver:
    """PPO Solver - Optimized version"""
    
    def __init__(self,
                 n_steps: int = 2048,
                 n_epochs: int = 4,
                 batch_size: int = 64,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 clip_ratio: float = 0.2,
                 max_episodes: int = 1000,
                 pretrained_path: Optional[str] = None):
        
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_episodes = max_episodes
        
        # Initialize agent
        self.agent = PPOAgent(
            lr=lr, gamma=gamma, lam=lam, 
            clip_ratio=clip_ratio
        )
        
        # Load pretrained model if available
        if pretrained_path and Path(pretrained_path).exists():
            self.agent.load(pretrained_path)
            logger.info("Loaded pretrained PPO model")
            
        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(n_steps)
        
    def train_single_env(self, env: MazeEnvironment, max_timesteps: int = 100000):
        """Train on single environment"""
        state = env.reset()
        episode_rewards = []
        episode_reward = 0
        episodes = 0
        
        for timestep in range(max_timesteps):
            # Act
            action, log_prob, value = self.agent.act(state)
            next_state, reward, done = env.step(action)
            
            # Store transition
            self.rollout_buffer.add(
                state.squeeze(0), action, reward, 
                value, log_prob, done
            )
            
            episode_reward += reward
            state = next_state
            
            # Episode done
            if done:
                episode_rewards.append(episode_reward)
                episodes += 1
                
                # Reset
                state = env.reset()
                episode_reward = 0
                self.agent.hidden = None  # Reset LSTM state
                
                # Check convergence
                if len(episode_rewards) >= 10:
                    recent_rewards = episode_rewards[-10:]
                    if all(r > 50 for r in recent_rewards):
                        logger.info(f"Converged at episode {episodes}")
                        break
                
                if episodes >= self.max_episodes:
                    break
            
            # Update policy
            if (timestep + 1) % self.n_steps == 0:
                with torch.no_grad():
                    _, last_value, _ = self.agent.network(state.to(self.agent.device))
                    last_value = last_value.item()
                
                self.rollout_buffer.compute_returns_and_advantages(last_value, self.agent.gamma, self.agent.lam)
                self.agent.update(self.rollout_buffer, self.n_epochs)
                self.rollout_buffer.reset()
        
        return {
            'episodes': episodes,
            'episode_rewards': episode_rewards,
            'converged': len(episode_rewards) >= 10 and all(r > 50 for r in episode_rewards[-10:])
        }
    
    def test(self, env: MazeEnvironment, max_steps: int = 500):
        """Test trained model"""
        state = env.reset()
        path = [env.current_pos]
        self.agent.hidden = None
        
        for step in range(max_steps):
            action, _, _ = self.agent.act(state, deterministic=True)
            state, _, done = env.step(action)
            path.append(env.current_pos)
            
            if done:
                break
                
        success = (env.current_pos == env.goal)
        return success, path, len(path)
    
    def solve(self, maze: np.ndarray, start: Tuple[int, int], 
             goal: Tuple[int, int], max_steps: int = 10000,
             train: bool = True) -> PPOResult:
        """Solve maze with PPO"""
        # 프로파일러 초기화 및 시작
        profiler = None
        try:
            from utils.profiler import get_profiler
            profiler = get_profiler()
            profiler.start_monitoring()
            logger.info("Performance monitoring started")
        except ImportError:
            logger.warning("Profiler not available")
        except Exception as e:
            logger.warning(f"Failed to start profiler: {e}")
        
        start_time = time.time()
        result = PPOResult()
        result.maze_size = maze.shape
        result.max_steps = max_steps
        result.inference_only = not train
        
        # Validate start and goal
        if maze[start[0], start[1]] == 1 or maze[goal[0], goal[1]] == 1:
            result.failure_reason = "시작점 또는 목표점이 벽입니다"
            result.execution_time = time.time() - start_time
            
            # 프로파일러 정리
            if profiler:
                profiler.stop_monitoring()
            
            return result
        
        try:
            # Create environment
            env = MazeEnvironment(maze, start, goal)
            
            if train:
                # Training mode
                training_start = time.time()
                train_stats = self.train_single_env(env)
                result.training_time = time.time() - training_start
                result.training_episodes = train_stats['episodes']
                result.average_reward = np.mean(train_stats['episode_rewards'][-10:]) if train_stats['episode_rewards'] else 0
                
                # Find convergence episode
                for i, reward in enumerate(train_stats['episode_rewards']):
                    if reward > 50:
                        result.convergence_episode = i
                        break
                
                # Get loss history
                result.policy_loss_history = list(self.agent.policy_losses)
                result.value_loss_history = list(self.agent.value_losses)
                result.entropy_history = list(self.agent.entropies)
            
            # Test the trained model
            success, path, steps = self.test(env)
            
            result.solution_found = success
            result.path = path
            result.solution_length = len(path) if success else 0
            result.total_steps = steps
            
            if not success:
                result.failure_reason = "최대 스텝 내에 해결하지 못했습니다"
                    
        except Exception as e:
            result.failure_reason = f"PPO 실행 오류: {e}"
            logger.error(f"PPO execution error: {e}", exc_info=True)
        
        finally:
            # 프로파일러 메트릭 수집
            if profiler:
                try:
                    profiler.stop_monitoring()
                    summary = profiler.get_summary_stats()
                    
                    # 피크/평균 메트릭 사용
                    result.vram_usage = summary.get('vram_used_mb', {}).get('peak', 0.0)
                    result.gpu_utilization = summary.get('gpu_percent', {}).get('avg', 0.0)
                    result.cpu_utilization = summary.get('cpu_percent', {}).get('avg', 0.0)
                    result.power_consumption = summary.get('power_watts', {}).get('avg', 0.0)
                    
                    logger.info(f"Performance metrics - VRAM Peak: {result.vram_usage:.1f}MB, "
                              f"GPU Avg: {result.gpu_utilization:.1f}%, "
                              f"Power Avg: {result.power_consumption:.1f}W")
                except Exception as e:
                    logger.warning(f"Failed to collect profiler metrics: {e}")
        
        result.execution_time = time.time() - start_time
        return result
    
    def save_model(self, path: str):
        """Save model"""
        self.agent.save(path)
    
    def load_model(self, path: str):
        """Load model"""
        self.agent.load(path)


class VectorizedPPOSolver:
    """PPO with vectorized environments for efficient training"""
    
    def __init__(self, 
                 n_envs: int = 4,
                 n_steps: int = 512,
                 config: Dict[str, Any] = None):
        
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.config = config or {}
        
        # Create base solver
        self.solver = PPOSolver(
            n_steps=n_steps * n_envs,  # Total steps across all envs
            **self.config
        )
        
    def train_on_batch(self, envs: List[MazeEnvironment], 
                      total_timesteps: int = 100000):
        """Train on multiple environments in parallel"""
        n_envs = len(envs)
        states = [env.reset() for env in envs]
        episode_rewards = [0.0] * n_envs
        episode_lengths = [0] * n_envs
        
        all_episode_rewards = []
        successes = 0
        
        for timestep in range(0, total_timesteps, n_envs):
            # Collect actions for all environments
            actions, log_probs, values = [], [], []
            
            for i, state in enumerate(states):
                action, log_prob, value = self.solver.agent.act(state)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
            
            # Step all environments
            for i, (env, action) in enumerate(zip(envs, actions)):
                next_state, reward, done = env.step(action)
                
                # Store experience
                self.solver.rollout_buffer.add(
                    states[i].squeeze(0), action, reward,
                    values[i], log_probs[i], done
                )
                
                episode_rewards[i] += reward
                episode_lengths[i] += 1
                
                if done:
                    all_episode_rewards.append(episode_rewards[i])
                    if env.current_pos == env.goal:
                        successes += 1
                    
                    # Reset
                    states[i] = env.reset()
                    episode_rewards[i] = 0
                    episode_lengths[i] = 0
                else:
                    states[i] = next_state
            
            # Update policy
            if (timestep + n_envs) % self.n_steps == 0:
                # Bootstrap value for non-terminal states
                last_values = []
                for i, state in enumerate(states):
                    with torch.no_grad():
                        _, value, _ = self.solver.agent.network(state.to(self.solver.agent.device))
                        last_values.append(value.item() if episode_lengths[i] > 0 else 0)
                
                # Average last value
                last_value = np.mean(last_values)
                
                self.solver.rollout_buffer.compute_returns_and_advantages(
                    last_value, self.solver.agent.gamma, self.solver.agent.lam
                )
                self.solver.agent.update(self.solver.rollout_buffer, self.solver.n_epochs)
                self.solver.rollout_buffer.reset()
        
        return {
            'total_episodes': len(all_episode_rewards),
            'success_rate': successes / len(all_episode_rewards) if all_episode_rewards else 0,
            'average_reward': np.mean(all_episode_rewards) if all_episode_rewards else 0
        }


class BatchPPOTrainer:
    """Batch training system for PPO"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.solver = PPOSolver(
            n_steps=config.get('n_steps', 2048),
            n_epochs=config.get('n_epochs', 4),
            batch_size=config.get('batch_size', 64),
            lr=config.get('lr', 3e-4),
            pretrained_path=config.get('pretrained_path')
        )
        
        self.model_dir = Path(config.get('model_dir', 'models/ppo'))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def train_on_dataset(self, dataset_loader, subset: str = 'train'):
        """Train on entire dataset"""
        from utils.maze_io import load_maze_as_array, load_metadata
        
        sample_ids = dataset_loader.get_sample_ids(subset)
        
        # Curriculum learning: sort by maze size
        if self.config.get('use_curriculum', True):
            maze_sizes = []
            for sample_id in sample_ids[:min(100, len(sample_ids))]:
                try:
                    maze, _ = load_maze_as_array(sample_id, subset)
                    maze_sizes.append((sample_id, maze.size))
                except:
                    continue
            
            # Sort by size
            maze_sizes.sort(key=lambda x: x[1])
            
            # Reorder sample_ids based on size
            if maze_sizes:
                size_order = [x[0] for x in maze_sizes]
                remaining = [sid for sid in sample_ids if sid not in size_order]
                sample_ids = size_order + remaining
        
        # Training parameters
        batch_size = self.config.get('env_batch_size', 4)
        save_interval = self.config.get('save_interval', 100)
        
        # Use vectorized solver for efficiency
        vec_solver = VectorizedPPOSolver(
            n_envs=batch_size,
            config=self.config
        )
        
        # Copy loaded model if any
        if self.solver.agent.training_step > 0:
            vec_solver.solver.agent = self.solver.agent
        
        total_trained = 0
        success_count = 0
        
        logger.info(f"Starting PPO training on {len(sample_ids)} mazes")
        
        for i in range(0, len(sample_ids), batch_size):
            batch_ids = sample_ids[i:i+batch_size]
            envs = []
            
            # Create environments
            for sample_id in batch_ids:
                try:
                    maze, metadata = load_maze_as_array(sample_id, subset)
                    start = tuple(metadata['entrance'])
                    goal = tuple(metadata['exit'])
                    env = MazeEnvironment(maze, start, goal)
                    envs.append(env)
                except Exception as e:
                    logger.warning(f"Failed to load maze {sample_id}: {e}")
                    continue
            
            if envs:
                # Train on batch
                stats = vec_solver.train_on_batch(envs, total_timesteps=50000)
                success_count += int(stats['success_rate'] * len(envs))
                total_trained += len(envs)
                
                logger.info(
                    f"Batch {i//batch_size + 1}: "
                    f"Success rate: {stats['success_rate']:.2%}, "
                    f"Avg reward: {stats['average_reward']:.2f}"
                )
            
            # Save checkpoint
            if (i + batch_size) % save_interval == 0:
                checkpoint_path = self.model_dir / f"checkpoint_batch_{i//batch_size}.pth"
                vec_solver.solver.save_model(str(checkpoint_path))
        
        # Save final model
        final_path = self.model_dir / "final_model.pth"
        vec_solver.solver.save_model(str(final_path))
        
        return {
            'total_samples': total_trained,
            'success_count': success_count,
            'success_rate': success_count / total_trained if total_trained > 0 else 0
        }
    
    def evaluate_on_dataset(self, dataset_loader, subset: str = 'test'):
        """Evaluate on dataset"""
        from utils.maze_io import load_maze_as_array, load_metadata
        
        sample_ids = dataset_loader.get_sample_ids(subset)
        results = []
        
        logger.info(f"Evaluating PPO on {len(sample_ids)} test mazes")
        
        for i, sample_id in enumerate(sample_ids):
            try:
                maze, metadata = load_maze_as_array(sample_id, subset)
                start = tuple(metadata['entrance'])
                goal = tuple(metadata['exit'])
                
                # Run inference only
                result = self.solver.solve(maze, start, goal, train=False)
                result.maze_id = sample_id
                results.append(result)
                
                if (i + 1) % 100 == 0:
                    success_rate = sum(r.solution_found for r in results) / len(results)
                    logger.info(f"Progress: {i+1}/{len(sample_ids)}, "
                              f"Success rate: {success_rate:.2%}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate maze {sample_id}: {e}")
                continue
        
        # Calculate statistics
        success_rate = sum(r.solution_found for r in results) / len(results) if results else 0
        avg_path_length = np.mean([r.solution_length for r in results if r.solution_found]) if results else 0
        avg_execution_time = np.mean([r.execution_time for r in results]) if results else 0
        
        return {
            'results': results,
            'success_rate': success_rate,
            'avg_path_length': avg_path_length,
            'avg_execution_time': avg_execution_time
        }


# Framework integration
class PPOMazeSolver:
    """PPO solver wrapper for framework integration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.solver = PPOSolver(
            n_steps=config.get('n_steps', 2048),
            n_epochs=config.get('n_epochs', 4),
            batch_size=config.get('batch_size', 64),
            lr=config.get('lr', 3e-4),
            pretrained_path=config.get('pretrained_path'),
            max_episodes=config.get('max_episodes', 0) if config.get('pretrained_path') else 500
        )
        self.algorithm_name = "PPO"
    
    def solve(self, maze: np.ndarray, start: Tuple[int, int], 
             goal: Tuple[int, int], max_steps: int = 10000) -> PPOResult:
        """Solve maze interface"""
        train = self.config.get('pretrained_path') is None
        return self.solver.solve(maze, start, goal, max_steps, train=train)
    
    def train(self, dataset_loader):
        """Train on dataset"""
        trainer = BatchPPOTrainer(self.config)
        return trainer.train_on_dataset(dataset_loader)
    
    def evaluate(self, dataset_loader, subset: str = 'test'):
        """Evaluate on dataset"""
        trainer = BatchPPOTrainer(self.config)
        return trainer.evaluate_on_dataset(dataset_loader, subset)


# Utility functions
def create_ppo_config(
    n_steps: int = 2048,
    n_epochs: int = 4,
    batch_size: int = 64,
    lr: float = 3e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_ratio: float = 0.2,
    max_episodes: int = 500,
    pretrained_path: Optional[str] = None,
    use_curriculum: bool = True,
    env_batch_size: int = 4,
    model_dir: str = "models/ppo"
) -> Dict[str, Any]:
    """Create PPO configuration"""
    return {
        'n_steps': n_steps,
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'lr': lr,
        'gamma': gamma,
        'lam': lam,
        'clip_ratio': clip_ratio,
        'max_episodes': max_episodes,
        'pretrained_path': pretrained_path,
        'use_curriculum': use_curriculum,
        'env_batch_size': env_batch_size,
        'model_dir': model_dir,
        'save_interval': 100
    }


def run_ppo_benchmark(sample_id: str, subset: str = "test",
                     pretrained_path: Optional[str] = None) -> PPOResult:
    """Run PPO benchmark on single maze"""
    from utils.maze_io import load_maze_as_array, load_metadata
    
    try:
        # Load data
        maze, metadata = load_maze_as_array(sample_id, subset)
        start = tuple(metadata['entrance'])
        goal = tuple(metadata['exit'])
        
        # Create config
        config = create_ppo_config(
            max_episodes=0 if pretrained_path else 500,
            pretrained_path=pretrained_path
        )
        
        # Run solver
        solver = PPOMazeSolver(config)
        result = solver.solve(maze, start, goal)
        result.maze_id = sample_id
        
        return result
        
    except Exception as e:
        result = PPOResult()
        result.maze_id = sample_id
        result.failure_reason = f"벤치마크 실행 오류: {e}"
        return result


# Test code
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test maze
    test_maze = np.array([
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ])
    
    start = (0, 0)
    goal = (6, 6)
    
    print("=" * 50)
    print("PPO 최적화 버전 테스트")
    print("=" * 50)
    
    # Create config
    config = create_ppo_config(
        n_steps=512,
        max_episodes=100,
        env_batch_size=1
    )
    
    # Create solver
    solver = PPOMazeSolver(config)
    
    # Solve maze
    print("\n테스트 미로 해결 중...")
    result = solver.solve(test_maze, start, goal)
    
    # Print results
    print(f"\n=== 결과 ===")
    print(f"해결 성공: {result.solution_found}")
    print(f"경로 길이: {result.solution_length}")
    print(f"총 스텝: {result.total_steps}")
    print(f"실행 시간: {result.execution_time:.2f}초")
    print(f"훈련 시간: {result.training_time:.2f}초")
    print(f"훈련 에피소드: {result.training_episodes}")
    print(f"평균 보상: {result.average_reward:.2f}")
    
    if result.solution_found:
        print(f"\n경로 (처음 10개):")
        for i, pos in enumerate(result.path[:10]):
            print(f"  스텝 {i}: {pos}")
    else:
        print(f"\n실패 원인: {result.failure_reason}")
    
    # GPU info
    if torch.cuda.is_available():
        print(f"\nGPU 정보:")
        print(f"  디바이스: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM 사용: {result.vram_usage:.1f}MB")
        print(f"  GPU 사용률: {result.gpu_utilization:.1f}%")
    
    print("\n테스트 완료!")