"""
DQN 신경망 구현
RTX 3060 6GB VRAM 최적화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DQNNetwork(nn.Module):
    """Deep Q-Network"""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: Tuple[int, ...] = (128, 128)):
        """
        Args:
            state_size: 상태 벡터 크기
            action_size: 행동 개수
            hidden_sizes: 은닉층 크기들
        """
        super(DQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # 네트워크 구조 (RTX 3060 메모리 고려하여 경량화)
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)  # 과적합 방지
            ])
            prev_size = hidden_size
        
        # 출력층
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화 (Xavier 초기화)"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """순전파"""
        return self.network(state)
    
    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        ε-greedy 행동 선택
        
        Args:
            state: 현재 상태
            epsilon: 탐색 확률
            
        Returns:
            선택된 행동
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)
        
        # 상태를 텐서로 변환
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Q값 계산
        with torch.no_grad():
            q_values = self.forward(state_tensor)
        
        return q_values.argmax().item()
    
    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Q값 반환 (배치 처리 가능)"""
        return self.forward(state)


class DoubleDQNNetwork(DQNNetwork):
    """Double DQN을 위한 확장"""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: Tuple[int, ...] = (128, 128)):
        super().__init__(state_size, action_size, hidden_sizes)
        
    def get_target_q_value(self, state: torch.Tensor, action: torch.Tensor, target_network: 'DoubleDQNNetwork') -> torch.Tensor:
        """Double DQN을 위한 타겟 Q값 계산"""
        with torch.no_grad():
            # 현재 네트워크로 최적 행동 선택
            best_actions = self.forward(state).argmax(dim=1, keepdim=True)
            
            # 타겟 네트워크로 Q값 계산
            target_q_values = target_network.forward(state)
            target_q = target_q_values.gather(1, best_actions)
            
        return target_q


class DuelingDQNNetwork(nn.Module):
    """Dueling DQN 구현"""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: Tuple[int, ...] = (128, 128)):
        super().__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # 공통 특성 추출 네트워크
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Value stream (상태 가치)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1] // 2, 1)
        )
        
        # Advantage stream (행동 우위)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1] // 2, action_size)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for module in [self.feature_extractor, self.value_stream, self.advantage_stream]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0.01)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """순전파 - Dueling 구조"""
        features = self.feature_extractor(state)
        
        # 상태 가치
        value = self.value_stream(features)
        
        # 행동 우위
        advantage = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
    
    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """ε-greedy 행동 선택"""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.forward(state_tensor)
        
        return q_values.argmax().item()


class CNNDQNNetwork(nn.Module):
    """CNN 기반 DQN (이미지 입력용)"""
    
    def __init__(self, input_channels: int, action_size: int, input_height: int, input_width: int):
        """
        Args:
            input_channels: 입력 채널 수 (1 for grayscale)
            action_size: 행동 개수
            input_height: 입력 이미지 높이
            input_width: 입력 이미지 너비
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.action_size = action_size
        
        # CNN 특성 추출기 (메모리 효율적으로 설계)
        self.conv_layers = nn.Sequential(
            # 첫 번째 conv 블록
            nn.Conv2d(input_channels, 16, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            
            # 두 번째 conv 블록
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # 세 번째 conv 블록  
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # Global Average Pooling (메모리 절약)
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # CNN 출력 크기 계산
        conv_output_size = 32 * 4 * 4  # 512
        
        # 완전연결층
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, action_size)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.01)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파"""
        # CNN 특성 추출
        conv_out = self.conv_layers(x)
        
        # Flatten
        flattened = conv_out.view(conv_out.size(0), -1)
        
        # 완전연결층
        q_values = self.fc_layers(flattened)
        
        return q_values
    
    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """ε-greedy 행동 선택 (이미지 입력)"""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)
        
        # 상태 전처리 (배치 차원 추가)
        if len(state.shape) == 3:  # (H, W, C) -> (1, C, H, W)
            state = np.transpose(state, (2, 0, 1))
            state = np.expand_dims(state, 0)
        elif len(state.shape) == 2:  # (H, W) -> (1, 1, H, W)
            state = np.expand_dims(np.expand_dims(state, 0), 0)
        
        state_tensor = torch.FloatTensor(state)
        
        with torch.no_grad():
            q_values = self.forward(state_tensor)
        
        return q_values.argmax().item()


def create_dqn_network(network_type: str, **kwargs) -> nn.Module:
    """DQN 네트워크 팩토리 함수"""
    
    network_types = {
        'basic': DQNNetwork,
        'double': DoubleDQNNetwork,
        'dueling': DuelingDQNNetwork,
        'cnn': CNNDQNNetwork
    }
    
    if network_type not in network_types:
        raise ValueError(f"Unknown network type: {network_type}. Available: {list(network_types.keys())}")
    
    NetworkClass = network_types[network_type]
    
    try:
        network = NetworkClass(**kwargs)
        logger.info(f"Created {network_type} DQN network")
        return network
    except Exception as e:
        logger.error(f"Failed to create {network_type} network: {e}")
        raise


if __name__ == "__main__":
    # 네트워크 테스트
    print("=== DQN 네트워크 테스트 ===")
    
    # 기본 DQN
    print("1. 기본 DQN 네트워크")
    basic_net = DQNNetwork(state_size=4, action_size=4)
    print(f"파라미터 수: {sum(p.numel() for p in basic_net.parameters())}")
    
    # 테스트 입력
    test_state = torch.randn(1, 4)
    output = basic_net(test_state)
    print(f"출력 모양: {output.shape}")
    
    # Dueling DQN
    print("\n2. Dueling DQN 네트워크") 
    dueling_net = DuelingDQNNetwork(state_size=4, action_size=4)
    print(f"파라미터 수: {sum(p.numel() for p in dueling_net.parameters())}")
    
    output = dueling_net(test_state)
    print(f"출력 모양: {output.shape}")
    
    # CNN DQN  
    print("\n3. CNN DQN 네트워크")
    cnn_net = CNNDQNNetwork(input_channels=1, action_size=4, input_height=64, input_width=64)
    print(f"파라미터 수: {sum(p.numel() for p in cnn_net.parameters())}")
    
    test_image = torch.randn(1, 1, 64, 64)
    output = cnn_net(test_image)
    print(f"출력 모양: {output.shape}")
    
    # 메모리 사용량 추정
    total_params = sum(p.numel() for p in cnn_net.parameters())
    memory_mb = total_params * 4 / (1024 * 1024)  # float32 = 4 bytes
    print(f"\n예상 메모리 사용량: {memory_mb:.1f}MB")
    
    print("\n✅ 모든 네트워크 테스트 완료")