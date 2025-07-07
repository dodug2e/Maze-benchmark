"""
CNN 모델 구현 - 미로 경로 예측용
RTX 3060 (6GB VRAM) 최적화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class MazePathCNN(nn.Module):
    """미로 경로 예측을 위한 CNN 모델 (가변 크기 지원)"""
    
    def __init__(self, 
                 input_size: int = 224,  # 표준 크기
                 num_classes: int = 4,   # 상하좌우 4방향
                 dropout_rate: float = 0.2):
        super(MazePathCNN, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Feature extraction layers (가변 크기 지원)
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/2, W/2
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/4, W/4
            
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/8, W/8
            
            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))  # 고정 크기로 변환
        )
        
        # Classifier (고정 크기)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파"""
        # Input shape: (batch_size, 1, height, width)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def predict_direction(self, maze: np.ndarray, 
                         current_pos: Tuple[int, int]) -> int:
        """현재 위치에서 최적 방향 예측"""
        self.eval()
        
        # 미로를 텐서로 변환
        maze_tensor = torch.FloatTensor(maze).unsqueeze(0).unsqueeze(0)
        
        # GPU 사용 가능하면 GPU로
        if torch.cuda.is_available():
            maze_tensor = maze_tensor.cuda()
            self = self.cuda()
        
        with torch.no_grad():
            outputs = self.forward(maze_tensor)
            probabilities = F.softmax(outputs, dim=1)
            direction = torch.argmax(probabilities, dim=1).item()
        
        return direction
    
    def get_model_size(self) -> Dict:
        """모델 크기 정보 반환"""
        param_count = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 대략적인 메모리 사용량 계산 (32-bit float 기준)
        memory_mb = (param_count * 4) / (1024 * 1024)
        
        return {
            'total_parameters': param_count,
            'trainable_parameters': trainable_params,
            'estimated_memory_mb': memory_mb
        }


class MazeDataset(torch.utils.data.Dataset):
    """미로 데이터셋 클래스"""
    
    def __init__(self, 
                 maze_loader,
                 sample_ids: list,
                 subset: str = "train",
                 target_size: int = 224,  # 표준 크기로 리사이즈
                 transform=None):
        self.maze_loader = maze_loader
        self.sample_ids = sample_ids
        self.subset = subset
        self.target_size = target_size
        self.transform = transform
        
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        # 미로 데이터 로드
        img, metadata, array = self.maze_loader.load_sample(sample_id, self.subset)
        
        # 배열로 변환
        if array is not None:
            maze_array = array
        else:
            maze_array = self.maze_loader.convert_image_to_array(img)
        
        # 크기 정규화 (패딩 + 리사이즈)
        maze_array = self._resize_maze(maze_array)
        
        # 텐서로 변환
        maze_tensor = torch.FloatTensor(maze_array).unsqueeze(0)  # 채널 차원 추가
        
        # 라벨 생성 (예시: 최단 경로 기반)
        # 실제로는 ACO나 A* 결과를 사용
        label = self._generate_label(maze_array, metadata)
        
        if self.transform:
            maze_tensor = self.transform(maze_tensor)
        
        return maze_tensor, label
    
    def _resize_maze(self, maze_array: np.ndarray) -> np.ndarray:
        """미로 크기를 target_size로 정규화"""
        current_h, current_w = maze_array.shape
        
        # 이미 목표 크기면 그대로 반환
        if current_h == self.target_size and current_w == self.target_size:
            return maze_array
        
        # 정사각형으로 패딩 (긴 쪽에 맞춤)
        max_size = max(current_h, current_w)
        
        # 패딩할 크기 계산
        pad_h = (max_size - current_h) // 2
        pad_w = (max_size - current_w) // 2
        
        # 패딩 (벽으로 채움)
        padded_maze = np.ones((max_size, max_size), dtype=maze_array.dtype)
        padded_maze[pad_h:pad_h+current_h, pad_w:pad_w+current_w] = maze_array
        
        # target_size로 리사이즈
        if max_size != self.target_size:
            from PIL import Image
            
            # PIL Image로 변환하여 리사이즈
            pil_image = Image.fromarray((padded_maze * 255).astype(np.uint8), mode='L')
            resized_image = pil_image.resize((self.target_size, self.target_size), Image.NEAREST)
            
            # 다시 0-1 범위로 변환
            resized_maze = np.array(resized_image, dtype=np.float32) / 255.0
            resized_maze = (resized_maze > 0.5).astype(np.float32)  # 임계값으로 이진화
        else:
            resized_maze = padded_maze.astype(np.float32)
        
        return resized_maze
    
    def _generate_label(self, maze_array: np.ndarray, metadata: dict) -> torch.Tensor:
        """라벨 생성 (임시 구현)"""
        # 실제로는 최적 경로에서 방향 정보를 추출해야 함
        # 여기서는 랜덤 방향으로 임시 구현
        direction = np.random.randint(0, 4)  # 0: 상, 1: 하, 2: 좌, 3: 우
        return torch.LongTensor([direction])


class CNNTrainer:
    """CNN 학습 클래스"""
    
    def __init__(self, 
                 model: MazePathCNN,
                 device: str = "auto",
                 learning_rate: float = 0.001):
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"CNN 모델이 {self.device}에 로드되었습니다.")
        
        # 모델 크기 출력
        model_info = self.model.get_model_size()
        logger.info(f"모델 파라미터: {model_info['total_parameters']:,}")
        logger.info(f"예상 메모리: {model_info['estimated_memory_mb']:.1f}MB")
    
    def train_epoch(self, train_loader, epoch: int):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device).squeeze()
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, '
                           f'Loss: {loss.item():.6f}, '
                           f'Acc: {100.*correct/total:.2f}%')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """검증"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device).squeeze()
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = val_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def save_model(self, filepath: str, epoch: int, loss: float):
        """모델 저장"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, filepath)
        logger.info(f"모델이 저장되었습니다: {filepath}")
    
    def load_model(self, filepath: str):
        """모델 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        logger.info(f"모델이 로드되었습니다: epoch {epoch}, loss {loss:.6f}")
        return epoch, loss


# 사용 예시
if __name__ == "__main__":
    # 모델 생성
    model = MazePathCNN(input_size=200, num_classes=4)
    
    # 모델 정보 출력
    model_info = model.get_model_size()
    print(f"총 파라미터: {model_info['total_parameters']:,}")
    print(f"예상 메모리: {model_info['estimated_memory_mb']:.1f}MB")
    
    # 더미 입력으로 테스트
    dummy_input = torch.randn(1, 1, 200, 200)
    output = model(dummy_input)
    print(f"출력 크기: {output.shape}")
    print(f"예측 방향: {torch.argmax(output, dim=1).item()}")