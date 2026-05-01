"""
Менеджер моделей - создание и управление архитектурами нейросетей
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Type
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Менеджер для создания и управления моделями
    Поддерживает различные архитектуры: Transformer, CNN, RNN, MLP
    """
    
    # Реестр архитектур
    ARCHITECTURES: Dict[str, Type[nn.Module]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Декоратор для регистрации архитектуры"""
        def decorator(model_class: Type[nn.Module]):
            cls.ARCHITECTURES[name] = model_class
            logger.info(f"Зарегистрирована архитектура: {name}")
            return model_class
        return decorator
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        Инициализация менеджера моделей
        
        Args:
            config: Конфигурация модели
            device: Устройство для размещения модели
        """
        self.config = config
        self.device = device
        self.model: Optional[nn.Module] = None
    
    def create_model(self, architecture: str = None, params: Dict[str, Any] = None,
                    pretrained: str = None) -> nn.Module:
        """
        Создание модели
        
        Args:
            architecture: Название архитектуры (transformer, cnn, rnn, mlp)
            params: Параметры архитектуры
            pretrained: Путь к предобученным весам или название модели
            
        Returns:
            Созданная модель
        """
        architecture = architecture or self.config.get('architecture', 'mlp')
        params = params or self.config.get('params', {})
        
        logger.info(f"Создание модели: {architecture}")
        
        if architecture == 'transformer':
            self.model = self._create_transformer(params)
        elif architecture == 'cnn':
            self.model = self._create_cnn(params)
        elif architecture == 'rnn':
            self.model = self._create_rnn(params)
        elif architecture == 'mlp':
            self.model = self._create_mlp(params)
        else:
            raise ValueError(f"Неизвестная архитектура: {architecture}")
        
        # Загрузка предобученных весов
        if pretrained:
            self._load_pretrained(pretrained)
        
        # Перемещение на устройство
        self.model = self.model.to(self.device)
        
        # Логирование информации о модели
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Модель создана: {total_params:,} параметров ({trainable_params:,} обучаемых)")
        
        return self.model
    
    def _create_transformer(self, params: Dict[str, Any]) -> nn.Module:
        """Создание трансформера"""
        hidden_size = params.get('hidden_size', 768)
        num_layers = params.get('num_layers', 6)
        num_heads = params.get('num_heads', 8)
        dropout = params.get('dropout', 0.1)
        vocab_size = params.get('vocab_size', 32000)
        max_seq_len = params.get('max_seq_len', 512)
        num_classes = params.get('num_classes', 2)
        
        class SimpleTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, hidden_size))
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=dropout,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.classifier = nn.Linear(hidden_size, num_classes)
            
            def forward(self, x):
                # x: [batch, seq_len]
                x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
                x = self.transformer(x)
                # Global average pooling
                x = x.mean(dim=1)
                return self.classifier(x)
        
        return SimpleTransformer()
    
    def _create_cnn(self, params: Dict[str, Any]) -> nn.Module:
        """Создание CNN для изображений"""
        input_channels = params.get('input_channels', 3)
        num_classes = params.get('num_classes', 10)
        channels = params.get('channels', [32, 64, 128, 256])
        dropout = params.get('dropout', 0.5)
        
        layers = []
        in_channels = input_channels
        
        for out_channels in channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ])
            in_channels = out_channels
        
        layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes)
        ])
        
        return nn.Sequential(*layers)
    
    def _create_rnn(self, params: Dict[str, Any]) -> nn.Module:
        """Создание RNN/LSTM"""
        input_size = params.get('input_size', 300)
        hidden_size = params.get('hidden_size', 512)
        num_layers = params.get('num_layers', 2)
        num_classes = params.get('num_classes', 2)
        dropout = params.get('dropout', 0.3)
        bidirectional = params.get('bidirectional', True)
        
        class SimpleRNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=bidirectional
                )
                self.classifier = nn.Linear(
                    hidden_size * (2 if bidirectional else 1), 
                    num_classes
                )
            
            def forward(self, x):
                # x: [batch, seq_len, input_size]
                _, (h_n, _) = self.rnn(x)
                # Объединяем последний слой
                if bidirectional:
                    h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
                else:
                    h_n = h_n[-1]
                return self.classifier(h_n)
        
        return SimpleRNN()
    
    def _create_mlp(self, params: Dict[str, Any]) -> nn.Module:
        """Создание MLP (полносвязная сеть)"""
        input_size = params.get('input_size', 768)
        hidden_dims = params.get('hidden_dims', [512, 256, 128])
        num_classes = params.get('num_classes', 2)
        dropout = params.get('dropout', 0.3)
        activation = params.get('activation', 'relu')
        
        layers = []
        prev_dim = input_size
        
        # Активация
        activations = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'silu': nn.SiLU,
            'tanh': nn.Tanh
        }
        act_fn = activations.get(activation.lower(), nn.ReLU)
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                act_fn(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        return nn.Sequential(*layers)
    
    def _load_pretrained(self, pretrained: str):
        """Загрузка предобученных весов"""
        if not self.model:
            raise ValueError("Модель не создана")
        
        try:
            state_dict = torch.load(pretrained, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"Предобученные веса загружены: {pretrained}")
        except Exception as e:
            logger.warning(f"Не удалось загрузить веса: {e}")
    
    def get_model(self) -> Optional[nn.Module]:
        """Получить текущую модель"""
        return self.model
    
    def save_model(self, path: str):
        """Сохранение модели"""
        if not self.model:
            raise ValueError("Модель не создана")
        
        torch.save(self.model.state_dict(), path)
        logger.info(f"Модель сохранена: {path}")
    
    def estimate_memory(self, batch_size: int = 32, input_shape: tuple = None) -> Dict[str, float]:
        """
        Оценка потребления памяти моделью
        
        Args:
            batch_size: Размер батча
            input_shape: Форма входных данных
            
        Returns:
            Dict с оценкой памяти (model_mb, batch_mb, total_mb)
        """
        if not self.model:
            return {"model_mb": 0, "batch_mb": 0, "total_mb": 0}
        
        # Размер модели
        model_params = sum(p.numel() for p in self.model.parameters())
        model_mb = model_params * 4 / (1024**2)  # FP32 = 4 bytes
        
        # Оценка размера батча
        if input_shape:
            batch_elements = batch_size
            for dim in input_shape:
                batch_elements *= dim
            batch_mb = batch_elements * 4 / (1024**2)
        else:
            batch_mb = 0
        
        return {
            "model_mb": round(model_mb, 2),
            "batch_mb": round(batch_mb, 2),
            "total_mb": round(model_mb + batch_mb, 2)
        }
