"""
Supervised Trainer - обучение с учителем (классификация, регрессия)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable
import logging

from core.trainer_base import BaseTrainer, TrainingMode

logger = logging.getLogger(__name__)


class SupervisedTrainer(BaseTrainer):
    """
    Тренер для обучения с учителем
    
    Поддерживает:
    - Классификацию (бинарную и многоклассовую)
    - Регрессию
    - Кастомные функции потерь
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        config: Dict[str, Any],
        train_data: Any,
        val_data: Any,
        loss_fn: Optional[nn.Module] = None,
        checkpoint_dir: str = "./checkpoints",
        log_callback: Optional[Callable[[str, str], None]] = None
    ):
        """
        Инициализация supervised тренера
        
        Args:
            model: PyTorch модель
            device: Устройство
            config: Конфигурация
            train_data: Обучающие данные (DataLoader или список)
            val_data: Валидационные данные
            loss_fn: Функция потерь (по умолчанию CrossEntropyLoss)
            checkpoint_dir: Директория чекпоинтов
            log_callback: Callback для логов в UI
        """
        super().__init__(model, device, config, checkpoint_dir, log_callback)
        
        self.train_data = train_data
        self.val_data = val_data
        
        # Функция потерь
        if loss_fn:
            self.loss_fn = loss_fn
        else:
            # Автовыбор функции потерь
            num_classes = config.get('model', {}).get('params', {}).get('num_classes', 2)
            if num_classes == 1:
                self.loss_fn = nn.BCEWithLogitsLoss()
            elif num_classes == 2:
                self.loss_fn = nn.BCEWithLogitsLoss()
            else:
                self.loss_fn = nn.CrossEntropyLoss()
        
        self.optimizer = None
        
        self._log("INFO", f"SupervisedTrainer инициализирован. Loss: {self.loss_fn.__class__.__name__}")
    
    def init_optimizer(self) -> torch.optim.Optimizer:
        """Инициализация оптимизатора"""
        lr = self.config.get('training', {}).get('learning_rate', 0.0001)
        weight_decay = self.config.get('training', {}).get('weight_decay', 0.01)
        optimizer_type = self.config.get('training', {}).get('optimizer', 'adamw').lower()
        
        if optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            momentum = self.config.get('training', {}).get('momentum', 0.9)
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:  # adamw по умолчанию
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        
        self._log("INFO", f"Оптимизатор: {self.optimizer.__class__.__name__} (lr={lr})")
        return self.optimizer
    
    def train_step(self, batch) -> Dict[str, float]:
        """Один шаг обучения"""
        self.optimizer.zero_grad()
        
        # Распаковка батча
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
        elif isinstance(batch, dict):
            inputs = batch.get('input_ids', batch.get('text', batch.get('image_path')))
            targets = batch.get('label', batch.get('targets'))
        else:
            raise ValueError(f"Неизвестный формат батча: {type(batch)}")
        
        # Forward pass
        with torch.autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu', 
                          enabled=self.config.get('gpu', {}).get('memory_management', {}).get('mixed_precision', False)):
            outputs = self.model(inputs)
            
            # Вычисление损失
            if targets.dtype == torch.long and outputs.dim() > 1:
                loss = self.loss_fn(outputs, targets)
            else:
                loss = self.loss_fn(outputs.squeeze(), targets.float())
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        clip_value = self.config.get('training', {}).get('gradient_clipping', 1.0)
        if clip_value:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
        
        self.optimizer.step()
        
        # Метрики
        metrics = {'loss': loss.item()}
        
        # Accuracy для классификации
        with torch.no_grad():
            if outputs.dim() > 1:
                preds = outputs.argmax(dim=-1)
            else:
                preds = (outputs.sigmoid() > 0.5).long()
            
            if targets.dim() > 1 and targets.dim() == outputs.dim():
                targets_flat = targets.argmax(dim=-1)
            else:
                targets_flat = targets
            
            accuracy = (preds == targets_flat).float().mean().item()
            metrics['accuracy'] = accuracy
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Валидация модели"""
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for batch in self.val_data:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
            elif isinstance(batch, dict):
                inputs = batch.get('input_ids', batch.get('text', batch.get('image_path')))
                targets = batch.get('label', batch.get('targets'))
            else:
                continue
            
            with torch.no_grad():
                outputs = self.model(inputs)
                
                if targets.dtype == torch.long and outputs.dim() > 1:
                    loss = self.loss_fn(outputs, targets)
                else:
                    loss = self.loss_fn(outputs.squeeze(), targets.float())
                
                total_loss += loss.item()
                
                # Accuracy
                if outputs.dim() > 1:
                    preds = outputs.argmax(dim=-1)
                else:
                    preds = (outputs.sigmoid() > 0.5).long()
                
                if targets.dim() > 1 and targets.dim() == outputs.dim():
                    targets_flat = targets.argmax(dim=-1)
                else:
                    targets_flat = targets
                
                accuracy = (preds == targets_flat).float().mean().item()
                total_accuracy += accuracy
                num_batches += 1
        
        if num_batches == 0:
            return {'loss': 0.0, 'accuracy': 0.0}
        
        return {
            'val_loss': total_loss / num_batches,
            'val_accuracy': total_accuracy / num_batches
        }
    
    def get_data_loader(self, split: str = 'train'):
        """Получение DataLoader"""
        if split == 'train':
            return self.train_data
        elif split == 'val':
            return self.val_data
        else:
            raise ValueError(f"Неизвестный сплит: {split}")
