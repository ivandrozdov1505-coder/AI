"""
Менеджер чекпоинтов - сохранение и загрузка состояний обучения
"""

import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import shutil

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Управление чекпоинтами модели и состояния обучения
    
    Функции:
    - Автосохранение каждые N шагов
    - Сохранение лучшего чекпоинта
    - История чекпоинтов
    - Очистка старых чекпоинтов
    """
    
    def __init__(self, checkpoint_dir: str, max_keep: int = 5):
        """
        Инициализация менеджера чекпоинтов
        
        Args:
            checkpoint_dir: Директория для сохранения чекпоинтов
            max_keep: Максимальное количество хранимых чекпоинтов
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_keep = max_keep
        self.checkpoint_history: List[Dict[str, Any]] = []
        self.best_metric: float = float('inf')
        self.best_checkpoint_path: Optional[Path] = None
        
        # Создаем директорию
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Загружаем существующие чекпоинты
        self._scan_existing_checkpoints()
    
    def _scan_existing_checkpoints(self):
        """Сканирование существующих чекпоинтов"""
        for path in self.checkpoint_dir.glob("checkpoint_*.pt"):
            try:
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                self.checkpoint_history.append({
                    'path': str(path),
                    'epoch': checkpoint.get('epoch', 0),
                    'step': checkpoint.get('step', 0),
                    'metric': checkpoint.get('best_metric', float('inf')),
                    'timestamp': checkpoint.get('timestamp', 0)
                })
                
                # Проверка на лучший чекпоинт
                if path.name == "checkpoint_best.pt":
                    self.best_checkpoint_path = path
                    self.best_metric = checkpoint.get('best_metric', float('inf'))
                    
            except Exception as e:
                logger.warning(f"Ошибка загрузки чекпоинта {path}: {e}")
        
        # Сортировка по времени
        self.checkpoint_history.sort(key=lambda x: x['timestamp'])
        logger.info(f"Найдено {len(self.checkpoint_history)} существующих чекпоинтов")
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metric: float,
        config: Dict[str, Any],
        scheduler: Optional[Any] = None,
        is_best: bool = False
    ) -> Path:
        """
        Сохранение чекпоинта
        
        Args:
            model: Модель
            optimizer: Оптимизатор
            epoch: Текущая эпоха
            step: Текущий шаг
            metric: Текущая метрика
            config: Конфигурация
            scheduler: Планировщик (опционально)
            is_best: Является ли лучшим чекпоинтом
            
        Returns:
            Путь к сохраненному файлу
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_metric': metric,
            'config': config,
            'timestamp': datetime.now().timestamp()
        }
        
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Имя файла
        if is_best or metric < self.best_metric:
            filename = "checkpoint_best.pt"
            self.best_metric = metric
            self.best_checkpoint_path = self.checkpoint_dir / filename
        else:
            filename = f"checkpoint_{timestamp}_epoch{epoch}_step{step}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Сохранение
        torch.save(checkpoint, checkpoint_path)
        
        # Обновление истории
        self.checkpoint_history.append({
            'path': str(checkpoint_path),
            'epoch': epoch,
            'step': step,
            'metric': metric,
            'timestamp': checkpoint['timestamp']
        })
        
        # Очистка старых чекпоинтов
        self._cleanup_old_checkpoints()
        
        logger.info(f"Чекпоинт сохранен: {checkpoint_path}")
        return checkpoint_path
    
    def load(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Загрузка чекпоинта
        
        Args:
            model: Модель для загрузки весов
            optimizer: Оптимизатор для загрузки состояния
            scheduler: Планировщик для загрузки состояния
            checkpoint_path: Путь к конкретному чекпоинту (или последний)
            
        Returns:
            Информация о загруженном чекпоинте
        """
        if checkpoint_path:
            path = Path(checkpoint_path)
        elif self.best_checkpoint_path:
            path = self.best_checkpoint_path
        elif self.checkpoint_history:
            path = Path(self.checkpoint_history[-1]['path'])
        else:
            raise ValueError("Нет доступных чекпоинтов для загрузки")
        
        if not path.exists():
            raise FileNotFoundError(f"Чекпоинт не найден: {path}")
        
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        
        # Загрузка модели
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Загрузка оптимизатора
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Загрузка планировщика
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        info = {
            'path': str(path),
            'epoch': checkpoint.get('epoch', 0),
            'step': checkpoint.get('step', 0),
            'metric': checkpoint.get('best_metric', float('inf')),
            'config': checkpoint.get('config', {})
        }
        
        logger.info(f"Чекпоинт загружен: {path} (эпоха {info['epoch']})")
        return info
    
    def load_latest(self, model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Optional[Dict[str, Any]]:
        """Загрузка последнего чекпоинта"""
        if not self.checkpoint_history:
            return None
        return self.load(model, optimizer)
    
    def _cleanup_old_checkpoints(self):
        """Удаление старых чекпоинтов"""
        # Исключаем лучший чекпоинт из удаления
        protected = {str(self.best_checkpoint_path)} if self.best_checkpoint_path else set()
        
        # Сортируем по времени (новые первые)
        sorted_checkpoints = sorted(
            self.checkpoint_history,
            key=lambda x: x['timestamp'],
            reverse=True
        )
        
        # Удаляем старые
        for cp in sorted_checkpoints[self.max_keep-1:]:
            if cp['path'] not in protected:
                try:
                    Path(cp['path']).unlink()
                    logger.debug(f"Удален старый чекпоинт: {cp['path']}")
                except Exception as e:
                    logger.warning(f"Не удалось удалить чекпоинт {cp['path']}: {e}")
        
        # Обновляем историю
        self.checkpoint_history = [
            cp for cp in sorted_checkpoints[:self.max_keep]
            if cp['path'] in protected or cp in sorted_checkpoints[:self.max_keep]
        ]
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Получение истории чекпоинтов"""
        return self.checkpoint_history.copy()
    
    def export_for_inference(self, output_path: str):
        """
        Экспорт модели для инференса (только веса модели)
        
        Args:
            output_path: Путь для сохранения
        """
        if not self.best_checkpoint_path:
            raise ValueError("Нет лучшего чекпоинта")
        
        checkpoint = torch.load(self.best_checkpoint_path, map_location='cpu', weights_only=True)
        
        inference_model = {
            'model_state_dict': checkpoint['model_state_dict'],
            'config': checkpoint.get('config', {}),
            'metric': checkpoint.get('best_metric', None)
        }
        
        torch.save(inference_model, output_path)
        logger.info(f"Модель экспортирована для инференса: {output_path}")
    
    def get_info(self) -> Dict[str, Any]:
        """Получение информации о менеджере чекпоинтов"""
        return {
            'checkpoint_dir': str(self.checkpoint_dir),
            'total_checkpoints': len(self.checkpoint_history),
            'best_metric': self.best_metric if self.best_metric != float('inf') else None,
            'best_checkpoint': str(self.best_checkpoint_path) if self.best_checkpoint_path else None,
            'max_keep': self.max_keep
        }
