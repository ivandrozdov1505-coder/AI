"""
Базовый класс для всех типов обучения
Поддержка бесконечного/конечного режима, чекпоинтов, безопасной остановки
"""

import torch
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from pathlib import Path


class TrainingMode(Enum):
    """Режимы обучения"""
    FINITE = "finite"      # Конечное (N эпох)
    INFINITE = "infinite"  # Бесконечное (до остановки)


class StopMode(Enum):
    """Режимы остановки"""
    SOFT = "soft"    # Мягкая: завершить текущий шаг и сохранить
    HARD = "hard"    # Жесткая: немедленная остановка


@dataclass
class TrainingState:
    """Состояние процесса обучения"""
    is_running: bool = False
    is_paused: bool = False
    current_epoch: int = 0
    current_step: int = 0
    total_epochs: int = 0
    total_steps: int = 0
    best_metric: float = float('inf')
    metrics_history: List[Dict[str, float]] = field(default_factory=list)
    stop_requested: bool = False
    stop_mode: StopMode = StopMode.SOFT
    start_time: float = 0.0
    last_checkpoint_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация состояния"""
        return {
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'total_epochs': self.total_epochs if self.total_epochs > 0 else None,
            'best_metric': self.best_metric if self.best_metric != float('inf') else None,
            'metrics_count': len(self.metrics_history),
            'elapsed_time': time.time() - self.start_time if self.start_time > 0 else 0
        }


class BaseTrainer(ABC):
    """
    Абстрактный базовый класс для всех тренеров
    
    Реализует:
    - Управление состоянием обучения
    - Безопасную остановку (мягкую/жесткую)
    - Систему чекпоинтов
    - Логирование метрик
    - Поддержку CUDA
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        config: Dict[str, Any],
        checkpoint_dir: str = "./checkpoints",
        log_callback: Optional[Callable[[str, str], None]] = None
    ):
        """
        Инициализация тренера
        
        Args:
            model: PyTorch модель
            device: Устройство для обучения (cuda/cpu)
            config: Конфигурация обучения
            checkpoint_dir: Директория для чекпоинтов
            log_callback: Callback для логирования в UI
        """
        self.model = model
        self.device = device
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_callback = log_callback
        
        self.state = TrainingState()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
        # Перемещаем модель на устройство
        self.model = self.model.to(self.device)
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Создаем директорию для чекпоинтов
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def init_optimizer(self) -> torch.optim.Optimizer:
        """Инициализация оптимизатора (должен быть реализован в наследнике)"""
        pass
    
    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, float]:
        """
        Один шаг обучения
        
        Args:
            batch: Батч данных
            
        Returns:
            Dict с метриками (loss, accuracy, etc.)
        """
        pass
    
    @abstractmethod
    def validate(self) -> Dict[str, float]:
        """
        Валидация модели
        
        Returns:
            Dict с метриками валидации
        """
        pass
    
    @abstractmethod
    def get_data_loader(self, split: str = 'train'):
        """Получение DataLoader для указанного сплита"""
        pass
    
    def _log(self, level: str, message: str):
        """Логирование с callback в UI"""
        self.logger.log(getattr(logging, level.upper()), message)
        if self.log_callback:
            try:
                self.log_callback(level, message)
            except Exception:
                pass
    
    def request_stop(self, mode: StopMode = StopMode.SOFT):
        """
        Запрос остановки обучения
        
        Args:
            mode: Режим остановки (SOFT или HARD)
        """
        with self._lock:
            self.state.stop_requested = True
            self.state.stop_mode = mode
            self._log("INFO", f"Запрошена остановка ({mode.value} режим)")
            
            if mode == StopMode.HARD:
                self._stop_event.set()
    
    def clear_stop_request(self):
        """Очистка запроса остановки"""
        with self._lock:
            self.state.stop_requested = False
            self.state.stop_mode = StopMode.SOFT
            self._stop_event.clear()
    
    def should_stop(self) -> bool:
        """Проверка флага остановки"""
        with self._lock:
            return self.state.stop_requested and self.state.stop_mode == StopMode.HARD
    
    def should_stop_soft(self) -> bool:
        """Проверка флага мягкой остановки"""
        with self._lock:
            return self.state.stop_requested
    
    def _save_checkpoint(self, optimizer: torch.optim.Optimizer, 
                        scheduler: Optional[Any] = None,
                        is_best: bool = False) -> Path:
        """
        Сохранение чекпоинта
        
        Args:
            optimizer: Оптимизатор
            scheduler: Планировщик (опционально)
            is_best: Является ли лучшим чекпоинтом
            
        Returns:
            Путь к сохраненному файлу
        """
        checkpoint = {
            'epoch': self.state.current_epoch,
            'step': self.state.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_metric': self.state.best_metric,
            'metrics_history': self.state.metrics_history[-100:],  # Последние 100 метрик
            'config': self.config,
            'timestamp': time.time()
        }
        
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Имя файла
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_{timestamp}_epoch{self.state.current_epoch}.pt"
        
        if is_best:
            filename = "checkpoint_best.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Сохраняем
        torch.save(checkpoint, checkpoint_path)
        self._log("INFO", f"Чекпоинт сохранен: {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str, 
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None) -> bool:
        """
        Загрузка чекпоинта
        
        Args:
            checkpoint_path: Путь к файлу чекпоинта
            optimizer: Оптимизатор для загрузки состояния
            scheduler: Планировщик для загрузки состояния
            
        Returns:
            True если загрузка успешна
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            self._log("ERROR", f"Чекпоинт не найден: {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.state.current_epoch = checkpoint['epoch']
            self.state.current_step = checkpoint['step']
            self.state.best_metric = checkpoint.get('best_metric', float('inf'))
            
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self._log("INFO", f"Чекпоинт загружен: {checkpoint_path} (эпоха {checkpoint['epoch']})")
            return True
            
        except Exception as e:
            self._log("ERROR", f"Ошибка загрузки чекпоинта: {e}")
            return False
    
    def _update_metrics(self, metrics: Dict[str, float], is_train: bool = True):
        """Обновление истории метрик"""
        metrics['epoch'] = self.state.current_epoch
        metrics['step'] = self.state.current_step
        metrics['is_train'] = is_train
        metrics['timestamp'] = time.time()
        
        with self._lock:
            self.state.metrics_history.append(metrics)
    
    def _check_early_stopping(self, metric: float, patience: int, 
                             min_delta: float = 0.0001) -> bool:
        """
        Проверка ранней остановки
        
        Returns:
            True если нужно остановить
        """
        if self.state.best_metric == float('inf'):
            self.state.best_metric = metric
            return False
        
        if metric < self.state.best_metric - min_delta:
            self.state.best_metric = metric
            return False
        
        # Считаем сколько эпох нет улучшения
        recent_metrics = [m.get('val_loss', m.get('loss', float('inf'))) 
                         for m in self.state.metrics_history[-patience:] 
                         if not m.get('is_train', True)]
        
        if len(recent_metrics) >= patience:
            if all(m >= self.state.best_metric - min_delta for m in recent_metrics):
                self._log("INFO", "Ранняя остановка: нет улучшений")
                return True
        
        return False
    
    def fit(self, num_epochs: int = 100, mode: TrainingMode = TrainingMode.FINITE,
           checkpoint_every: int = 10, save_best: bool = True,
           early_stopping_patience: Optional[int] = None) -> TrainingState:
        """
        Основной цикл обучения
        
        Args:
            num_epochs: Количество эпох
            mode: Режим обучения (FINITE/INFINITE)
            checkpoint_every: Сохранять чекпоинт каждые N эпох
            save_best: Сохранять лучший чекпоинт
            early_stopping_patience: Патентс для ранней остановки
            
        Returns:
            Финальное состояние обучения
        """
        self._log("INFO", f"Начало обучения: {num_epochs} эпох, режим={mode.value}")
        
        # Инициализация
        optimizer = self.init_optimizer()
        scheduler = None  # Можно добавить планировщик
        
        self.state.is_running = True
        self.state.total_epochs = num_epochs if mode == TrainingMode.FINITE else 0
        self.state.start_time = time.time()
        
        train_loader = self.get_data_loader('train')
        
        try:
            epoch = 0
            while True:
                # Проверка режима
                if mode == TrainingMode.FINITE and epoch >= num_epochs:
                    break
                
                # Проверка остановки
                if self.should_stop_soft():
                    self._log("INFO", "Мягкая остановка: завершение эпохи...")
                    if self.state.stop_mode == StopMode.HARD:
                        break
                
                self.state.current_epoch = epoch
                self._log("INFO", f"Эпоха {epoch + 1}/{num_epochs if mode == TrainingMode.FINITE else '∞'}")
                
                # Обучение
                self.model.train()
                epoch_metrics = []
                
                for batch_idx, batch in enumerate(train_loader):
                    # Проверка жесткой остановки
                    if self._stop_event.is_set():
                        self._log("WARNING", "Жесткая остановка!")
                        break
                    
                    # Перемещаем батч на устройство
                    if isinstance(batch, (tuple, list)):
                        batch = tuple(b.to(self.device) if hasattr(b, 'to') else b for b in batch)
                    elif hasattr(batch, 'to'):
                        batch = batch.to(self.device)
                    
                    # Шаг обучения
                    step_metrics = self.train_step(batch)
                    epoch_metrics.append(step_metrics)
                    
                    self.state.current_step += 1
                
                # Агрегация метрик эпохи
                avg_train_metrics = {
                    k: sum(m[k] for m in epoch_metrics) / len(epoch_metrics)
                    for k in epoch_metrics[0].keys()
                }
                self._update_metrics(avg_train_metrics, is_train=True)
                
                # Валидация
                self._log("INFO", "Валидация...")
                self.model.eval()
                with torch.no_grad():
                    val_metrics = self.validate()
                    self._update_metrics(val_metrics, is_train=False)
                
                # Логирование
                self._log("INFO", 
                    f"Train Loss: {avg_train_metrics.get('loss', 'N/A'):.4f}, "
                    f"Val Loss: {val_metrics.get('loss', 'N/A'):.4f}")
                
                # Чекпоинты
                if save_best and val_metrics.get('loss', float('inf')) < self.state.best_metric:
                    self._save_checkpoint(optimizer, scheduler, is_best=True)
                    self.state.best_metric = val_metrics.get('loss', float('inf'))
                
                if (epoch + 1) % checkpoint_every == 0:
                    self._save_checkpoint(optimizer, scheduler)
                
                # Ранняя остановка
                if early_stopping_patience:
                    if self._check_early_stopping(
                        val_metrics.get('loss', float('inf')),
                        early_stopping_patience
                    ):
                        break
                
                epoch += 1
                
                # Для бесконечного режима очищаем кэш
                if mode == TrainingMode.INFINITE:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        except Exception as e:
            self._log("ERROR", f"Ошибка обучения: {e}")
            raise
        
        finally:
            self.state.is_running = False
            self._log("INFO", "Обучение завершено")
        
        return self.state
    
    def get_state(self) -> Dict[str, Any]:
        """Получить текущее состояние"""
        return self.state.to_dict()
    
    def cleanup(self):
        """Очистка ресурсов"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._log("INFO", "Ресурсы очищены")
