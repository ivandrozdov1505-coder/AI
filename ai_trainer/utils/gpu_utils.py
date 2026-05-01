"""
GPU утилиты для управления CUDA и памятью
Поддержка CUDA 12 с автодетектом и fallback на CPU
"""

import torch
import logging
from typing import Dict, Any, Optional, List
import subprocess
import re

logger = logging.getLogger(__name__)


class GPUManager:
    """
    Менеджер GPU для управления устройствами CUDA и памятью
    """
    
    def __init__(self, device_ids: Optional[List[int]] = None, max_memory_fraction: float = 0.9):
        """
        Инициализация GPU менеджера
        
        Args:
            device_ids: Список ID GPU для использования. Если None, используется все доступные
            max_memory_fraction: Максимальная доля памяти GPU для выделения
        """
        self.device_ids = device_ids
        self.max_memory_fraction = max_memory_fraction
        self.device = self._select_device()
        
    def _select_device(self) -> torch.device:
        """
        Выбор лучшего доступного устройства
        
        Returns:
            torch.device: Выбранное устройство (cuda или cpu)
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA не доступна. Используем CPU.")
            return torch.device("cpu")
        
        # Получаем количество доступных GPU
        num_gpus = torch.cuda.device_count()
        logger.info(f"Обнаружено GPU: {num_gpus}")
        
        # Логируем информацию о каждом GPU
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
        
        # Выбираем GPU
        if self.device_ids is None or len(self.device_ids) == 0:
            selected_gpu = 0
        else:
            selected_gpu = min(self.device_ids[0], num_gpus - 1)
        
        # Проверяем версию CUDA
        cuda_version = torch.version.cuda
        if cuda_version:
            logger.info(f"Версия CUDA: {cuda_version}")
            if not cuda_version.startswith("12"):
                logger.warning(
                    f"Рекомендуется CUDA 12.x для оптимальной производительности. "
                    f"Текущая версия: {cuda_version}"
                )
        
        device = torch.device(f"cuda:{selected_gpu}")
        logger.info(f"Используемое устройство: {device}")
        
        # Настраиваем параметры для TF32 (доступно в Ampere и новее)
        if torch.cuda.is_bf16_supported():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 включен для ускорения матричных операций")
        
        return device
    
    def get_device(self) -> torch.device:
        """Получить текущее устройство"""
        return self.device
    
    def is_cuda_available(self) -> bool:
        """Проверить доступность CUDA"""
        return torch.cuda.is_available() and self.device.type == "cuda"
    
    def empty_cache(self):
        """Очистка кэша CUDA"""
        if self.is_cuda_available():
            torch.cuda.empty_cache()
            logger.debug("Кэш CUDA очищен")
    
    def get_memory_info(self) -> Dict[str, float]:
        """
        Получить информацию о памяти GPU
        
        Returns:
            Dict с ключами: allocated, reserved, free (в GB)
        """
        if not self.is_cuda_available():
            return {"allocated": 0.0, "reserved": 0.0, "free": 0.0}
        
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
        free = total - allocated
        
        return {
            "allocated": round(allocated, 2),
            "reserved": round(reserved, 2),
            "free": round(free, 2),
            "total": round(total, 2)
        }
    
    def check_memory_safety(self, required_gb: float = 0.0) -> bool:
        """
        Проверка безопасности выделения памяти
        
        Args:
            required_gb: Требуемый объем памяти в GB
            
        Returns:
            bool: True если выделение безопасно
        """
        if not self.is_cuda_available():
            return True
        
        mem_info = self.get_memory_info()
        available = mem_info["free"] * self.max_memory_fraction
        
        if required_gb > 0:
            is_safe = required_gb <= available
            if not is_safe:
                logger.warning(
                    f"Недостаточно памяти GPU. Требуется: {required_gb:.2f} GB, "
                    f"Доступно: {available:.2f} GB"
                )
            return is_safe
        
        # Предупреждение если занято больше 90%
        occupancy = 1 - (mem_info["free"] / mem_info["total"])
        if occupancy > 0.9:
            logger.warning(f"Заполнено {occupancy*100:.1f}% памяти GPU. Риск OOM!")
            return False
        
        return True
    
    def move_to_device(self, obj: Any) -> Any:
        """
        Переместить тензор или модель на устройство
        
        Args:
            obj: Тензор, модель или модуль PyTorch
            
        Returns:
            Объект на целевом устройстве
        """
        if hasattr(obj, 'to'):
            return obj.to(self.device)
        return obj
    
    def optimize_batch_size(self, initial_batch_size: int, model: torch.nn.Module, 
                           sample_input: torch.Tensor) -> int:
        """
        Автоматический подбор размера батча для предотвращения OOM
        
        Args:
            initial_batch_size: Начальный размер батча
            model: Модель для тестирования
            sample_input: Пример входных данных
            
        Returns:
            Оптимальный размер батча
        """
        if not self.is_cuda_available():
            return initial_batch_size
        
        batch_size = initial_batch_size
        original_training_mode = model.training
        model.eval()
        
        try:
            while batch_size > 1:
                try:
                    test_input = sample_input[:batch_size].to(self.device)
                    with torch.autocast(device_type='cuda', enabled=True):
                        _ = model(test_input)
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"OOM при batch_size={batch_size}. Уменьшаем...")
                        batch_size //= 2
                        self.empty_cache()
                    else:
                        raise
        finally:
            model.train(original_training_mode)
            self.empty_cache()
        
        if batch_size < initial_batch_size:
            logger.info(f"Оптимальный batch_size: {batch_size} (был {initial_batch_size})")
        
        return batch_size


def check_cuda() -> Dict[str, Any]:
    """
    Проверка доступности и версии CUDA
    
    Returns:
        Dict с информацией о CUDA
    """
    result = {
        "available": torch.cuda.is_available(),
        "version": torch.version.cuda,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": []
    }
    
    if result["available"]:
        for i in range(result["device_count"]):
            result["devices"].append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_gb": torch.cuda.get_device_properties(i).total_memory / (1024**3),
                "compute_capability": torch.cuda.get_device_capability(i)
            })
    
    return result


def get_device_info() -> str:
    """
    Получить строку с информацией об устройстве
    
    Returns:
        Форматированная строка с информацией
    """
    cuda_info = check_cuda()
    
    if not cuda_info["available"]:
        return "CPU (CUDA недоступна)"
    
    devices_str = ", ".join([f"{d['name']} ({d['memory_gb']:.1f}GB)" for d in cuda_info["devices"]])
    return f"CUDA {cuda_info['version']} | GPU: {devices_str}"
