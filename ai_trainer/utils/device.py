"""
Модуль для автоопределения устройства (CUDA/CPU) с фолбэком
Поддержка CUDA 12+ с информативным логированием
"""

import torch
import logging

logger = logging.getLogger(__name__)


def get_auto_device(preferred: str = "auto") -> torch.device:
    """
    Автоопределение лучшего доступного устройства с фолбэком на CPU
    
    Args:
        preferred: Предпочтительное устройство ("auto", "cuda", "cpu")
        
    Returns:
        torch.device: Выбранное устройство
    """
    if preferred == "cpu":
        logger.info("📌 Принудительно используем CPU (по запросу пользователя)")
        return torch.device("cpu")
    
    if preferred == "cuda":
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda or "unknown"
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            logger.info(f"✅ Используем CUDA {cuda_version} | GPU: {gpu_name}")
            return torch.device("cuda")
        else:
            logger.warning("⚠️ Запрошен CUDA, но он недоступен. Фолбэк на CPU.")
            return torch.device("cpu")
    
    # Auto режим - автоматический выбор
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda or "unknown"
        gpu_count = torch.cuda.device_count()
        
        # Логируем информацию о GPU
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"✅ Обнаружен GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
        
        # Проверяем версию CUDA
        if not cuda_version.startswith(("12", "11")):
            logger.warning(
                f"⚠️ Версия CUDA {cuda_version} может быть устаревшей. "
                f"Рекомендуется CUDA 12.x для оптимальной производительности."
            )
        
        selected_gpu = 0
        device = torch.device(f"cuda:{selected_gpu}")
        logger.info(f"🚀 Автоматически выбрано устройство: {device}")
        
        # Включаем TF32 для карт Ampere и новее
        if torch.cuda.is_bf16_supported():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("⚡ TF32 включен для ускорения матричных операций")
        
        return device
    else:
        logger.warning("⚠️ CUDA недоступна. Используем CPU.")
        logger.info(
            "💡 Для включения GPU поддержки установите PyTorch с CUDA:\n"
            "   pip install -r requirements-cuda.txt\n"
            "   или\n"
            "   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
        )
        return torch.device("cpu")


def check_cuda_availability() -> dict:
    """
    Проверка доступности CUDA и возврат подробной информации
    
    Returns:
        dict: Информация о CUDA доступности
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


def print_device_summary():
    """Вывод сводной информации об устройстве в лог"""
    cuda_info = check_cuda_availability()
    
    if not cuda_info["available"]:
        logger.info("=" * 50)
        logger.info("📌 Устройство: CPU")
        logger.info("   CUDA: Недоступна")
        logger.info("=" * 50)
        return
    
    logger.info("=" * 50)
    logger.info("📌 Устройство: GPU (CUDA)")
    logger.info(f"   Версия CUDA: {cuda_info['version']}")
    logger.info(f"   Количество GPU: {cuda_info['device_count']}")
    
    for device in cuda_info["devices"]:
        logger.info(f"   GPU {device['id']}: {device['name']} ({device['memory_gb']:.1f} GB)")
    
    logger.info("=" * 50)
