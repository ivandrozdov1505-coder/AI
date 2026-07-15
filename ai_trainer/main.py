#!/usr/bin/env python3
"""
AI Trainer Platform - Точка входа в приложение
Запуск: python main.py

Автоматически создает виртуальное окружение и устанавливает зависимости.
Для NVIDIA GPU автоматически используется requirements-cuda.txt.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Добавляем проект в path
sys.path.insert(0, str(Path(__file__).parent))

# Bootstrap: автоматическая установка зависимостей
if os.environ.get('AI_TRAINER_BOOTSTRAPPED') != '1':
    # Проверяем, запущены ли из .venv
    in_venv = (hasattr(sys, 'real_prefix') or 
               (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
    
    if not in_venv:
        # Запускаем bootstrap
        try:
            from utils.bootstrap import bootstrap
            bootstrap()
        except Exception as e:
            print(f"⚠️  Bootstrap failed: {e}")
            print("Please install dependencies manually:")
            print("  pip install -r requirements.txt")
            print("  или для CUDA:")
            print("  pip install -r requirements-cuda.txt")


def main():
    """Основная функция запуска"""
    parser = argparse.ArgumentParser(
        description="AI Trainer Platform - Платформа для обучения нейросетей"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="./configs/default_config.yaml",
        help="Путь к конфигурационному файлу"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Хост для веб-интерфейса"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=7860,
        help="Порт для веб-интерфейса"
    )
    
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Использовать только CPU (без GPU)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Режим отладки"
    )
    
    args = parser.parse_args()
    
    # Настройка логгера
    log_level = logging.DEBUG if args.debug else logging.INFO
    
    from utils.logger import setup_logger
    setup_logger(
        name="ai_trainer",
        level="DEBUG" if args.debug else "INFO",
        log_dir="./logs",
        console=True,
        file=True
    )
    
    logger = logging.getLogger("ai_trainer")
    logger.info("=" * 60)
    logger.info("AI Trainer Platform v1.0")
    logger.info("=" * 60)
    
    # Проверка CUDA
    if not args.cpu_only:
        from utils.gpu_utils import check_cuda, get_device_info, GPUManager
        cuda_info = check_cuda()
        
        if cuda_info['available']:
            logger.info(f"✅ CUDA доступна: {get_device_info()}")
        else:
            logger.warning("⚠️ CUDA недоступна, будет использоваться CPU")
            logger.info("Для GPU поддержки установите PyTorch с CUDA:")
            logger.info("  pip install -r requirements-cuda.txt")
            logger.info("Или вручную:")
            logger.info("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    
    # Инициализация менеджера устройств
    from utils.device import get_auto_device
    device = get_auto_device()
    logger.info(f"📌 Используемое устройство: {device}")
    
    # Загрузка конфигурации
    config_path = Path(args.config)
    if config_path.exists():
        logger.info(f"📄 Конфигурация: {config_path}")
    else:
        logger.warning(f"⚠️ Конфигурация не найдена: {config_path}")
        logger.info("Будет создана конфигурация по умолчанию")
    
    # Запуск приложения
    try:
        from ui.app import launch_app
        
        logger.info(f"🚀 Запуск веб-интерфейса на http://{args.host}:{args.port}")
        logger.info("Нажмите Ctrl+C для остановки")
        
        launch_app(server_name=args.host, server_port=args.port)
        
    except KeyboardInterrupt:
        logger.info("\n👋 Остановка по запросу пользователя")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    logger.info("✅ Работа завершена")


if __name__ == "__main__":
    main()
