#!/usr/bin/env python3
"""
AI Trainer Platform - Точка входа в приложение
Запуск: python main.py
"""

import sys
import argparse
import logging
from pathlib import Path

# Добавляем проект в path
sys.path.insert(0, str(Path(__file__).parent))


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
        from utils.gpu_utils import check_cuda, get_device_info
        cuda_info = check_cuda()
        
        if cuda_info['available']:
            logger.info(f"✅ CUDA доступна: {get_device_info()}")
        else:
            logger.warning("⚠️ CUDA недоступна, будет использоваться CPU")
            logger.info("Для GPU поддержки установите PyTorch с CUDA:")
            logger.info("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    
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
