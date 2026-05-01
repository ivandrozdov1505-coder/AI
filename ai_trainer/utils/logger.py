"""
Модуль логирования с цветным выводом и сохранением в файлы
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import threading


class ColoredFormatter(logging.Formatter):
    """
    Цветной форматтер для консольного вывода логов
    """
    
    # ANSI коды цветов
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        super().__init__(fmt, datefmt)
        
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Добавляем цвет к уровню логирования
        record.levelname = f"{color}{record.levelname}{reset}"
        
        return super().format(record)


# Глобальное хранилище для UI логов
_ui_log_handler: Optional['UILogHandler'] = None
_ui_log_lock = threading.Lock()


class UILogHandler(logging.Handler):
    """
    Обработчик логов для передачи в UI
    Сохраняет последние N записей для отображения
    """
    
    def __init__(self, max_lines: int = 1000):
        super().__init__()
        self.max_lines = max_lines
        self.logs: list[Dict[str, Any]] = []
        self.callbacks: list[callable] = []
    
    def emit(self, record: logging.LogRecord):
        msg = self.format(record)
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname.replace('\033[0m', '').replace('\033[36m', '')
                                     .replace('\033[32m', '').replace('\033[33m', '')
                                     .replace('\033[31m', '').replace('\033[35m', ''),
            'message': msg,
            'logger': record.name
        }
        
        with _ui_log_lock:
            self.logs.append(log_entry)
            # Удаляем старые логи если превышен лимит
            if len(self.logs) > self.max_lines:
                self.logs = self.logs[-self.max_lines:]
            
            # Уведомляем подписчиков
            for callback in self.callbacks:
                try:
                    callback(log_entry)
                except Exception:
                    pass
    
    def get_logs(self, level: Optional[str] = None, limit: int = 100) -> list[Dict[str, Any]]:
        """Получить логи с фильтрацией по уровню"""
        with _ui_log_lock:
            logs = self.logs.copy()
        
        if level:
            logs = [log for log in logs if log['level'] == level]
        
        return logs[-limit:]
    
    def clear(self):
        """Очистить логи"""
        with _ui_log_lock:
            self.logs.clear()
    
    def register_callback(self, callback: callable):
        """Зарегистрировать callback для новых логов"""
        self.callbacks.append(callback)


def setup_logger(
    name: str = "ai_trainer",
    level: str = "INFO",
    log_dir: Optional[str] = None,
    console: bool = True,
    file: bool = True,
    rotation_max_bytes: int = 10*1024*1024,
    rotation_backup_count: int = 5
) -> logging.Logger:
    """
    Настройка логгера с консольным и файловым выводом
    
    Args:
        name: Имя логгера
        level: Уровень логирования
        log_dir: Директория для сохранения логов
        console: Вывод в консоль
        file: Сохранение в файл
        rotation_max_bytes: Максимальный размер файла лога
        rotation_backup_count: Количество резервных копий
        
    Returns:
        Настроенный logger
    """
    global _ui_log_handler
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Очищаем существующие обработчики
    logger.handlers.clear()
    
    # Формат сообщений
    format_str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Консольный обработчик с цветами
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_formatter = ColoredFormatter(format_str, date_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Файловый обработчик с ротацией
    if file and log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"{name}_{timestamp}.log"
        
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=rotation_max_bytes,
            backupCount=rotation_backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(format_str, date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Логи сохраняются в: {log_file}")
    
    # UI обработчик
    with _ui_log_lock:
        if _ui_log_handler is None:
            _ui_log_handler = UILogHandler(max_lines=1000)
            _ui_log_handler.setLevel(getattr(logging, level.upper()))
            ui_formatter = logging.Formatter(format_str, date_format)
            _ui_log_handler.setFormatter(ui_formatter)
    
    logger.addHandler(_ui_log_handler)
    
    return logger


def get_logger(name: str = "ai_trainer") -> logging.Logger:
    """Получить существующий логгер или создать новый"""
    return logging.getLogger(name)


def get_ui_logs(level: Optional[str] = None, limit: int = 100) -> list[Dict[str, Any]]:
    """Получить логи для отображения в UI"""
    if _ui_log_handler:
        return _ui_log_handler.get_logs(level, limit)
    return []


def clear_ui_logs():
    """Очистить UI логи"""
    if _ui_log_handler:
        _ui_log_handler.clear()
