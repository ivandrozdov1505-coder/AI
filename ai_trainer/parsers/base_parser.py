"""
Базовый класс для всех парсеров
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """Абстрактный базовый класс для парсеров"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.supported_extensions: List[str] = []
    
    @abstractmethod
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Парсинг файла
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Список извлеченных данных
        """
        pass
    
    @abstractmethod
    def can_parse(self, file_path: str) -> bool:
        """Проверка возможности парсинга файла"""
        pass
    
    def validate_file(self, file_path: str) -> bool:
        """Валидация файла перед парсингом"""
        path = Path(file_path)
        return path.exists() and path.is_file()
