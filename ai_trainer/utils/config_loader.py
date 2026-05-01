"""
Загрузчик конфигураций из YAML/JSON файлов
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Загрузчик и валидатор конфигураций
    Поддерживает YAML и JSON форматы, слияние конфигов
    """
    
    def __init__(self, default_config_path: Optional[str] = None):
        """
        Инициализация загрузчика
        
        Args:
            default_config_path: Путь к конфигурации по умолчанию
        """
        self.default_config: Dict[str, Any] = {}
        
        if default_config_path:
            self.default_config = self.load(default_config_path)
            logger.info(f"Загружена конфигурация по умолчанию: {default_config_path}")
    
    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Загрузка конфигурации из файла
        
        Args:
            path: Путь к файлу конфигурации
            
        Returns:
            Словарь с конфигурацией
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Конфигурация не найдена: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif path.suffix == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Неподдерживаемый формат: {path.suffix}")
        
        logger.debug(f"Загружена конфигурация: {path}")
        return config or {}
    
    def merge(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Глубокое слияние нескольких конфигураций
        
        Args:
            *configs: Словари для слияния (последний имеет наивысший приоритет)
            
        Returns:
            Объединенная конфигурация
        """
        result = {}
        
        for config in configs:
            result = self._deep_merge(result, config)
        
        return result
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Рекурсивное слияние словарей"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get_nested(self, config: Dict[str, Any], path: str, default: Any = None) -> Any:
        """
        Получение вложенного значения по пути
        
        Args:
            config: Конфигурация
            path: Путь в формате "section.subsection.key"
            default: Значение по умолчанию
            
        Returns:
            Значение или default
        """
        keys = path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def validate(self, config: Dict[str, Any], schema: Dict[str, Any] = None) -> bool:
        """
        Базовая валидация конфигурации
        
        Args:
            config: Конфигурация для проверки
            schema: Схема валидации (опционально)
            
        Returns:
            True если конфигурация валидна
        """
        # Проверка обязательных секций
        required_sections = ['general', 'data', 'model', 'training']
        
        for section in required_sections:
            if section not in config:
                logger.warning(f"Отсутствует секция: {section}")
                return False
        
        # Проверка типов данных
        if not isinstance(config.get('general', {}), dict):
            logger.error("Секция 'general' должна быть словарем")
            return False
        
        if not isinstance(config.get('data', {}), dict):
            logger.error("Секция 'data' должна быть словарем")
            return False
        
        logger.info("Конфигурация прошла валидацию")
        return True
    
    def create_default(self, output_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Создание файла конфигурации по умолчанию
        
        Args:
            output_path: Путь для сохранения
            
        Returns:
            Созданная конфигурация
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        default_config = {
            'general': {
                'project_name': 'My AI Project',
                'output_dir': './checkpoints',
                'log_dir': './logs',
                'random_seed': 42,
                'device': 'auto'
            },
            'data': {
                'paths': ['./data'],
                'file_types': ['.txt', '.csv', '.json', '.pdf', '.docx', '.jpg', '.png'],
                'preprocessing': {
                    'text': {
                        'lowercase': True,
                        'remove_special_chars': True,
                        'max_length': 512
                    },
                    'split': {
                        'train': 0.8,
                        'val': 0.1,
                        'test': 0.1
                    }
                }
            },
            'model': {
                'type': 'supervised',
                'architecture': 'transformer',
                'params': {
                    'hidden_size': 768,
                    'num_layers': 6,
                    'dropout': 0.1
                }
            },
            'training': {
                'mode': 'finite',
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.0001,
                'optimizer': 'adamw'
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, allow_unicode=True, default_flow_style=False)
        
        logger.info(f"Создана конфигурация по умолчанию: {output_path}")
        return default_config


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Удобная функция для загрузки конфигурации
    
    Args:
        config_path: Путь к файлу конфигурации
        overrides: Переопределения параметров
        
    Returns:
        Итоговая конфигурация
    """
    loader = ConfigLoader()
    
    configs = [loader.default_config]
    
    if config_path:
        configs.append(loader.load(config_path))
    
    if overrides:
        configs.append(overrides)
    
    return loader.merge(*configs)
