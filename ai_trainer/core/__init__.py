"""
Core модуль - основные компоненты платформы
"""

from .trainer_base import BaseTrainer, TrainingState
from .model_manager import ModelManager
from .data_manager import DataManager
from .checkpoint_manager import CheckpointManager

__all__ = [
    "BaseTrainer",
    "TrainingState",
    "ModelManager",
    "DataManager",
    "CheckpointManager"
]
