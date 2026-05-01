"""
Trainers - модуль с реализациями различных типов обучения
"""

from .supervised_trainer import SupervisedTrainer
from .genetic_trainer import GeneticTrainer

__all__ = ["SupervisedTrainer", "GeneticTrainer"]
