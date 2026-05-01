"""
Genetic Trainer - генетические алгоритмы и эволюционная оптимизация
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Callable, List
import logging
import copy
from core.trainer_base import BaseTrainer, TrainingMode, TrainingState

logger = logging.getLogger(__name__)


class GeneticTrainer(BaseTrainer):
    """
    Тренер на основе генетических алгоритмов
    
    Поддерживает:
    - Эволюцию популяции моделей/параметров
    - Различные функции приспособленности (fitness)
    - Мутации и кроссовер
    """
    
    def __init__(
        self,
        model_template: torch.nn.Module,
        device: torch.device,
        config: Dict[str, Any],
        train_data: Any,
        val_data: Any,
        fitness_fn: Optional[Callable] = None,
        checkpoint_dir: str = "./checkpoints",
        log_callback: Optional[Callable[[str, str], None]] = None
    ):
        """
        Инициализация генетического тренера
        
        Args:
            model_template: Шаблон модели для клонирования
            device: Устройство
            config: Конфигурация
            train_data: Обучающие данные
            val_data: Валидационные данные
            fitness_fn: Функция приспособленности
            checkpoint_dir: Директория чекпоинтов
            log_callback: Callback для логов
        """
        # Инициализируем без модели - создадим популяцию
        super().__init__(model_template, device, config, checkpoint_dir, log_callback)
        
        self.train_data = train_data
        self.val_data = val_data
        self.fitness_fn = fitness_fn or self._default_fitness_fn
        
        # Параметры GA
        ga_config = config.get('genetic', {})
        self.population_size = ga_config.get('population_size', 10)
        self.mutation_rate = ga_config.get('mutation_rate', 0.1)
        self.mutation_std = ga_config.get('mutation_std', 0.01)
        self.elite_count = ga_config.get('elite_count', 2)
        self.tournament_size = ga_config.get('tournament_size', 3)
        
        # Популяция
        self.population: List[torch.nn.Module] = []
        self.fitness_scores: List[float] = []
        
        self._log("INFO", f"GeneticTrainer инициализирован. Популяция: {self.population_size}")
    
    def _default_fitness_fn(self, model: torch.nn.Module) -> float:
        """Функция приспособленности по умолчанию (accuracy на валидации)"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_data:
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]
                elif isinstance(batch, dict):
                    inputs = batch.get('input_ids', batch.get('text'))
                    targets = batch.get('label', batch.get('targets'))
                else:
                    continue
                
                outputs = model(inputs)
                
                if outputs.dim() > 1:
                    preds = outputs.argmax(dim=-1)
                else:
                    preds = (outputs.sigmoid() > 0.5).long()
                
                if targets.dim() > 1:
                    targets = targets.argmax(dim=-1)
                
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def init_optimizer(self) -> None:
        """GA не использует оптимизатор в традиционном смысле"""
        return None
    
    def _create_population(self) -> List[torch.nn.Module]:
        """Создание начальной популяции"""
        self.population = []
        
        for i in range(self.population_size):
            # Клонирование модели
            model_copy = copy.deepcopy(self.model)
            
            # Добавление случайности в веса
            with torch.no_grad():
                for param in model_copy.parameters():
                    noise = torch.randn_like(param) * self.mutation_std
                    param.add_(noise)
            
            self.population.append(model_copy)
        
        self._log("INFO", f"Создана популяция из {len(self.population)} моделей")
        return self.population
    
    def _evaluate_population(self) -> List[float]:
        """Оценка приспособленности всей популяции"""
        self.fitness_scores = []
        
        for i, model in enumerate(self.population):
            fitness = self.fitness_fn(model)
            self.fitness_scores.append(fitness)
            
            if i == 0:  # Лог только для первой
                self._log("DEBUG", f"Fitness модели 0: {fitness:.4f}")
        
        return self.fitness_scores
    
    def _selection(self) -> int:
        """Турнирная селекция"""
        indices = np.random.choice(len(self.population), size=self.tournament_size, replace=False)
        best_idx = max(indices, key=lambda i: self.fitness_scores[i])
        return best_idx
    
    def _crossover(self, parent1: torch.nn.Module, parent2: torch.nn.Module) -> torch.nn.Module:
        """Кроссовер между двумя родителями"""
        child = copy.deepcopy(parent1)
        
        with torch.no_grad():
            for (c_param, p1_param, p2_param) in zip(
                child.parameters(), 
                parent1.parameters(), 
                parent2.parameters()
            ):
                # Случайная маска для кроссовера
                mask = torch.rand_like(c_param) > 0.5
                c_param[mask] = p1_param[mask]
                c_param[~mask] = p2_param[~mask]
        
        return child
    
    def _mutate(self, model: torch.nn.Module) -> torch.nn.Module:
        """Мутация модели"""
        mutated = copy.deepcopy(model)
        
        with torch.no_grad():
            for param in mutated.parameters():
                mask = torch.rand_like(param) < self.mutation_rate
                noise = torch.randn_like(param) * self.mutation_std
                param[mask] += noise
        
        return mutated
    
    def _evolve(self):
        """Эволюция популяции"""
        # Оценка
        self._evaluate_population()
        
        # Сортировка по fitness
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        self.population = [self.population[i] for i in sorted_indices]
        self.fitness_scores = [self.fitness_scores[i] for i in sorted_indices]
        
        # Элитизм - сохраняем лучших
        new_population = [copy.deepcopy(self.population[i]) for i in range(self.elite_count)]
        
        # Создание нового поколения
        while len(new_population) < self.population_size:
            # Селекция
            parent1_idx = self._selection()
            parent2_idx = self._selection()
            
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            # Кроссовер
            child = self._crossover(parent1, parent2)
            
            # Мутация
            child = self._mutate(child)
            
            new_population.append(child)
        
        self.population = new_population
    
    def train_step(self, batch) -> Dict[str, float]:
        """Для GA train_step не используется в традиционном смысле"""
        return {'loss': 0.0}
    
    def validate(self) -> Dict[str, float]:
        """Валидация лучшей модели"""
        if not self.population:
            return {'val_loss': 0.0, 'val_accuracy': 0.0}
        
        best_model = self.population[0]
        fitness = self.fitness_fn(best_model)
        
        return {
            'val_loss': 1.0 - fitness,  # Инвертируем для совместимости
            'val_accuracy': fitness
        }
    
    def get_data_loader(self, split: str = 'train'):
        """Получение DataLoader"""
        if split == 'train':
            return self.train_data
        elif split == 'val':
            return self.val_data
        else:
            raise ValueError(f"Неизвестный сплит: {split}")
    
    def fit(self, num_generations: int = 100, **kwargs) -> TrainingState:
        """
        Запуск эволюции
        
        Args:
            num_generations: Количество поколений
        """
        self._log("INFO", f"Запуск эволюции: {num_generations} поколений")
        
        # Инициализация популяции
        self._create_population()
        
        self.state.is_running = True
        self.state.total_epochs = num_generations
        self.state.start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if self.state.start_time:
            self.state.start_time.record()
        
        try:
            for gen in range(num_generations):
                # Проверка остановки
                if self.should_stop_soft():
                    self._log("INFO", "Остановка эволюции...")
                    break
                
                self.state.current_epoch = gen
                
                # Эволюция
                self._evolve()
                
                # Получаем лучшую модель
                best_model = self.population[0]
                best_fitness = self.fitness_scores[0] if self.fitness_scores else 0.0
                
                # Логирование
                self._log("INFO", f"Поколение {gen + 1}/{num_generations}: "
                         f"Best Fitness = {best_fitness:.4f}, "
                         f"Avg Fitness = {np.mean(self.fitness_scores):.4f}")
                
                # Обновление метрик
                self._update_metrics({
                    'fitness': best_fitness,
                    'avg_fitness': np.mean(self.fitness_scores),
                    'std_fitness': np.std(self.fitness_scores)
                }, is_train=True)
                
                # Чекпоинт лучшей модели
                if gen % kwargs.get('checkpoint_every', 10) == 0:
                    # Временно устанавливаем лучшую модель для чекпоинта
                    original_model = self.model
                    self.model = best_model
                    optimizer = None  # GA не использует оптимизатор
                    self._save_checkpoint(optimizer, is_best=True)
                    self.model = original_model
        
        except Exception as e:
            self._log("ERROR", f"Ошибка эволюции: {e}")
            raise
        
        finally:
            self.state.is_running = False
            
            # Сохраняем лучшую модель как основную
            if self.population:
                self.model = copy.deepcopy(self.population[0])
            
            self._log("INFO", "Эволюция завершена")
        
        return self.state
