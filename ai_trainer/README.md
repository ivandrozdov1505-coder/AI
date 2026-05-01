# AI Trainer Platform

Платформа для создания, настройки и обучения нейросетей с графическим интерфейсом.

## 📁 Структура проекта

```
ai_trainer/
├── main.py                 # Точка входа в приложение
├── requirements.txt        # Зависимости проекта
├── README.md              # Эта документация
├── configs/
│   └── default_config.yaml # Конфигурация по умолчанию
├── core/
│   ├── __init__.py
│   ├── trainer_base.py    # Базовый класс для всех типов обучения
│   ├── model_manager.py   # Управление моделями и архитектурами
│   ├── data_manager.py    # Управление данными и предобработка
│   └── checkpoint_manager.py # Система чекпоинтов
├── ui/
│   ├── __init__.py
│   └── app.py             # Основное Gradio приложение
├── trainers/
│   ├── __init__.py
│   ├── supervised_trainer.py  # Supervised learning
│   └── genetic_trainer.py     # Генетические алгоритмы
├── parsers/
│   ├── __init__.py
│   ├── base_parser.py     # Базовый парсер
│   ├── text_parser.py     # TXT, CSV, JSON
│   ├── document_parser.py # PDF, DOCX, DOC, XLSX
│   └── image_parser.py    # Изображения с OCR
└── utils/
    ├── __init__.py
    ├── gpu_utils.py       # Утилиты для GPU/CUDA
    ├── logger.py          # Настройка логирования
    └── config_loader.py   # Загрузка конфигураций
```

## 🚀 Быстрый старт

### Установка зависимостей

```bash
# Базовая установка (CPU)
pip install -r requirements.txt

# Для CUDA 12 (GPU с поддержкой CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Запуск приложения

```bash
# Стандартный запуск
python main.py

# С указанием порта
python main.py --port 7860

# Только CPU (без GPU)
python main.py --cpu-only

# Режим отладки
python main.py --debug
```

После запуска откройте браузер по адресу: `http://localhost:7860`

## 🔑 Основные возможности

### 1. Подготовка данных (вкладка 📦)
- **Многоформатная загрузка**: .txt, .csv, .json, .pdf, .docx, .xlsx, изображения
- **Автоматическая предобработка**: токенизация, нормализация, удаление дубликатов
- **Разбиение на сплиты**: train/val/test с настраиваемыми пропорциями
- **Построение словаря**: автоматическое создание vocab для текстовых данных

### 2. Обучение (вкладка 🧠)
- **Конструктор архитектуры**: выбор типа модели (Transformer, CNN, RNN, MLP)
- **Настройка гиперпараметров**: hidden_size, num_layers, learning_rate, batch_size
- **Типы обучения**:
  - Supervised (обучение с учителем)
  - Genetic (генетические алгоритмы)
- **Управление обучением**:
  - ▶️ Запуск
  - 💾 Мягкая остановка (завершение эпохи + сохранение)
  - ⚡ Жесткая остановка (немедленная)

### 3. Мониторинг (вкладка 📊)
- **Статус в реальном времени**: текущая эпоха, метрики
- **Логи**: цветные логи с фильтрацией
- **Автообновление**: каждые 2 секунды

## 🛠 Технические особенности

### Поддержка CUDA 12
- Автодетект GPU и проверка версии CUDA
- Fallback на CPU при отсутствии GPU
- Оптимизация памяти: `torch.cuda.empty_cache()`, gradient checkpointing
- TF32 acceleration для карт Ampere и новее

### Безопасная остановка
```python
from core.trainer_base import StopMode

# Мягкая остановка (завершить эпоху и сохранить чекпоинт)
trainer.request_stop(StopMode.SOFT)

# Жесткая остановка (немедленно)
trainer.request_stop(StopMode.HARD)
```

### Чекпоинты
- Автосохранение каждые N эпох
- Сохранение лучшей модели
- Возможность продолжения обучения

## 📝 Пример конфигурации (configs/default_config.yaml)

```yaml
general:
  project_name: "My AI Project"
  output_dir: "./checkpoints"
  log_dir: "./logs"
  random_seed: 42

data:
  paths: ["./data"]
  file_types: [".txt", ".csv", ".json", ".pdf", ".docx"]
  preprocessing:
    text:
      lowercase: true
      max_length: 512
    split:
      train: 0.8
      val: 0.1
      test: 0.1

model:
  type: "supervised"
  architecture: "transformer"
  params:
    hidden_size: 768
    num_layers: 6
    num_heads: 8
    dropout: 0.1

training:
  mode: "finite"
  epochs: 100
  batch_size: 32
  learning_rate: 0.0001
  optimizer: "adamw"
  checkpoint:
    save_every: 10
    save_best: true
```

## ➕ Добавление новых типов обучения

1. Создайте новый файл в `trainers/`:

```python
# trainers/custom_trainer.py
from core.trainer_base import BaseTrainer

class CustomTrainer(BaseTrainer):
    def init_optimizer(self):
        return torch.optim.Adam(self.model.parameters())
    
    def train_step(self, batch):
        # Ваш код шага обучения
        return {"loss": loss.item()}
    
    def validate(self):
        # Ваш код валидации
        return {"val_loss": val_loss}
    
    def get_data_loader(self, split="train"):
        # Возврат DataLoader
        pass
```

2. Импортируйте в `ui/app.py`:
```python
from trainers.custom_trainer import CustomTrainer
```

## ➕ Добавление новых парсеров

1. Создайте новый файл в `parsers/`:

```python
# parsers/custom_parser.py
from parsers.base_parser import BaseParser

class CustomParser(BaseParser):
    def __init__(self, config=None):
        super().__init__(config)
        self.supported_extensions = [".custom"]
    
    def can_parse(self, file_path):
        return Path(file_path).suffix.lower() in self.supported_extensions
    
    def parse(self, file_path):
        # Ваш код парсинга
        return [{"text": "parsed content", "source": file_path}]
```

2. Добавьте обработку в `core/data_manager.py`

## 🔍 Диагностика

### Проверка GPU
```bash
python -c "from utils.gpu_utils import check_cuda; print(check_cuda())"
```

### Проверка импортов
```bash
python -c "
from core.trainer_base import BaseTrainer
from core.model_manager import ModelManager
from trainers.supervised_trainer import SupervisedTrainer
print('✅ Все модули импортируются корректно')
"
```

## 📄 Лицензия

MIT License
