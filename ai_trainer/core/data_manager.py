"""
Менеджер данных - загрузка, предобработка, аугментация
Поддержка множественных форматов: текст, документы, изображения
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import logging
import json
import csv
from collections import defaultdict

logger = logging.getLogger(__name__)


class MultiModalDataset(Dataset):
    """Универсальный датасет для разнородных данных"""
    
    def __init__(self, data: List[Dict[str, Any]], transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        if self.transform:
            item = self.transform(item)
        return item


class DataManager:
    """
    Менеджер для загрузки и предобработки данных
    Поддерживает: TXT, CSV, JSON, PDF, DOCX, изображения
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация менеджера данных
        
        Args:
            config: Конфигурация данных
        """
        self.config = config
        self.raw_data: List[Dict[str, Any]] = []
        self.processed_data: Dict[str, List[Dict[str, Any]]] = {}
        self.vocab: Optional[Dict[str, int]] = None
        self.label_encoder: Dict[str, int] = {}
    
    def load_files(self, paths: Union[str, List[str]], file_types: List[str] = None) -> int:
        """
        Загрузка файлов из указанных путей
        
        Args:
            paths: Пути к файлам или директориям
            file_types: Фильтр по расширениям
            
        Returns:
            Количество загруженных файлов
        """
        if isinstance(paths, str):
            paths = [paths]
        
        file_types = file_types or self.config.get('file_types', [])
        loaded_count = 0
        
        for path in paths:
            path = Path(path)
            
            if path.is_file():
                # Загрузка отдельного файла
                if not file_types or path.suffix.lower() in file_types:
                    if self._load_file(path):
                        loaded_count += 1
            elif path.is_dir():
                # Рекурсивный обход директории
                for ext in (file_types or ['*']):
                    pattern = f"**/*{ext}" if ext.startswith('.') else f"**/{ext}"
                    for file_path in path.glob(pattern):
                        if self._load_file(file_path):
                            loaded_count += 1
        
        logger.info(f"Загружено {loaded_count} файлов")
        return loaded_count
    
    def _load_file(self, path: Path) -> bool:
        """Загрузка одного файла"""
        try:
            suffix = path.suffix.lower()
            
            if suffix == '.txt':
                data = self._load_txt(path)
            elif suffix == '.csv':
                data = self._load_csv(path)
            elif suffix == '.json':
                data = self._load_json(path)
            elif suffix == '.pdf':
                data = self._load_pdf(path)
            elif suffix == '.docx':
                data = self._load_docx(path)
            elif suffix in ['.jpg', '.jpeg', '.png', '.webp']:
                data = self._load_image(path)
            else:
                logger.warning(f"Неподдерживаемый формат: {path}")
                return False
            
            self.raw_data.extend(data)
            logger.debug(f"Загружен файл: {path} ({len(data)} записей)")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки {path}: {e}")
            return False
    
    def _load_txt(self, path: Path) -> List[Dict[str, Any]]:
        """Загрузка текстового файла"""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Разбиение на строки/абзацы
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        return [{'text': line, 'source': str(path), 'type': 'text'} for line in lines]
    
    def _load_csv(self, path: Path) -> List[Dict[str, Any]]:
        """Загрузка CSV файла"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['source'] = str(path)
                row['type'] = 'csv'
                data.append(row)
        return data
    
    def _load_json(self, path: Path) -> List[Dict[str, Any]]:
        """Загрузка JSON файла"""
        with open(path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        
        if isinstance(content, list):
            for item in content:
                item['source'] = str(path)
                item['type'] = 'json'
            return content
        else:
            content['source'] = str(path)
            content['type'] = 'json'
            return [content]
    
    def _load_pdf(self, path: Path) -> List[Dict[str, Any]]:
        """Загрузка PDF файла с использованием PyMuPDF"""
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(path)
            data = []
            
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    data.append({
                        'text': text,
                        'source': str(path),
                        'page': page_num + 1,
                        'type': 'pdf'
                    })
            
            doc.close()
            return data
        except ImportError:
            logger.warning("PyMuPDF не установлен. Установите: pip install PyMuPDF")
            return []
    
    def _load_docx(self, path: Path) -> List[Dict[str, Any]]:
        """Загрузка DOCX файла"""
        try:
            from docx import Document
            
            doc = Document(path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            
            return [{
                'text': para,
                'source': str(path),
                'type': 'docx'
            } for para in paragraphs]
        except ImportError:
            logger.warning("python-docx не установлен. Установите: pip install python-docx")
            return []
    
    def _load_image(self, path: Path) -> List[Dict[str, Any]]:
        """Загрузка изображения"""
        return [{
            'image_path': str(path),
            'source': str(path),
            'type': 'image'
        }]
    
    def preprocess(self, text_config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Предобработка данных
        
        Args:
            text_config: Настройки предобработки текста
            
        Returns:
            Обработанные данные
        """
        text_config = text_config or self.config.get('preprocessing', {}).get('text', {})
        
        processed = []
        for item in self.raw_data:
            if 'text' in item:
                text = item['text']
                
                # Lowercase
                if text_config.get('lowercase', True):
                    text = text.lower()
                
                # Удаление специальных символов
                if text_config.get('remove_special_chars', True):
                    import re
                    text = re.sub(r'[^\w\s.,!?]', '', text)
                
                # Ограничение длины
                max_length = text_config.get('max_length', 512)
                if len(text) > max_length:
                    text = text[:max_length]
                
                item['text'] = text
            
            processed.append(item)
        
        # Удаление дубликатов
        seen = set()
        unique_processed = []
        for item in processed:
            key = item.get('text', str(item))
            if key not in seen:
                seen.add(key)
                unique_processed.append(item)
        
        logger.info(f"Предобработка: {len(processed)} -> {len(unique_processed)} записей")
        self.processed_data['all'] = unique_processed
        return unique_processed
    
    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.1, 
             test_ratio: float = 0.1, stratify_key: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Разбиение на train/val/test
        
        Args:
            train_ratio: Доля тренировочной выборки
            val_ratio: Доля валидационной выборки
            test_ratio: Доля тестовой выборки
            stratify_key: Ключ для стратифицированного разбиения
            
        Returns:
            Dict со сплитами
        """
        data = self.processed_data.get('all', self.raw_data)
        
        if not data:
            raise ValueError("Нет данных для разбиения")
        
        import random
        random.seed(self.config.get('random_seed', 42))
        
        # Стратификация
        if stratify_key:
            groups = defaultdict(list)
            for item in data:
                label = item.get(stratify_key, 'default')
                groups[label].append(item)
            
            train, val, test = [], [], []
            for group_items in groups.values():
                random.shuffle(group_items)
                n = len(group_items)
                n_train = max(1, int(n * train_ratio))
                n_val = max(1, int(n * val_ratio))
                
                train.extend(group_items[:n_train])
                val.extend(group_items[n_train:n_train+n_val])
                test.extend(group_items[n_train+n_val:])
        else:
            # Случайное перемешивание
            data_copy = data.copy()
            random.shuffle(data_copy)
            
            n = len(data_copy)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            
            train = data_copy[:n_train]
            val = data_copy[n_train:n_train+n_val]
            test = data_copy[n_train+n_val:]
        
        self.processed_data['train'] = train
        self.processed_data['val'] = val
        self.processed_data['test'] = test
        
        logger.info(f"Разбиение: train={len(train)}, val={len(val)}, test={len(test)}")
        return self.processed_data
    
    def build_vocab(self, texts: List[str] = None, min_freq: int = 1, 
                   max_size: int = 32000) -> Dict[str, int]:
        """
        Построение словаря
        
        Args:
            texts: Тексты для построения словаря
            min_freq: Минимальная частота токена
            max_size: Максимальный размер словаря
            
        Returns:
            Словарь token -> id
        """
        if texts is None:
            texts = [item.get('text', '') for item in self.processed_data.get('train', []) 
                    if 'text' in item]
        
        from collections import Counter
        counter = Counter()
        
        for text in texts:
            tokens = text.split()
            counter.update(tokens)
        
        # Специальные токены
        self.vocab = {'<pad>': 0, '<unk>': 1, '<cls>': 2, '<sep>': 3}
        
        # Добавляем частые токены
        for token, count in counter.most_common(max_size - len(self.vocab)):
            if count >= min_freq:
                self.vocab[token] = len(self.vocab)
        
        logger.info(f"Словарь построен: {len(self.vocab)} токенов")
        return self.vocab
    
    def tokenize(self, text: str, max_length: int = 512) -> List[int]:
        """Токенизация текста"""
        if self.vocab is None:
            self.build_vocab()
        
        tokens = text.split()[:max_length]
        ids = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        
        # Padding
        if len(ids) < max_length:
            ids.extend([self.vocab['<pad>']] * (max_length - len(ids)))
        
        return ids
    
    def create_dataloader(self, split: str = 'train', batch_size: int = 32,
                         shuffle: bool = True, collate_fn=None) -> DataLoader:
        """
        Создание DataLoader
        
        Args:
            split: Название сплита (train/val/test)
            batch_size: Размер батча
            shuffle: Перемешивание
            collate_fn: Функция коллации батчей
            
        Returns:
            DataLoader
        """
        data = self.processed_data.get(split, [])
        
        if not data:
            raise ValueError(f"Нет данных для сплита: {split}")
        
        dataset = MultiModalDataset(data)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if split == 'train' else False,
            collate_fn=collate_fn,
            num_workers=0,  # Можно увеличить для ускорения
            pin_memory=True  # Для ускорения передачи на GPU
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики по данным"""
        stats = {
            'total_raw': len(self.raw_data),
            'total_processed': sum(len(v) for v in self.processed_data.values()),
            'splits': {k: len(v) for k, v in self.processed_data.items()},
            'vocab_size': len(self.vocab) if self.vocab else 0,
            'sources': defaultdict(int)
        }
        
        for item in self.raw_data:
            source_type = item.get('type', 'unknown')
            stats['sources'][source_type] += 1
        
        stats['sources'] = dict(stats['sources'])
        return stats
