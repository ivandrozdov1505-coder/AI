"""
Парсер текстовых файлов: TXT, CSV, JSON
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional
from .base_parser import BaseParser
import logging

logger = logging.getLogger(__name__)


class TextParser(BaseParser):
    """Парсер для текстовых форматов"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.supported_extensions = ['.txt', '.csv', '.json']
        self.encoding = config.get('encoding', 'utf-8') if config else 'utf-8'
    
    def can_parse(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.supported_extensions
    
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """Парсинг текстового файла"""
        if not self.validate_file(file_path):
            logger.error(f"Файл не найден: {file_path}")
            return []
        
        suffix = Path(file_path).suffix.lower()
        
        if suffix == '.txt':
            return self._parse_txt(file_path)
        elif suffix == '.csv':
            return self._parse_csv(file_path)
        elif suffix == '.json':
            return self._parse_json(file_path)
        else:
            logger.warning(f"Неподдерживаемый формат: {suffix}")
            return []
    
    def _parse_txt(self, file_path: str) -> List[Dict[str, Any]]:
        """Парсинг TXT файла"""
        data = []
        with open(file_path, 'r', encoding=self.encoding) as f:
            content = f.read()
        
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        for i, line in enumerate(lines):
            data.append({
                'text': line,
                'source': str(file_path),
                'line': i + 1,
                'type': 'text'
            })
        
        logger.debug(f"TXT: загружено {len(data)} строк")
        return data
    
    def _parse_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """Парсинг CSV файла"""
        data = []
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                # Пробуем определить разделитель
                sample = f.read(4096)
                f.seek(0)
                
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample, delimiters=',;\t|')
                
                reader = csv.DictReader(f, dialect=dialect)
                for row in reader:
                    row['source'] = str(file_path)
                    row['type'] = 'csv'
                    data.append(dict(row))
        except Exception as e:
            logger.warning(f"Ошибка парсинга CSV (пробуем простой режим): {e}")
            # Fallback к простому чтению
            with open(file_path, 'r', encoding=self.encoding) as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                for i, row in enumerate(reader):
                    if headers:
                        data.append({
                            **dict(zip(headers, row)),
                            'source': str(file_path),
                            'row': i + 1,
                            'type': 'csv'
                        })
        
        logger.debug(f"CSV: загружено {len(data)} записей")
        return data
    
    def _parse_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Парсинг JSON файла"""
        with open(file_path, 'r', encoding=self.encoding) as f:
            content = json.load(f)
        
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    item['source'] = str(file_path)
                    item['type'] = 'json'
                    data.append(item)
                else:
                    data.append({
                        'text': str(item),
                        'source': str(file_path),
                        'type': 'json'
                    })
        elif isinstance(content, dict):
            content['source'] = str(file_path)
            content['type'] = 'json'
            data.append(content)
        
        logger.debug(f"JSON: загружено {len(data)} записей")
        return data
