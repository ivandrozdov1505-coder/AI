"""
Парсер изображений с поддержкой OCR
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from .base_parser import BaseParser
import logging

logger = logging.getLogger(__name__)


class ImageParser(BaseParser):
    """Парсер для изображений с OCR"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.supported_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']
        self.ocr_enabled = config.get('ocr', {}).get('enabled', False) if config else False
        self.ocr_engine = config.get('ocr', {}).get('engine', 'tesseract') if config else 'tesseract'
        self.ocr_languages = config.get('ocr', {}).get('languages', ['eng']) if config else ['eng']
    
    def can_parse(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.supported_extensions
    
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """Парсинг изображения"""
        if not self.validate_file(file_path):
            logger.error(f"Файл не найден: {file_path}")
            return []
        
        data = [{
            'image_path': str(file_path),
            'source': str(file_path),
            'type': 'image'
        }]
        
        # OCR если включен
        if self.ocr_enabled:
            ocr_text = self._perform_ocr(file_path)
            if ocr_text:
                data[0]['text'] = ocr_text
                data[0]['type'] = 'image_ocr'
        
        return data
    
    def _perform_ocr(self, file_path: str) -> Optional[str]:
        """Выполнение OCR на изображении"""
        try:
            if self.ocr_engine == 'tesseract':
                return self._ocr_tesseract(file_path)
            elif self.ocr_engine == 'paddle':
                return self._ocr_paddle(file_path)
            else:
                logger.warning(f"Неизвестный OCR движок: {self.ocr_engine}")
                return None
        except Exception as e:
            logger.error(f"Ошибка OCR: {e}")
            return None
    
    def _ocr_tesseract(self, file_path: str) -> Optional[str]:
        """OCR с использованием Tesseract"""
        try:
            import pytesseract
            from PIL import Image
            
            img = Image.open(file_path)
            lang = '+'.join(self.ocr_languages)
            
            text = pytesseract.image_to_string(img, lang=lang)
            logger.debug(f"Tesseract OCR: извлечено {len(text)} символов")
            return text.strip() if text else None
            
        except ImportError:
            logger.error("pytesseract не установлен. Установите: pip install pytesseract")
            logger.error("Также требуется установить Tesseract OCR системно")
            return None
        except Exception as e:
            logger.error(f"Tesseract ошибка: {e}")
            return None
    
    def _ocr_paddle(self, file_path: str) -> Optional[str]:
        """OCR с использованием PaddleOCR"""
        try:
            from paddleocr import PaddleOCR
            
            ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.ocr_languages[0] if self.ocr_languages else 'en'
            )
            
            result = ocr.ocr(file_path, cls=True)
            
            texts = []
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        texts.append(line[1][0])
            
            text = '\n'.join(texts)
            logger.debug(f"PaddleOCR: извлечено {len(texts)} строк")
            return text.strip() if text else None
            
        except ImportError:
            logger.error("PaddleOCR не установлен. Установите: pip install paddleocr")
            return None
        except Exception as e:
            logger.error(f"PaddleOCR ошибка: {e}")
            return None
