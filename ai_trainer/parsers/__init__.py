"""
Parsers - модуль для парсинга различных форматов файлов
"""

from .base_parser import BaseParser
from .text_parser import TextParser
from .document_parser import DocumentParser
from .image_parser import ImageParser

__all__ = ["BaseParser", "TextParser", "DocumentParser", "ImageParser"]
