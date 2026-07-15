#!/bin/bash
# AI Trainer Platform - Linux/Mac launch script

echo "============================================================"
echo "AI Trainer Platform - Запуск"
echo "============================================================"
echo ""

# Set UTF-8 encoding
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if .venv exists
if [ ! -f ".venv/bin/python" ]; then
    echo "[Bootstrap] Создание виртуального окружения..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Не удалось создать виртуальное окружение"
        exit 1
    fi
fi

# Activate virtual environment
source .venv/bin/activate

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "[GPU] Обнаружена NVIDIA видеокарта"
    echo "[Install] Установка CUDA зависимостей..."
    .venv/bin/pip install -r requirements-cuda.txt
else
    echo "[CPU] NVIDIA не обнаружена, установка CPU зависимостей..."
    .venv/bin/pip install -r requirements.txt
fi

echo ""
echo "[Launch] Запуск приложения..."
echo ""

# Run the application
.venv/bin/python main.py "$@"
