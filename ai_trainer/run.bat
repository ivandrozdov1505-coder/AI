@echo off
chcp 65001 >nul
title AI Trainer Platform

echo ============================================================
echo AI Trainer Platform - Запуск
echo ============================================================
echo.

REM Set UTF-8 encoding
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

REM Check if .venv exists
if not exist ".venv\Scripts\python.exe" (
    echo [Bootstrap] Создание виртуального окружения...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Не удалось создать виртуальное окружение
        pause
        exit /b 1
    )
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Check for NVIDIA GPU
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [GPU] Обнаружена NVIDIA видеокарта
    echo [Install] Установка CUDA зависимостей...
    .venv\Scripts\pip install -r requirements-cuda.txt
) else (
    echo [CPU] NVIDIA не обнаружена, установка CPU зависимостей...
    .venv\Scripts\pip install -r requirements.txt
)

echo.
echo [Launch] Запуск приложения...
echo.

REM Run the application
.venv\Scripts\python main.py %*

pause
