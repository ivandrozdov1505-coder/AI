@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================================
echo   AI Trainer Platform - Автозапуск через Docker
echo ============================================================
echo.

REM Проверка наличия Docker
echo [1/4] Проверка установки Docker...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ❌ ОШИБКА: Docker не найден или не запущен!
    echo.
    echo Пожалуйста, установите Docker Desktop:
    echo https://www.docker.com/products/docker-desktop/
    echo.
    echo Если Docker уже установлен, убедитесь, что он запущен.
    echo.
    pause
    exit /b 1
)
echo ✅ Docker найден: 
docker --version
echo.

REM Переход в папку со скриптом
cd /d "%~dp0"

REM Сборка образа (если еще не собран или изменился Dockerfile)
echo [2/4] Подготовка образа (это займет время только при первом запуске)...
echo.
docker-compose build
if %errorlevel% neq 0 (
    echo.
    echo ❌ Ошибка при сборке образа!
    pause
    exit /b 1
)
echo.

REM Запуск контейнера
echo [3/4] Запуск приложения...
echo.
echo ============================================================
echo 🚀 Приложение запускается!
echo 🌐 Откройте браузер по адресу: http://localhost:7860
echo 📝 Логи будут отображаться здесь. Для остановки нажмите Ctrl+C
echo ============================================================
echo.

docker-compose up

REM Если выполнение прервано
echo.
echo [4/4] Работа завершена.
echo Для повторного запуска просто откройте этот файл снова.
pause
