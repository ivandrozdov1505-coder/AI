# Инструкция по запуску AI Trainer в Docker

## Быстрый старт (CPU версия)

### 1. Установите Docker Desktop
Если ещё не установлен: https://www.docker.com/products/docker-desktop/

### 2. Откройте терминал в папке проекта
```powershell
cd C:\Users\gribik\Downloads\ai_trainer\ai_trainer
```

### 3. Соберите образ (выполняется один раз, ~5-10 минут)
```bash
docker-compose build
```

### 4. Запустите контейнер
```bash
docker-compose up
```
Или в фоновом режиме:
```bash
docker-compose up -d
```

### 5. Откройте веб-интерфейс
Перейдите в браузере: **http://localhost:7860**

---

## Полезные команды

| Команда | Описание |
|---------|----------|
| `docker-compose up` | Запуск контейнера |
| `docker-compose up -d` | Запуск в фоне |
| `docker-compose logs -f` | Просмотр логов в реальном времени |
| `docker-compose down` | Остановка и удаление контейнера |
| `docker-compose ps` | Статус контейнеров |
| `docker-compose build --no-cache` | Пересборка без кэша |

---

## GPU поддержка (NVIDIA)

Если у вас есть видеокарта NVIDIA и вы хотите использовать CUDA:

### Шаг 1: Установите NVIDIA Container Toolkit
Следуйте инструкции: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

### Шаг 2: Измените Dockerfile
Замените первую строку в `Dockerfile`:
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
```

### Шаг 3: Раскомментируйте GPU секцию в docker-compose.yml
Удалите символы `#` перед секцией `deploy`:
```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### Шаг 4: Пересоберите и запустите
```bash
docker-compose build
docker-compose up -d
```

---

## Структура томов

Контейнер сохраняет данные на вашем компьютере:

- `./logs` → логи приложения
- `./configs` → конфигурационные файлы
- `./models` → обученные модели

Эти данные сохранятся даже после удаления контейнера.

---

## Преимущества Docker

✅ Все зависимости уже установлены внутри контейнера  
✅ Не нужно устанавливать Python, PyTorch, Gradio на хост-машину  
✅ Изоляция от системы  
✅ Легко переносить на другие компьютеры  
✅ Быстрый запуск после первой сборки  

---

## Решение проблем

### Ошибка "port already in use"
Измените порт в `docker-compose.yml`:
```yaml
ports:
  - "7861:7860"  # Используйте другой свободный порт
```

### Контейнер не запускается
Проверьте логи:
```bash
docker-compose logs
```

### Нужно обновить зависимости
После изменения `requirements.txt`:
```bash
docker-compose build --no-cache
docker-compose up -d
```
