# Dockerfile
FROM python:3.10-slim

# Установка системных зависимостей для PDF
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

# Создаём папки (на случай, если не смонтированы)
RUN mkdir -p documents chroma_db

# Запуск
CMD ["python", "main.py"]