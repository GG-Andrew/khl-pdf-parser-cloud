# Slim Python base
FROM python:3.11-slim

# Install system deps for pdf & tesseract
RUN apt-get update && apt-get install -y     tesseract-ocr     tesseract-ocr-rus     poppler-utils     libglib2.0-0 libgl1-mesa-glx libglib2.0-data     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY main.py ./

EXPOSE 8080
ENV PORT=8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]