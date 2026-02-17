FROM python:3.12-slim

# System deps:
#   libgl1 + libglib2.0-0  → MediaPipe / OpenCV (face detection)
#   libcairo2               → CairoSVG (SVG→PNG logo uploads)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway injects $PORT at runtime
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
