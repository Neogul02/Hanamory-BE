# YOLOv5 Flask API - 이전 작동 버전 기반
FROM python:3.8-slim

# 시스템 패키지 설치 (간단하게)
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 종속성 설치 (순서 수정)
RUN pip install --upgrade pip
RUN pip install flask flask-cors gunicorn
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install ultralytics pillow opencv-python-headless numpy pyyaml requests psutil

# Flask 백엔드와 모델 복사
COPY backend/app.py .
COPY backend/models/ ./models/

# 필요한 디렉토리 생성
RUN mkdir -p static/uploads static/results

# 환경변수 설정
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TORCH_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# 간단하게 Flask 앱 직접 실행
CMD ["python", "app.py"]
