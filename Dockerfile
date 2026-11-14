# TactileCheck 백엔드: FastAPI + YOLOv8-seg 두 모델(블록·장면) + 규칙 판정.
# Hugging Face Spaces(Docker)·EC2·Cloud Run 어디서나 같은 이미지로 돌아간다.
#
#   docker build -t tactilecheck-api .
#   docker run -p 7860:7860 --env-file .env tactilecheck-api      # → http://localhost:7860/docs
#
# 다른 가중치로 빌드:  docker build --build-arg BLOCK_WEIGHTS_FILE=v6_best.pt -t tactilecheck-api .
# 비밀 값(SUPABASE_URL·SUPABASE_KEY·ADMIN_TOKEN·CORS_ORIGINS)은 이미지에 넣지 않고 실행 환경(Spaces Secrets 등)에서 준다.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=2

# OpenCV가 쓰는 시스템 라이브러리 (ultralytics가 GUI판 opencv도 함께 설치함)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU 전용 PyTorch를 먼저 설치 (기본 설치는 GPU용이라 수 GB 더 큼)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements-ml.txt

# Hugging Face Spaces는 uid 1000 사용자로 실행한다
RUN useradd -m -u 1000 user

COPY --chown=user app ./app
COPY --chown=user pipeline ./pipeline

ARG BLOCK_WEIGHTS_FILE=v4_best.pt
ARG SCENE_WEIGHTS_FILE=scene_v1_best.pt
COPY --chown=user weights/${BLOCK_WEIGHTS_FILE} weights/${SCENE_WEIGHTS_FILE} ./weights/

ENV BLOCK_WEIGHTS=weights/${BLOCK_WEIGHTS_FILE} \
    SCENE_WEIGHTS=weights/${SCENE_WEIGHTS_FILE} \
    PORT=7860

USER user
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
