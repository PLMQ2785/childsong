FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1 \
    YUE_PATH=/opt/YuE/inference \
    HF_HOME=/runpod-volume/.cache/huggingface \
    TRANSFORMERS_CACHE=/runpod-volume/.cache/huggingface \
    JOBS_DIR=/tmp/yue_jobs \
    SUNO_API_BASE=https://api.sunoapi.org/api/v1 \
    SUNO_API_TOKEN=

WORKDIR /opt

RUN apt-get update && apt-get install -y git git-lfs ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install
RUN pip install --upgrade pip setuptools wheel
RUN pip install torchcodec hf_transfer
RUN pip install flash-attn --no-build-isolation

# YuE를 /opt 아래로
RUN git clone https://github.com/multimodal-art-projection/YuE.git /opt/YuE
RUN git clone https://huggingface.co/m-a-p/xcodec_mini_infer /opt/YuE/inference/xcodec_mini_infer
RUN cd /opt/YuE && git lfs pull || true
RUN if [ -f /opt/YuE/requirements.txt ]; then pip install -r /opt/YuE/requirements.txt; fi

# API deps + demucs + aiohttp for Suno
RUN pip install fastapi "uvicorn[standard]" pydantic python-multipart aiofiles demucs aiohttp

# 앱
WORKDIR /opt/app
COPY providers/ /opt/app/providers/
COPY api_server.py /opt/app/api_server.py

EXPOSE 8000
CMD ["python", "-u", "/opt/app/api_server.py"]

