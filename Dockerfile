FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Set git global config to avoid warnings
RUN git config --global --add safe.directory '*'

# Setup environment for huggingface
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1

# Install Python dependencies
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
RUN pip install --no-deps chatterbox-tts

# Copy files
COPY rp_handler.py /
COPY download_model.py /

# Download and verify model with detailed error reporting
RUN python download_model.py

# Start the container
CMD ["python3", "-u", "rp_handler.py"]


