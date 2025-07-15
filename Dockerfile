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

# Install Python dependencies
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
RUN pip install --no-deps chatterbox-tts

# Copy handler
COPY rp_handler.py /

# Download and verify model
RUN python -c "from chatterbox.tts import ChatterboxTTS; print('Downloading model...'); model = ChatterboxTTS.from_pretrained(device='cuda'); print('Model downloaded successfully')"

# Start the container
CMD ["python3", "-u", "rp_handler.py"]


