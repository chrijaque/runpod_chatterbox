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

# Copy handler
COPY rp_handler.py /

# Download and verify model with detailed error reporting
RUN python -c "import sys; \
    import traceback; \
    try: \
        print('Python version:', sys.version); \
        print('Importing ChatterboxTTS...'); \
        from chatterbox.tts import ChatterboxTTS; \
        print('Import successful. Downloading model...'); \
        model = ChatterboxTTS.from_pretrained(device='cuda'); \
        print('Model downloaded and loaded successfully') \
    except Exception as e: \
        print('Error occurred:'); \
        print(str(e)); \
        traceback.print_exc(); \
        sys.exit(1)"

# Start the container
CMD ["python3", "-u", "rp_handler.py"]


