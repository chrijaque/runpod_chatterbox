# Build cache buster - change this to force rebuild
ARG BUILD_DATE=2025-07-23
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /

# Set environment variables for logging
ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1
ENV PYTHON_UNBUFFERED="true"
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including FFmpeg
RUN apt-get update && \
    apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    libavcodec-extra \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev && \
    rm -rf /var/lib/apt/lists/*

# Set git global config to avoid warnings
RUN git config --global --add safe.directory '*'

# Setup environment for huggingface
ENV HF_HOME=/root/.cache/huggingface

# Install Python dependencies with verbose pip output
COPY requirements.txt /requirements.txt

# Debug: Show what's in requirements.txt
RUN echo "📋 Contents of requirements.txt:" && cat requirements.txt

# Install forked repository FIRST to prevent dependency conflicts
RUN echo "🔧 Installing forked repository..." && \
    echo "🔍 Testing repository access..." && \
    curl -s -o /dev/null -w "%{http_code}" https://github.com/chrijaque/chatterbox_embed && \
    echo "🔧 Installing forked repository..." && \
    pip install --no-cache-dir --force-reinstall git+https://github.com/chrijaque/chatterbox_embed.git@master#egg=chatterbox-tts && \
    echo "🔍 Verifying forked repository installation..." && \
    pip show chatterbox-tts | grep -E "(Location|Version)" && \
    python -c "import chatterbox; print('✅ chatterbox imported from:', chatterbox.__file__)"

# Debug: Check which repository was installed
RUN echo "🔍 Checking installed repository..." && \
    pip show chatterbox-tts

# Install other requirements (excluding chatterbox-tts since it's already installed)
RUN echo "🔧 Installing other requirements..." && \
    pip install -v -r requirements.txt --no-deps && \
    echo "🔍 Verifying chatterbox-tts is still from forked repo..." && \
    pip show chatterbox-tts | grep -E "(Location|Version)"

# Copy files
COPY rp_handler.py /
COPY diagnose_startup.py /

# Create required directories
RUN mkdir -p /voice_profiles /voice_samples /temp_voice /tts_generated

# Model will be downloaded at runtime when needed

# Final verification after all installations
RUN echo "🔍 Final verification after all installations..." && \
    pip show chatterbox-tts

# Start the container with diagnostic
CMD ["sh", "-c", "python3 -u diagnose_startup.py && python3 -u rp_handler.py"]


