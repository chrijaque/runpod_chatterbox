# Build cache buster - change this to force rebuild
ARG BUILD_DATE=2025-07-23
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /

# Set environment variables for logging
ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1
ENV PYTHON_UNBUFFERED="true"
ENV DEBIAN_FRONTEND=noninteractive

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

# Install Python dependencies with verbose pip output
COPY requirements.txt /requirements.txt

# Debug: Show what's in requirements.txt
RUN echo "📋 Contents of requirements.txt:" && cat requirements.txt

# Step 1: Install forked repository FIRST to prevent dependency conflicts
RUN echo "🔧 STEP 1: Installing forked repository FIRST..." && \
    pip install -v git+https://github.com/chrijaque/chatterbox_embed.git#egg=chatterbox-tts

# Step 2: Verify forked repository is installed
RUN echo "🔍 STEP 2: Verifying forked repository installation..." && \
    python -c "import chatterbox; print('📦 chatterbox module path:', chatterbox.__file__)" && \
    python -c "import chatterbox; import os; repo_path = os.path.dirname(chatterbox.__file__); print('📁 chatterbox directory:', repo_path)" && \
    python -c "import chatterbox; import os; repo_path = os.path.dirname(chatterbox.__file__); git_path = os.path.join(repo_path, '.git'); print('🔍 .git exists:', os.path.exists(git_path))" && \
    python -c "from chatterbox.tts import ChatterboxTTS; model = ChatterboxTTS.from_pretrained(device='cpu'); print('✅ Model loaded successfully'); print('🔍 save_voice_profile available:', hasattr(model, 'save_voice_profile)); print('🔍 load_voice_profile available:', hasattr(model, 'load_voice_profile))"

# Step 3: Install other requirements with dependency conflict prevention
RUN echo "🔧 STEP 3: Installing other requirements..." && \
    pip install -v -r requirements.txt --no-deps

# Step 4: Final verification after all installations
RUN echo "🔍 STEP 4: Final verification after all installations..." && \
    python -c "import chatterbox; print('📦 FINAL chatterbox module path:', chatterbox.__file__)" && \
    python -c "import chatterbox; import os; repo_path = os.path.dirname(chatterbox.__file__); git_path = os.path.join(repo_path, '.git'); print('📁 FINAL chatterbox directory:', repo_path)" && \
    python -c "import chatterbox; import os; repo_path = os.path.dirname(chatterbox.__file__); git_path = os.path.join(repo_path, '.git'); print('🔍 FINAL .git exists:', os.path.exists(git_path))" && \
    python -c "from chatterbox.tts import ChatterboxTTS; model = ChatterboxTTS.from_pretrained(device='cpu'); print('✅ FINAL Model loaded successfully'); print('🔍 FINAL save_voice_profile available:', hasattr(model, 'save_voice_profile)); print('🔍 FINAL load_voice_profile available:', hasattr(model, 'load_voice_profile))" && \
    pip show chatterbox-tts

# Debug: Check which chatterbox is installed
RUN echo "🔍 Checking chatterbox installation:" && \
    python -c "import chatterbox; print(f'📦 chatterbox module path: {chatterbox.__file__}')" && \
    python -c "import chatterbox; import os; repo_path = os.path.dirname(chatterbox.__file__); print(f'📁 chatterbox directory: {repo_path}')" && \
    python -c "import chatterbox; import os; repo_path = os.path.dirname(chatterbox.__file__); git_path = os.path.join(repo_path, '.git'); print(f'🔍 .git exists: {os.path.exists(git_path)}')" && \
    python -c "import chatterbox; print(f'📋 chatterbox version: {getattr(chatterbox, \"__version__\", \"Unknown\")}')" && \
    python -c "from chatterbox.tts import ChatterboxTTS; model = ChatterboxTTS.from_pretrained(device='cpu'); print(f'✅ Model loaded successfully'); print(f'🔍 Available methods: {[m for m in dir(model) if \"voice\" in m.lower() or \"profile\" in m.lower()]}')"

# Copy files
COPY rp_handler.py /
COPY download_model.py /
COPY diagnose_chatterbox.py /

# Create required directories
RUN mkdir -p /voice_profiles /voice_samples /temp_voice /tts_generated

# Download and verify model with detailed error reporting
RUN python -u download_model.py 2>&1

# Run diagnostic script to verify installation
RUN echo "🔍 Running diagnostic script..." && python diagnose_chatterbox.py

# Start the container
CMD ["python3", "-u", "rp_handler.py"]


