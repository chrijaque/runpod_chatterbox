FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl 

WORKDIR /
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
COPY rp_handler.py /

# Start the container
CMD ["python3", "-u", "rp_handler.py"]


