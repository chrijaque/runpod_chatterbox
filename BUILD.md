# 🚀 RunPod Chatterbox Build Guide

## Quick Build

```bash
# Build with default tag (chatterbox:latest)
./build.sh

# Build with custom tag
./build.sh chatterbox:v1.0.0
```

## What the Build Script Does

1. **Pre-build checks** - Verifies all required files exist
2. **Docker build** - Builds the image with verbose output
3. **Success feedback** - Shows build time and image info

## Required Files

The build script checks for these files:
- ✅ `Dockerfile`
- ✅ `requirements.txt`
- ✅ `rp_handler.py`


## Build Process

1. **Base image**: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
2. **System dependencies**: FFmpeg, git, curl, etc.
3. **Forked repository**: Installs chatterbox from your GitHub fork
4. **Python dependencies**: Installs all requirements with dependencies
5. **Runtime setup**: Copies handlers and creates directories

## Troubleshooting

### Build Fails with No Error
- **Try again** - Often a temporary infrastructure issue
- **Check resources** - Ensure enough disk space and memory

### Missing Dependencies



### Repository Issues
- Verify GitHub repository is accessible
- Check branch name (should be `master`)

## Deployment

After successful build:
1. **Tag the image** for your registry
2. **Push to registry** (Docker Hub, ECR, etc.)
3. **Deploy to RunPod** using the image URL

## Example Workflow

```bash
# 1. Build the image
./build.sh chatterbox:latest

# 2. Tag for your registry
docker tag chatterbox:latest your-registry/chatterbox:latest

# 3. Push to registry
docker push your-registry/chatterbox:latest

# 4. Deploy to RunPod using the image URL
``` 