#!/bin/bash

# Simple build script for RunPod Chatterbox
# Usage: ./build.sh [tag]

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default tag
TAG=${1:-"chatterbox:latest"}

echo -e "${BLUE}🚀 ===== RUNPOD CHATTERBOX BUILD SCRIPT =====${NC}"
echo -e "${YELLOW}📦 Building with tag: ${TAG}${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "Dockerfile.vc" ]; then
    echo -e "${RED}❌ Error: Dockerfile.vc not found in current directory${NC}"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ Error: requirements.txt not found${NC}"
    exit 1
fi

echo -e "${BLUE}🔍 Pre-build checks...${NC}"

# Check if key files exist
FILES=("vc_handler.py")
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ Found: $file${NC}"
    else
        echo -e "${RED}❌ Missing: $file${NC}"
        exit 1
    fi
done

echo ""
echo -e "${BLUE}🔧 Starting Docker build...${NC}"
echo -e "${YELLOW}📋 Build command: docker build -t $TAG .${NC}"
echo ""

# Start the build
start_time=$(date +%s)

if docker build -t "$TAG" .; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo ""
    echo -e "${GREEN}✅ ===== BUILD SUCCESSFUL =====${NC}"
    echo -e "${GREEN}📦 Image built successfully: $TAG${NC}"
    echo -e "${GREEN}⏱️  Build time: ${duration} seconds${NC}"
    echo ""
    
    # Show image info
    echo -e "${BLUE}📊 Image information:${NC}"
    docker images "$TAG" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    
    echo ""
    echo -e "${GREEN}🎉 Ready to deploy to RunPod!${NC}"
    
else
    echo ""
    echo -e "${RED}❌ ===== BUILD FAILED =====${NC}"
    echo -e "${RED}🔧 Check the error messages above for details${NC}"
    exit 1
fi 