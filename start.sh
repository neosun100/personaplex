#!/bin/bash
# PersonaPlex One-Click Start Script
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🎙️ PersonaPlex - One-Click Start${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check Docker
if ! command -v docker &>/dev/null; then
    echo -e "${RED}❌ Docker not found. Please install Docker first.${NC}"
    exit 1
fi

# Check nvidia-docker
if ! docker info 2>/dev/null | grep -q "nvidia"; then
    if ! command -v nvidia-smi &>/dev/null; then
        echo -e "${YELLOW}⚠️  NVIDIA GPU not detected, will use CPU mode${NC}"
        export DEVICE=cpu
    fi
fi

# Auto-select GPU with lowest memory usage
if command -v nvidia-smi &>/dev/null && [ "${DEVICE}" != "cpu" ]; then
    GPU_ID=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
             sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')
    export NVIDIA_VISIBLE_DEVICES=${GPU_ID}
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i ${GPU_ID})
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader -i ${GPU_ID})
    echo -e "${GREEN}🖥️  Selected GPU ${GPU_ID}: ${GPU_NAME} (${GPU_MEM})${NC}"
fi

# Check HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    if [ -f "$HOME/.env" ] && grep -q "HUGGING_FACE_HUB_TOKEN" "$HOME/.env"; then
        export HF_TOKEN=$(grep "HUGGING_FACE_HUB_TOKEN" "$HOME/.env" | cut -d'=' -f2)
    elif [ -f "$HOME/.cache/huggingface/token" ]; then
        export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
    else
        echo -e "${RED}❌ HF_TOKEN not set. Please set it:${NC}"
        echo "   export HF_TOKEN=your_huggingface_token"
        exit 1
    fi
fi

# Check port availability - find a free port starting from 8998
PORT=${PORT:-8998}
while ss -tlnp | grep -q ":${PORT} " 2>/dev/null; do
    echo -e "${YELLOW}⚠️  Port ${PORT} is in use, trying $((PORT+1))${NC}"
    PORT=$((PORT+1))
done
export PORT

echo -e "${GREEN}📡 Using port: ${PORT}${NC}"

# Start
docker-compose up -d

echo ""
echo -e "${GREEN}✅ PersonaPlex started successfully!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "  🌐 Web UI:    http://0.0.0.0:${PORT}"
echo -e "  📄 API Docs:  http://0.0.0.0:${PORT}/docs"
echo -e "  ❤️  Health:    http://0.0.0.0:${PORT}/health"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
