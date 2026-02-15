#!/bin/bash
# PersonaPlex One-Click Start Script

set -e

echo "🎙️ PersonaPlex Startup Script"
echo "=============================="

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Check nvidia-docker
if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    echo "⚠️  nvidia-docker runtime not detected. GPU support may not work."
fi

# Auto-select GPU with least memory usage
echo "🔍 Detecting GPUs..."
if command -v nvidia-smi &> /dev/null; then
    GPU_ID=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | \
             sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')
    
    if [ -n "$GPU_ID" ]; then
        echo "✅ Selected GPU $GPU_ID (lowest memory usage)"
        export NVIDIA_VISIBLE_DEVICES=$GPU_ID
    else
        echo "⚠️  Could not detect GPU, using default"
        export NVIDIA_VISIBLE_DEVICES=0
    fi
else
    echo "⚠️  nvidia-smi not found, using default GPU"
    export NVIDIA_VISIBLE_DEVICES=0
fi

# Check for .env file
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo "📝 Creating .env from .env.example..."
        cp .env.example .env
        echo "⚠️  Please edit .env and set your HF_TOKEN"
    fi
fi

# Load environment
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Check HF_TOKEN
if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" = "your_huggingface_token_here" ]; then
    echo "❌ HF_TOKEN not set. Please set it in .env file or environment."
    echo "   Get your token from: https://huggingface.co/settings/tokens"
    echo "   Accept model license: https://huggingface.co/nvidia/personaplex-7b-v1"
    exit 1
fi

# Find available port
PORT=${PORT:-8998}
while ss -tlnp 2>/dev/null | grep -q ":$PORT "; do
    echo "⚠️  Port $PORT is in use, trying next..."
    PORT=$((PORT + 1))
done
export PORT
echo "📡 Using port: $PORT"

# Start with docker-compose
echo ""
echo "🚀 Starting PersonaPlex..."
docker-compose up -d --build

# Wait for health check
echo "⏳ Waiting for service to be ready..."
for i in {1..60}; do
    if curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then
        echo ""
        echo "✅ PersonaPlex is ready!"
        echo ""
        echo "=============================="
        echo "🌐 Web UI:    http://0.0.0.0:$PORT"
        echo "📄 API Docs:  http://0.0.0.0:$PORT/docs"
        echo "❤️  Health:    http://0.0.0.0:$PORT/health"
        echo "🖥️  GPU:       $NVIDIA_VISIBLE_DEVICES"
        echo "=============================="
        exit 0
    fi
    sleep 2
    echo -n "."
done

echo ""
echo "⚠️  Service may still be starting. Check logs with:"
echo "   docker-compose logs -f"
