#!/bin/bash
set -e

echo "=================================================="
echo "🚀 Starting MuseTalk Live Avatar Realtime Server"
echo "=================================================="

# Check if model weights exist; if not, download them
if [ ! -f "models/musetalkV15/unet.pth" ]; then
    echo "📦 Weights not found in ./models. Initiating automatic download..."
    python /workspace/download_weights.py
else
    echo "✅ Model weights detected in ./models."
fi

# Ensure avatar cache directories exist
mkdir -p results/v15/avatars data/avatars_upload

echo "🔥 Launching FastAPI + WebRTC Server on 0.0.0.0:8000..."
exec python -m server.main --host 0.0.0.0 --port 8000
