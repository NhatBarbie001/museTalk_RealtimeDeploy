#!/bin/bash
set -e

# Set the checkpoints directory
CheckpointsDir="models"

# Create necessary directories
mkdir -p models/musetalk models/musetalkV15 models/syncnet models/dwpose models/face-parse-bisent models/sd-vae models/whisper

echo "📥 [1/7] Downloading MuseTalk V1.0 weights..."
huggingface-cli download TMElyralab/MuseTalk \
  --local-dir $CheckpointsDir \
  --include "musetalk/musetalk.json" "musetalk/pytorch_model.bin"

echo "📥 [2/7] Downloading MuseTalk V1.5 weights (unet.pth)..."
huggingface-cli download TMElyralab/MuseTalk \
  --local-dir $CheckpointsDir \
  --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth"

echo "📥 [3/7] Downloading SD VAE weights..."
huggingface-cli download stabilityai/sd-vae-ft-mse \
  --local-dir $CheckpointsDir/sd-vae \
  --include "config.json" "diffusion_pytorch_model.bin"

echo "📥 [4/7] Downloading Whisper weights..."
huggingface-cli download openai/whisper-tiny \
  --local-dir $CheckpointsDir/whisper \
  --include "config.json" "pytorch_model.bin" "preprocessor_config.json"

echo "📥 [5/7] Downloading DWPose weights..."
huggingface-cli download yzd-v/DWPose \
  --local-dir $CheckpointsDir/dwpose \
  --include "dw-ll_ucoco_384.pth"

echo "📥 [6/7] Downloading SyncNet weights..."
huggingface-cli download ByteDance/LatentSync \
  --local-dir $CheckpointsDir/syncnet \
  --include "latentsync_syncnet.pt"

echo "📥 [7/7] Downloading Face Parse Bisent weights..."
if [ ! -f "$CheckpointsDir/face-parse-bisent/79999_iter.pth" ]; then
    gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O $CheckpointsDir/face-parse-bisent/79999_iter.pth || true
fi

if [ ! -f "$CheckpointsDir/face-parse-bisent/resnet18-5c106cde.pth" ]; then
    curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
      -o $CheckpointsDir/face-parse-bisent/resnet18-5c106cde.pth
fi

echo "=================================================="
echo "✅ All weights have been downloaded successfully!"
echo "=================================================="
 
