#!/bin/bash
# ====================================================================
# Automated Setup Script for IBM Cloud GPU Virtual Server (Ubuntu 22.04)
# Installs: NVIDIA Drivers, Docker, NVIDIA Container Toolkit
# ====================================================================

set -e

echo "=================================================="
echo "🔧 Setting up IBM Cloud GPU Instance for MuseTalk"
echo "=================================================="

# 1. Update and install basic tools
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y curl wget git unzip build-essential apt-transport-https ca-certificates gnupg lsb-release

# 2. Check and Install NVIDIA Driver
if ! command -v nvidia-smi &> /dev/null; then
    echo "🎮 Installing NVIDIA GPU Driver..."
    sudo apt-get install -y linux-headers-$(uname -r)
    sudo apt-get install -y nvidia-driver-535
    echo "⚠️ NVIDIA Driver installed. A reboot may be recommended if nvidia-smi fails."
else
    echo "✅ NVIDIA Driver already present:"
    nvidia-smi
fi

# 3. Install Docker Engine
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker Engine..."
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin docker-compose
    sudo usermod -aG docker $USER
    echo "✅ Docker installed successfully."
else
    echo "✅ Docker is already installed."
fi

# 4. Install NVIDIA Container Toolkit (nvidia-docker2)
echo "⚡ Installing NVIDIA Container Toolkit..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker daemon for NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 5. Verify GPU access inside Docker
echo "🔍 Verifying GPU passthrough in Docker..."
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

echo "=================================================="
echo "🎉 Setup Complete! You can now run:"
echo "   docker compose up -d --build"
echo "=================================================="
