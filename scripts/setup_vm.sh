#!/bin/bash

# GPU VM Setup Script for proj-grounded-telescopes

set -e

echo "=== Setting up proj-grounded-telescopes on GPU VM ==="

# Check if we're running as root
if [ "$EUID" -eq 0 ]; then
    echo "Running as root - consider using a virtual environment"
fi

# Update system
echo "Updating system packages..."
apt-get update
apt-get install -y git curl wget build-essential

# Check NVIDIA GPU
echo "Checking NVIDIA GPU..."
nvidia-smi || echo "Warning: nvidia-smi not found"

# Check Python version
echo "Python version:"
python3 --version

# Install pip if not present
if ! command -v pip3 &> /dev/null; then
    echo "Installing pip..."
    apt-get install -y python3-pip
fi

# Clone repo if not present
if [ ! -d "proj-grounded-telescopes" ]; then
    echo "Cloning repository..."
    git clone https://github.com/sigamani/proj-grounded-telescopes.git
fi

cd proj-grounded-telescopes

# Install dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Test GPU setup
echo "Testing GPU setup..."
python3 scripts/test_gpu.py

echo "=== Setup complete! ==="
echo "Run 'docker build -t grounded-telescopes .' to build the container"