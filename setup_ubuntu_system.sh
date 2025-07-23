#!/bin/bash

# Ubuntu 22.04 DeepStream 7.1 + YOLO11/12 Setup Script
# This script sets up all dependencies for GPU RTSP processing with DeepStream

set -e  # Exit on any error

echo "=== Ubuntu 22.04 DeepStream 7.1 + YOLO11/12 Setup ==="
echo "Starting comprehensive system setup..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# Update system packages
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential system dependencies
print_status "Installing essential system dependencies..."
sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    wget \
    curl \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    python3-pip \
    python3-dev \
    python3-venv \
    python3-setuptools \
    python3-wheel \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libcanberra-gtk3-module \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    python3-pyqt5

# Add NVIDIA repository
print_status "Adding NVIDIA repository..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install CUDA toolkit (if not already installed)
if ! command -v nvcc &> /dev/null; then
    print_status "Installing CUDA Toolkit..."
    sudo apt install -y cuda-toolkit-12-9
else
    print_success "CUDA Toolkit already installed"
fi

# Set up CUDA environment variables
print_status "Setting up CUDA environment variables..."
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc

# Source the updated bashrc
source ~/.bashrc

# Verify CUDA installation
print_status "Verifying CUDA installation..."
nvcc --version
nvidia-smi

# Install DeepStream 7.1
print_status "Installing DeepStream 7.1..."
if [ -f "deepstream-7.1_7.1.0-1_amd64.deb" ]; then
    sudo apt install -y ./deepstream-7.1_7.1.0-1_amd64.deb
    print_success "DeepStream 7.1 installed from local .deb file"
else
    print_warning "DeepStream .deb file not found, attempting to download..."
    wget https://developer.nvidia.com/deepstream-getting-started
    # Note: You'll need to manually download from NVIDIA developer portal
    print_error "Please download DeepStream 7.1 from NVIDIA developer portal"
    exit 1
fi

# Extract DeepStream SDK if needed
if [ -f "deepstream_sdk_v7.1.0_x86_64.tbz2" ]; then
    print_status "Extracting DeepStream SDK..."
    tar -xvf deepstream_sdk_v7.1.0_x86_64.tbz2
    sudo cp -r deepstream_sdk_v7.1.0_x86_64/sources/deepstream_python_apps/bindings /opt/nvidia/deepstream/deepstream-7.1/
    print_success "DeepStream SDK extracted and Python bindings copied"
fi

# Set up DeepStream environment
print_status "Setting up DeepStream environment variables..."
echo 'export DEEPSTREAM_DIR=/opt/nvidia/deepstream/deepstream-7.1' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$DEEPSTREAM_DIR/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export GST_PLUGIN_PATH=$DEEPSTREAM_DIR/lib/gst-plugins' >> ~/.bashrc
echo 'export GST_PLUGIN_SCANNER=$DEEPSTREAM_DIR/lib/gst-plugins' >> ~/.bashrc

# Install TensorRT (if not already installed)
print_status "Checking TensorRT installation..."
if ! dpkg -l | grep -q tensorrt; then
    print_warning "TensorRT not found. Please install TensorRT from NVIDIA developer portal"
    print_warning "Download from: https://developer.nvidia.com/tensorrt"
    print_warning "After downloading, run: sudo dpkg -i TensorRT-*.deb"
else
    print_success "TensorRT already installed"
fi

# Install Python dependencies system-wide
print_status "Installing Python dependencies system-wide..."

# Upgrade pip
print_status "Upgrading pip..."
pip3 install --upgrade pip

# Install CUDA-specific packages first
pip3 install nvidia-pyindex
pip3 install nvidia-vpf

# Install core dependencies
pip3 install numpy==1.24.3
pip3 install opencv-python==4.7.0.72
pip3 install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install CUDA-specific packages
pip3 install pycuda==2022.2.2
pip3 install cupy-cuda11x==12.0.0

# Install PyNvVideoCodec (if available)
pip3 install PyNvVideoCodec==1.0.2

# Install ML models
pip3 install ultralytics==8.0.124
pip3 install supervision==0.11.1

# Install code quality & testing
pip3 install black==23.3.0
pip3 install ruff==0.0.262
pip3 install isort==5.12.0
pip3 install pytest==7.3.1

# Install web interface
pip3 install websockets==11.0.3
pip3 install aiohttp==3.8.4

# Install utilities
pip3 install tqdm==4.65.0
pip3 install pillow==9.5.0
pip3 install requests==2.30.0
pip3 install python-dotenv==1.0.0
pip3 install pyyaml==6.0.1
pip3 install prometheus-client==0.19.0

# Install FFmpeg GPU Decoding
pip3 install av>=10.0.0
pip3 install psutil>=5.9.0

# Install 3D Coordinate Transformation
pip3 install trimesh>=3.9.0

# Install additional dependencies for DeepStream
pip3 install pyds
pip3 install nvidia-dali-cuda110

# Verify installations
print_status "Verifying installations..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python3 -c "import numpy as np; print(f'NumPy: {np.__version__}')"

# Test DeepStream installation
print_status "Testing DeepStream installation..."
if command -v deepstream-app &> /dev/null; then
    print_success "DeepStream app found"
    deepstream-app --version
else
    print_warning "DeepStream app not found in PATH"
fi

# Test GStreamer NVIDIA plugins
print_status "Testing GStreamer NVIDIA plugins..."
gst-inspect-1.0 | grep -i nvidia || print_warning "No NVIDIA GStreamer plugins found"

# Create activation script
print_status "Creating activation script..."
cat > activate_deepstream.sh << 'EOF'
#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
export DEEPSTREAM_DIR=/opt/nvidia/deepstream/deepstream-7.1
export LD_LIBRARY_PATH=$DEEPSTREAM_DIR/lib:$LD_LIBRARY_PATH
export GST_PLUGIN_PATH=$DEEPSTREAM_DIR/lib/gst-plugins
export GST_PLUGIN_SCANNER=$DEEPSTREAM_DIR/lib/gst-plugins
echo "DeepStream environment activated!"
echo "CUDA: $(nvcc --version | head -1)"
echo "DeepStream: $(deepstream-app --version 2>/dev/null || echo 'Not found')"
echo "Python: $(python3 --version)"
EOF

chmod +x activate_deepstream.sh

print_success "=== Setup Complete ==="
print_status "To activate the environment, run:"
echo "source activate_deepstream.sh"
print_status "Next steps:"
echo "1. Install TensorRT if not already installed"
echo "2. Download YOLO11/12 models"
echo "3. Build custom parser for YOLO11/12"
echo "4. Test DeepStream pipeline"
print_status "Note: All packages installed system-wide (no virtual environment)"

print_success "System setup completed successfully!" 