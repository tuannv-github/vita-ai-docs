# VITA-1.5 Installation Guide

This guide provides detailed installation instructions for VITA-1.5, including system requirements, dependencies, and troubleshooting tips.

## 📋 System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended: RTX 3090/4090 or better)
- **RAM**: Minimum 16GB, recommended 32GB+
- **Storage**: At least 50GB free space for model weights and dependencies
- **CPU**: Multi-core processor (recommended: 8+ cores)

### Software Requirements
- **Operating System**: Linux (Ubuntu 18.04+), macOS, or Windows with WSL2
- **Python**: 3.10 (required)
- **CUDA**: 11.8 or 12.1 (depending on PyTorch version)
- **Conda**: Latest version recommended

## 🚀 Installation Steps

### Step 1: Clone the Repository

```bash
git clone https://github.com/VITA-MLLM/VITA
cd VITA
git submodule update --init --recursive
```

### Step 2: Create Conda Environment

```bash
# Create new conda environment
conda create -n vita python=3.10 -y
conda activate vita

# Verify Python version
python --version  # Should output Python 3.10.x
```

### Step 3: Install PyTorch

Choose the appropriate PyTorch installation based on your CUDA version:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU-only (not recommended for inference)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 4: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install flash-attention (requires compilation)
pip install flash-attn --no-build-isolation
```

### Step 5: Download Model Weights

Create a directory for model weights and download the required files:

```bash
# Create model directory
mkdir -p models/vita-1.5

# Download VITA-1.5 checkpoint
cd models/vita-1.5
git lfs install
git clone https://huggingface.co/VITA-MLLM/VITA-1.5

# Download vision encoder
cd ..
git clone https://huggingface.co/OpenGVLab/InternViT-300M-448px

# Download audio encoder
git clone https://huggingface.co/VITA-MLLM/VITA-1.5/audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning
```

## 🔧 Configuration

### Environment Variables

Set up environment variables for optimal performance:

```bash
# Add to your ~/.bashrc or ~/.zshrc
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0  # Adjust based on your GPU setup
```

### Model Configuration

Update the model paths in your configuration:

```python
# In your script or config file
MODEL_CONFIG = {
    "model_path": "/path/to/models/vita-1.5",
    "vision_tower": "/path/to/models/InternViT-300M-448px",
    "audio_encoder": "/path/to/models/audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning"
}
```

## 🧪 Verification

Test your installation with a simple script:

```python
# test_installation.py
import torch
from vita.model import VITA

def test_installation():
    print("Testing VITA-1.5 installation...")
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    
    # Test model loading (optional - requires model weights)
    try:
        # model = VITA.from_pretrained("/path/to/vita-1.5")
        print("Model loading test passed!")
    except Exception as e:
        print(f"Model loading test failed: {e}")
    
    print("Installation test completed!")

if __name__ == "__main__":
    test_installation()
```

Run the test:

```bash
python test_installation.py
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Flash Attention Installation Fails

**Error**: `flash-attn` installation fails with compilation errors

**Solution**:
```bash
# Install build dependencies
sudo apt-get update
sudo apt-get install build-essential

# Try alternative installation
pip install flash-attn --no-build-isolation --no-cache-dir
```

#### 2. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce batch size in your scripts
- Use gradient checkpointing
- Set `max_dynamic_patch` to 1 in config.json for real-time inference

#### 3. Model Loading Issues

**Error**: Model weights not found or corrupted

**Solution**:
```bash
# Re-download with git lfs
git lfs pull
# Or download manually from Hugging Face
```

#### 4. Import Errors

**Error**: `ModuleNotFoundError` for vita modules

**Solution**:
```bash
# Ensure PYTHONPATH is set correctly
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Reinstall in development mode
pip install -e .
```

### Performance Optimization

#### GPU Memory Optimization

```python
# In your inference script
import torch

# Enable memory efficient attention
torch.backends.cuda.enable_flash_sdp(True)

# Set memory fraction
torch.cuda.set_per_process_memory_fraction(0.9)
```

#### CPU Optimization

```bash
# Set number of threads
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

## 📦 Alternative Installation Methods

### Docker Installation

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt
RUN pip install flash-attn --no-build-isolation

CMD ["python", "video_audio_demo.py"]
```

### Conda Environment Export

```bash
# Export environment
conda env export > environment.yml

# Recreate environment
conda env create -f environment.yml
```

## 🔄 Updates

To update VITA-1.5:

```bash
# Update repository
git pull origin main
git submodule update --init --recursive

# Update dependencies
pip install -r requirements.txt --upgrade

# Update model weights (if new versions available)
cd models/vita-1.5
git pull origin main
```

## 📞 Support

If you encounter issues not covered in this guide:

1. Check the [GitHub Issues](https://github.com/VITA-MLLM/VITA/issues)
2. Join the [WeChat Group](./asset/wechat-group.jpg)
3. Review the [VLMEvalKit documentation](https://github.com/open-compass/VLMEvalKit)

---

**Note**: Installation times may vary based on your system specifications. GPU installation typically takes 10-30 minutes, while CPU-only installation may take longer.
