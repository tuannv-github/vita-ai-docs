# VITA-1.5 Usage Guide

This comprehensive guide covers all aspects of using VITA-1.5, from basic inference to advanced features and real-time interaction.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [Real-Time Interaction](#real-time-interaction)
- [Web Demo](#web-demo)
- [API Reference](#api-reference)
- [Configuration Options](#configuration-options)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### Basic Text Query

```python
import torch
from vita.model import VITA

# Initialize model
model = VITA.from_pretrained("/path/to/vita-1.5")

# Text-only query
response = model.generate(
    text="What is artificial intelligence?",
    max_length=512
)
print(response)
```

### Image Understanding

```python
# Image + text query
response = model.generate(
    text="Describe what you see in this image.",
    image_path="path/to/image.jpg",
    max_length=512
)
print(response)
```

### Audio Processing

```python
# Audio + text query
response = model.generate(
    text="What did the person say?",
    audio_path="path/to/audio.wav",
    max_length=512
)
print(response)
```

## 📖 Basic Usage

### Command Line Interface

#### Text Query
```bash
CUDA_VISIBLE_DEVICES=2 python video_audio_demo.py \
    --model_path [vita/path] \
    --image_path asset/vita_newlog.jpg \
    --model_type qwen2p5_instruct \
    --conv_mode qwen2p5_instruct \
    --question "Describe this image."
```

#### Audio Query
```bash
CUDA_VISIBLE_DEVICES=4 python video_audio_demo.py \
    --model_path [vita/path] \
    --image_path asset/vita_newlog.png \
    --model_type qwen2p5_instruct \
    --conv_mode qwen2p5_instruct \
    --audio_path asset/q1.wav
```

#### Noisy Audio Query
```bash
CUDA_VISIBLE_DEVICES=4 python video_audio_demo.py \
    --model_path [vita/path] \
    --image_path asset/vita_newlog.png \
    --model_type qwen2p5_instruct \
    --conv_mode qwen2p5_instruct \
    --audio_path asset/q2.wav
```

### Python API

#### Model Initialization

```python
from vita.model import VITA
from vita.config import ModelConfig

# Basic initialization
model = VITA.from_pretrained(
    model_path="/path/to/vita-1.5",
    device="cuda:0"
)

# Advanced initialization with custom config
config = ModelConfig(
    model_path="/path/to/vita-1.5",
    vision_tower="/path/to/InternViT-300M-448px",
    audio_encoder="/path/to/audio-encoder",
    max_length=2048,
    temperature=0.7,
    top_p=0.9
)
model = VITA(config)
```

#### Multimodal Inference

```python
# Text + Image
response = model.generate(
    text="Analyze this image and provide detailed insights.",
    image_path="image.jpg",
    max_length=1024,
    temperature=0.7
)

# Text + Audio
response = model.generate(
    text="What is the main topic of this conversation?",
    audio_path="conversation.wav",
    max_length=512
)

# Text + Image + Audio (multimodal)
response = model.generate(
    text="Based on the image and audio, what's happening?",
    image_path="scene.jpg",
    audio_path="background.wav",
    max_length=1024
)
```

## 🔥 Advanced Features

### Video Processing

```python
# Video analysis with frame extraction
response = model.generate(
    text="Describe the key events in this video.",
    video_path="video.mp4",
    max_frames=16,  # Number of frames to extract
    max_length=1024
)
```

### Batch Processing

```python
# Process multiple inputs
inputs = [
    {"text": "Describe this image.", "image_path": "img1.jpg"},
    {"text": "What's in this image?", "image_path": "img2.jpg"},
    {"text": "Analyze this scene.", "image_path": "img3.jpg"}
]

responses = model.generate_batch(inputs, max_length=512)
for i, response in enumerate(responses):
    print(f"Response {i+1}: {response}")
```

### Custom Prompts

```python
# Define custom conversation templates
custom_prompt = """
You are an expert image analyst. Please provide a detailed analysis of the given image, including:
1. Object identification
2. Scene description
3. Color analysis
4. Composition assessment

Image: {image}
Question: {question}
"""

response = model.generate(
    text=custom_prompt.format(
        image="<image>",
        question="Analyze this image comprehensively."
    ),
    image_path="image.jpg",
    max_length=1024
)
```

### Streaming Responses

```python
# For real-time response generation
def stream_generator(model, text, **kwargs):
    for chunk in model.generate_stream(text, **kwargs):
        yield chunk

# Usage
for chunk in stream_generator(
    model,
    text="Write a detailed story about...",
    max_length=1024
):
    print(chunk, end="", flush=True)
```

## 🎤 Real-Time Interaction

### Voice Activity Detection Setup

```bash
# Download VAD models
mkdir -p web_demo/wakeup_and_vad/resource/
cd web_demo/wakeup_and_vad/resource/

# Download silero VAD models
wget https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx
wget https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.jit
```

### Real-Time Server

```bash
# Start real-time interactive server
python -m web_demo.server \
    --model_path /path/to/vita-1.5 \
    --ip 0.0.0.0 \
    --port 8081 \
    --max_dynamic_patch 1  # Optimize for real-time
```

### Client Integration

```python
import requests
import json

# Real-time API client
class VITAClient:
    def __init__(self, server_url="http://localhost:8081"):
        self.server_url = server_url
    
    def send_audio(self, audio_data):
        response = requests.post(
            f"{self.server_url}/audio",
            files={"audio": audio_data}
        )
        return response.json()
    
    def send_text(self, text):
        response = requests.post(
            f"{self.server_url}/text",
            json={"text": text}
        )
        return response.json()

# Usage
client = VITAClient()
response = client.send_text("Hello, how are you?")
print(response["answer"])
```

## 🌐 Web Demo

### Basic Web Demo

```bash
# Setup demo environment
conda create -n vita_demo python=3.10 -y
conda activate vita_demo
pip install -r web_demo/web_demo_requirements.txt

# Prepare model weights
cp -rL VITA_ckpt/ demo_VITA_ckpt/
mv demo_VITA_ckpt/config.json demo_VITA_ckpt/origin_config.json

cd web_demo/vllm_tools
cp -rf qwen2p5_model_weight_file/* ../../demo_VITA_ckpt/
cp -rf vllm_file/* $CONDA_PREFIX/lib/python3.10/site-packages/vllm/model_executor/models/

# Run demo
python -m web_demo.web_ability_demo demo_VITA_ckpt/
```

### Custom Web Interface

```python
# Custom Flask app
from flask import Flask, request, jsonify, render_template
from vita.model import VITA

app = Flask(__name__)
model = VITA.from_pretrained("/path/to/vita-1.5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    text = data.get('text', '')
    image = data.get('image', None)
    audio = data.get('audio', None)
    
    response = model.generate(
        text=text,
        image_path=image,
        audio_path=audio,
        max_length=512
    )
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## 🔧 API Reference

### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | Required | Path to VITA-1.5 checkpoint |
| `vision_tower` | str | None | Path to vision encoder |
| `audio_encoder` | str | None | Path to audio encoder |
| `device` | str | "cuda:0" | Device for inference |
| `max_length` | int | 2048 | Maximum sequence length |
| `temperature` | float | 0.7 | Sampling temperature |
| `top_p` | float | 0.9 | Nucleus sampling parameter |
| `top_k` | int | 50 | Top-k sampling parameter |

### Generation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | Required | Input text prompt |
| `image_path` | str | None | Path to input image |
| `audio_path` | str | None | Path to input audio |
| `video_path` | str | None | Path to input video |
| `max_length` | int | 512 | Maximum output length |
| `temperature` | float | 0.7 | Sampling temperature |
| `top_p` | float | 0.9 | Nucleus sampling |
| `do_sample` | bool | True | Enable sampling |
| `stream` | bool | False | Enable streaming |

## ⚙️ Configuration Options

### Model Configuration

```python
# config.json
{
    "model_type": "qwen2p5_instruct",
    "conv_mode": "qwen2p5_instruct",
    "max_dynamic_patch": 12,  # 1 for real-time, 12 for quality
    "vision_tower": "/path/to/InternViT-300M-448px",
    "audio_encoder": "/path/to/audio-encoder",
    "mm_projector_type": "mlp2x_gelu",
    "mm_use_im_start_end": false,
    "mm_use_im_patch_token": true,
    "mm_patch_merge_type": "flat",
    "mm_vision_select_layer": -2,
    "mm_vision_select_feature": "patch"
}
```

### Environment Variables

```bash
# Performance optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# Model paths
export VITA_MODEL_PATH="/path/to/vita-1.5"
export VISION_TOWER_PATH="/path/to/InternViT-300M-448px"
export AUDIO_ENCODER_PATH="/path/to/audio-encoder"
```

## 💡 Best Practices

### Performance Optimization

1. **GPU Memory Management**
   ```python
   # Use gradient checkpointing for large models
   model.gradient_checkpointing_enable()
   
   # Set memory fraction
   torch.cuda.set_per_process_memory_fraction(0.9)
   ```

2. **Batch Processing**
   ```python
   # Process multiple inputs efficiently
   responses = model.generate_batch(
       inputs,
       batch_size=4,
       max_length=512
   )
   ```

3. **Caching**
   ```python
   # Enable model caching for repeated queries
   model.enable_cache()
   ```

### Input Preprocessing

1. **Image Optimization**
   ```python
   from PIL import Image
   
   def preprocess_image(image_path, max_size=448):
       image = Image.open(image_path)
       image = image.convert('RGB')
       image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
       return image
   ```

2. **Audio Preprocessing**
   ```python
   import librosa
   
   def preprocess_audio(audio_path, sr=16000):
       audio, _ = librosa.load(audio_path, sr=sr)
       return audio
   ```

### Error Handling

```python
def safe_generate(model, **kwargs):
    try:
        response = model.generate(**kwargs)
        return response
    except torch.cuda.OutOfMemoryError:
        # Reduce batch size or sequence length
        kwargs['max_length'] = kwargs.get('max_length', 512) // 2
        return model.generate(**kwargs)
    except Exception as e:
        print(f"Error: {e}")
        return None
```

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```python
# Solution: Reduce memory usage
model.generate(
    text="Your prompt",
    max_length=256,  # Reduce from default 512
    batch_size=1     # Ensure batch size is 1
)
```

#### 2. Slow Inference
```python
# Solution: Optimize configuration
config = {
    "max_dynamic_patch": 1,  # Reduce for speed
    "temperature": 0.1,      # Reduce randomness
    "do_sample": False       # Use greedy decoding
}
```

#### 3. Poor Audio Quality
```python
# Solution: Use proper audio format
# Ensure audio is 16kHz, mono, WAV format
audio = librosa.load(audio_path, sr=16000, mono=True)
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Verbose model loading
model = VITA.from_pretrained(
    model_path="/path/to/vita-1.5",
    verbose=True
)
```

---

**Note**: For optimal performance, ensure your system meets the minimum requirements and follow the configuration guidelines for your specific use case.
