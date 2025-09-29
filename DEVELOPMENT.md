# VITA-1.5 Development Guide

This guide covers development aspects of VITA-1.5, including training, fine-tuning, data preparation, and contributing to the project.

## 📋 Table of Contents

- [Development Setup](#development-setup)
- [Training](#training)
- [Data Preparation](#data-preparation)
- [Fine-tuning](#fine-tuning)
- [Model Architecture](#model-architecture)
- [Contributing](#contributing)
- [Code Style](#code-style)
- [Testing](#testing)
- [Debugging](#debugging)

## 🛠 Development Setup

### Prerequisites

```bash
# Install development dependencies
pip install -r requirements-dev.txt
pip install pre-commit
pip install pytest
pip install black isort flake8
```

### Development Environment

```bash
# Clone repository
git clone https://github.com/VITA-MLLM/VITA
cd VITA

# Create development environment
conda create -n vita-dev python=3.10 -y
conda activate vita-dev

# Install in development mode
pip install -e .

# Setup pre-commit hooks
pre-commit install
```

### IDE Configuration

#### VS Code Settings

```json
{
    "python.defaultInterpreterPath": "~/anaconda3/envs/vita-dev/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

#### PyCharm Configuration

1. Set Python interpreter to conda environment
2. Enable code inspection
3. Configure external tools for black and isort

## 🎓 Training

### Training Pipeline Overview

VITA-1.5 uses a progressive training strategy with three main stages:

1. **Stage 1**: Vision-Language Alignment
2. **Stage 2**: Audio-Language Alignment  
3. **Stage 3**: End-to-End Multimodal Training

### Stage 1: Vision-Language Training

```bash
# Prepare vision-language data
python data_tools/prepare_vision_language_data.py \
    --input_dir /path/to/vision_language_data \
    --output_dir /path/to/processed_data

# Start training
export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash script/train/stage1_vision_language.sh \
    --data_path /path/to/processed_data \
    --output_dir /path/to/stage1_output \
    --num_gpus 8
```

### Stage 2: Audio-Language Training

```bash
# Download required weights
# 1. VITA-1.5 checkpoint: https://huggingface.co/VITA-MLLM/VITA-1.5/tree/main
# 2. InternViT-300M-448px: https://huggingface.co/OpenGVLab/InternViT-300M-448px
# 3. Audio encoder: https://huggingface.co/VITA-MLLM/VITA-1.5/tree/main/audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning

# Replace paths in script/train/finetuneTaskNeg_qwen_nodes.sh:
# --model_name_or_path VITA1.5_ckpt
# --vision_tower InternViT-300M-448px
# --audio_encoder audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning

# Start training
export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUTPUT_DIR=/mnt/cfs/lhj/videomllm_ckpt/outputs/vita_video_audio
bash script/train/finetuneTaskNeg_qwen_nodes.sh ${OUTPUT_DIR}
```

### Stage 3: End-to-End Training

```bash
# Prepare multimodal data
python data_tools/prepare_multimodal_data.py \
    --vision_language_data /path/to/vision_language_data \
    --audio_language_data /path/to/audio_language_data \
    --output_dir /path/to/multimodal_data

# Start end-to-end training
bash script/train/stage3_end_to_end.sh \
    --data_path /path/to/multimodal_data \
    --stage2_ckpt /path/to/stage2_output \
    --output_dir /path/to/final_output \
    --num_gpus 8
```

### Training Configuration

#### Training Script Template

```bash
#!/bin/bash
# script/train/custom_training.sh

export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTPUT_DIR=$1
DATA_PATH=$2
NUM_GPUS=${3:-8}

python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    vita/train/trainer.py \
    --model_name_or_path /path/to/base_model \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
```

### Monitoring Training

#### Weights & Biases Integration

```python
# Add to training script
import wandb

wandb.init(
    project="vita-1.5-training",
    config={
        "learning_rate": 2e-5,
        "batch_size": 16,
        "epochs": 3,
        "model": "vita-1.5"
    }
)

# Log metrics during training
wandb.log({
    "train_loss": train_loss,
    "learning_rate": current_lr,
    "epoch": epoch
})
```

#### TensorBoard Integration

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/vita_training')

# Log metrics
writer.add_scalar('Loss/Train', train_loss, step)
writer.add_scalar('Learning_Rate', learning_rate, step)
writer.add_histogram('Model/Weights', model.parameters(), step)
```

## 📊 Data Preparation

### Data Format

#### Vision-Language Data

```json
[
    {
        "id": "sample_001",
        "image": "path/to/image.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nDescribe this image in detail."
            },
            {
                "from": "gpt",
                "value": "This image shows a beautiful sunset over a mountain range..."
            }
        ]
    }
]
```

#### Audio-Language Data

```json
[
    {
        "id": "audio_001",
        "audio": "path/to/audio.wav",
        "conversations": [
            {
                "from": "human",
                "value": "<audio>\nWhat did the person say?"
            },
            {
                "from": "gpt",
                "value": "The person said: 'Hello, how are you today?'"
            }
        ]
    }
]
```

#### Multimodal Data

```json
[
    {
        "set": "sharegpt4",
        "id": "000000000164",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n<audio>\n"
            },
            {
                "from": "gpt",
                "value": "This is a well-organized kitchen with a clean, modern aesthetic. The kitchen features a white countertop against a white wall, creating a bright and airy atmosphere."
            }
        ],
        "image": "coco/images/train2017/000000000164.jpg",
        "audio": [
            "new_value_dict_0717/output_wavs/f61cf238b7872b4903e1fc15dcb5a50c.wav"
        ]
    }
]
```

### Data Configuration

Add the data class configuration in `vita/config/__init__.py`:

```python
from .dataset_config import *

NaturalCap = [ShareGPT4V]

DataConfig = {
    "Pretrain_video": NaturalCap,
}
```

Update `vita/config/dataset_config.py`:

```python
AudioFolder = ""
FolderDict = {
    "sharegpt4": "",
}

ShareGPT4V = {"chat_path": ""}
```

### Data Processing Pipeline

```python
# data_tools/process_data.py
import json
import os
from pathlib import Path
from typing import List, Dict, Any

class DataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.audio_extensions = {'.wav', '.mp3', '.flac'}
    
    def process_dataset(self, input_dir: str, output_file: str):
        """Process raw dataset into training format"""
        data = []
        
        for item in self._load_raw_data(input_dir):
            processed_item = self._process_item(item)
            if processed_item:
                data.append(processed_item)
        
        # Save processed data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load_raw_data(self, input_dir: str) -> List[Dict]:
        """Load raw data from directory"""
        # Implementation depends on data format
        pass
    
    def _process_item(self, item: Dict) -> Dict:
        """Process individual data item"""
        # Validate and clean data
        if not self._validate_item(item):
            return None
        
        # Convert to training format
        processed = {
            "id": item.get("id", ""),
            "conversations": self._format_conversations(item),
        }
        
        # Add modality-specific fields
        if "image" in item:
            processed["image"] = item["image"]
        if "audio" in item:
            processed["audio"] = item["audio"]
        
        return processed
    
    def _validate_item(self, item: Dict) -> bool:
        """Validate data item"""
        # Check required fields
        if "conversations" not in item:
            return False
        
        # Validate file paths
        if "image" in item and not os.path.exists(item["image"]):
            return False
        if "audio" in item and not os.path.exists(item["audio"]):
            return False
        
        return True
    
    def _format_conversations(self, item: Dict) -> List[Dict]:
        """Format conversations for training"""
        conversations = []
        
        for conv in item["conversations"]:
            formatted_conv = {
                "from": conv["from"],
                "value": conv["value"]
            }
            conversations.append(formatted_conv)
        
        return conversations

# Usage
processor = DataProcessor({
    "max_length": 2048,
    "image_size": 448,
    "audio_sample_rate": 16000
})

processor.process_dataset(
    input_dir="/path/to/raw_data",
    output_file="/path/to/processed_data.json"
)
```

### Data Augmentation

```python
# data_tools/augmentation.py
import torch
import torchvision.transforms as transforms
import librosa
import numpy as np

class DataAugmentation:
    def __init__(self):
        self.image_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomResizedCrop(448, scale=(0.8, 1.0))
        ])
    
    def augment_image(self, image):
        """Augment image data"""
        return self.image_transforms(image)
    
    def augment_audio(self, audio, sample_rate=16000):
        """Augment audio data"""
        # Add noise
        noise = np.random.normal(0, 0.01, len(audio))
        audio = audio + noise
        
        # Time stretching
        if np.random.random() < 0.3:
            stretch_factor = np.random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=stretch_factor)
        
        # Pitch shifting
        if np.random.random() < 0.3:
            pitch_shift = np.random.randint(-2, 3)
            audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_shift)
        
        return audio
```

## 🔧 Fine-tuning

### Continual Training

```bash
# Fine-tune on custom dataset
export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTPUT_DIR=/path/to/finetune_output
DATA_PATH=/path/to/custom_data

bash script/train/finetune_custom.sh \
    --model_path /path/to/vita-1.5 \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --num_epochs 1 \
    --learning_rate 1e-5
```

### LoRA Fine-tuning

```python
# vita/train/lora_trainer.py
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

def setup_lora_model(model, config):
    """Setup LoRA for efficient fine-tuning"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    return model

# Usage in training script
model = setup_lora_model(base_model, lora_config)
```

### Custom Loss Functions

```python
# vita/train/custom_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        self.alpha = alpha  # Text loss weight
        self.beta = beta    # Vision loss weight
        self.gamma = gamma  # Audio loss weight
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets):
        # Text loss
        text_loss = self.ce_loss(outputs.logits, targets.text_labels)
        
        # Vision loss (if applicable)
        vision_loss = 0
        if hasattr(outputs, 'vision_logits') and targets.vision_labels is not None:
            vision_loss = self.ce_loss(outputs.vision_logits, targets.vision_labels)
        
        # Audio loss (if applicable)
        audio_loss = 0
        if hasattr(outputs, 'audio_logits') and targets.audio_labels is not None:
            audio_loss = self.ce_loss(outputs.audio_logits, targets.audio_labels)
        
        total_loss = (
            self.alpha * text_loss +
            self.beta * vision_loss +
            self.gamma * audio_loss
        )
        
        return total_loss, {
            'text_loss': text_loss,
            'vision_loss': vision_loss,
            'audio_loss': audio_loss,
            'total_loss': total_loss
        }
```

## 🏗 Model Architecture

### Custom Model Components

```python
# vita/model/components.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class VisionEncoder(nn.Module):
    def __init__(self, model_name="OpenGVLab/InternViT-300M-448px"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.encoder.config.hidden_size, 4096)
    
    def forward(self, images):
        features = self.encoder(images).last_hidden_state
        projected = self.projection(features)
        return projected

class AudioEncoder(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.encoder.config.hidden_size, 4096)
    
    def forward(self, audio):
        features = self.encoder(audio).last_hidden_state
        projected = self.projection(features)
        return projected

class MultimodalProjector(nn.Module):
    def __init__(self, input_dim=4096, output_dim=4096):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, features):
        return self.projection(features)
```

### Model Configuration

```python
# vita/config/model_config.py
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class VITAConfig:
    # Model paths
    model_path: str
    vision_tower: Optional[str] = None
    audio_encoder: Optional[str] = None
    
    # Architecture
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    intermediate_size: int = 11008
    
    # Training
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    
    # Modality settings
    use_vision: bool = True
    use_audio: bool = True
    vision_select_layer: int = -2
    audio_select_layer: int = -2
    
    # Optimization
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True
```

## 🤝 Contributing

### Development Workflow

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Add tests**
5. **Run tests and linting**
   ```bash
   pytest tests/
   black vita/
   isort vita/
   flake8 vita/
   ```
6. **Commit your changes**
   ```bash
   git commit -m "Add your feature"
   ```
7. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
8. **Create a pull request**

### Code Style

#### Python Style Guide

```python
# Follow PEP 8
# Use type hints
def process_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process input data and return results.
    
    Args:
        data: List of data items to process
        
    Returns:
        Dictionary containing processed results
        
    Raises:
        ValueError: If data format is invalid
    """
    if not data:
        raise ValueError("Data cannot be empty")
    
    # Process data
    results = {}
    for item in data:
        processed_item = _process_item(item)
        results[item['id']] = processed_item
    
    return results

def _process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Process individual data item."""
    return {
        'id': item['id'],
        'processed': True
    }
```

#### Documentation Style

```python
class VITAModel:
    """VITA-1.5 Multimodal Large Language Model.
    
    This class implements the VITA-1.5 model for multimodal understanding,
    supporting text, image, audio, and video inputs.
    
    Example:
        >>> model = VITAModel.from_pretrained("/path/to/model")
        >>> response = model.generate(
        ...     text="Describe this image",
        ...     image_path="image.jpg"
        ... )
        >>> print(response)
    """
    
    def __init__(self, config: VITAConfig):
        """Initialize VITA model.
        
        Args:
            config: Model configuration object
        """
        self.config = config
        self._setup_model()
    
    def generate(self, text: str, **kwargs) -> str:
        """Generate response for given input.
        
        Args:
            text: Input text prompt
            **kwargs: Additional arguments (image_path, audio_path, etc.)
            
        Returns:
            Generated response text
        """
        # Implementation
        pass
```

## 🧪 Testing

### Unit Tests

```python
# tests/test_model.py
import pytest
import torch
from vita.model import VITAModel
from vita.config import VITAConfig

class TestVITAModel:
    def setup_method(self):
        """Setup test fixtures."""
        self.config = VITAConfig(
            model_path="/path/to/test/model",
            max_length=512
        )
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = VITAModel(self.config)
        assert model is not None
        assert model.config == self.config
    
    def test_text_generation(self):
        """Test text generation."""
        model = VITAModel(self.config)
        response = model.generate("Hello, world!")
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_image_understanding(self):
        """Test image understanding."""
        model = VITAModel(self.config)
        response = model.generate(
            text="Describe this image",
            image_path="tests/fixtures/test_image.jpg"
        )
        assert isinstance(response, str)
    
    @pytest.mark.parametrize("input_text,expected_length", [
        ("Short text", 10),
        ("This is a longer text that should generate a longer response", 50),
    ])
    def test_response_length(self, input_text, expected_length):
        """Test response length for different inputs."""
        model = VITAModel(self.config)
        response = model.generate(input_text)
        assert len(response) >= expected_length
```

### Integration Tests

```python
# tests/test_integration.py
import pytest
from vita.model import VITAModel
from vita.data import DataLoader

class TestIntegration:
    def test_end_to_end_pipeline(self):
        """Test complete inference pipeline."""
        model = VITAModel.from_pretrained("/path/to/model")
        dataloader = DataLoader("/path/to/test/data")
        
        for batch in dataloader:
            responses = model.generate_batch(batch)
            assert len(responses) == len(batch)
            assert all(isinstance(r, str) for r in responses)
    
    def test_multimodal_inference(self):
        """Test multimodal inference."""
        model = VITAModel.from_pretrained("/path/to/model")
        
        response = model.generate(
            text="What do you see and hear?",
            image_path="tests/fixtures/test_image.jpg",
            audio_path="tests/fixtures/test_audio.wav"
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_model.py

# Run with coverage
pytest --cov=vita tests/

# Run with verbose output
pytest -v tests/
```

## 🐛 Debugging

### Debug Configuration

```python
# vita/utils/debug.py
import logging
import torch
import os

def setup_debug_logging(level=logging.DEBUG):
    """Setup debug logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('debug.log'),
            logging.StreamHandler()
        ]
    )

def enable_debug_mode():
    """Enable debug mode for development."""
    # Enable CUDA debugging
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Enable PyTorch debugging
    torch.autograd.set_detect_anomaly(True)
    
    # Setup logging
    setup_debug_logging()

def debug_model_parameters(model):
    """Debug model parameters and gradients."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            logging.debug(f"{name}: {param.data.norm():.4f} (grad: {param.grad.norm():.4f})")
        else:
            logging.debug(f"{name}: {param.data.norm():.4f} (no grad)")
```

### Memory Debugging

```python
# vita/utils/memory_debug.py
import torch
import psutil
import GPUtil

def log_memory_usage():
    """Log current memory usage."""
    # CPU memory
    cpu_memory = psutil.virtual_memory()
    logging.info(f"CPU Memory: {cpu_memory.percent}% used")
    
    # GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        gpu_memory_max = torch.cuda.max_memory_allocated() / 1024**3
        logging.info(f"GPU Memory: {gpu_memory:.2f}GB allocated, {gpu_memory_max:.2f}GB max")

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
```

---

**Note**: This development guide assumes familiarity with PyTorch, transformers, and multimodal machine learning. For specific implementation details, refer to the source code and existing training scripts.
