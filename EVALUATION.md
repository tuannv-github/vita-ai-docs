# VITA-1.5 Evaluation Guide

This guide covers comprehensive evaluation of VITA-1.5 on various multimodal benchmarks, including setup, running evaluations, and interpreting results.

## 📋 Table of Contents

- [Overview](#overview)
- [Benchmark Setup](#benchmark-setup)
- [VLMEvalKit Integration](#vlmevalkit-integration)
- [Video-MME Evaluation](#video-mme-evaluation)
- [Custom Benchmarks](#custom-benchmarks)
- [Performance Metrics](#performance-metrics)
- [Results Interpretation](#results-interpretation)
- [Troubleshooting](#troubleshooting)

## 🌟 Overview

VITA-1.5 has been evaluated on multiple benchmarks to demonstrate its capabilities across different modalities and tasks:

- **Image Understanding**: MME, MMBench, MathVista, MMStar
- **Video Understanding**: Video-MME
- **Multimodal Reasoning**: MMMU, HallusionBench
- **OCR Tasks**: OCRBench
- **Mathematical Reasoning**: MathVista

## 🛠 Benchmark Setup

### Prerequisites

```bash
# Install evaluation dependencies
pip install opencompass
pip install vlmeval
pip install torch torchvision torchaudio
pip install transformers datasets
pip install openai  # For GPT-4 evaluation
```

### Environment Setup

```bash
# Set up evaluation environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_KEY="your-api-key"  # For GPT-4 evaluation
```

## 📊 VLMEvalKit Integration

### Installation

```bash
# Clone VLMEvalKit
git clone https://github.com/open-compass/VLMEvalKit
cd VLMEvalKit
pip install -e .
```

### Model Configuration

Edit `VLMEvalKit/vlmeval/config.py`:

```python
from functools import partial
from vlmeval.models import VITA, VITAQwen2

# Configure VITA models
vita_series = { 
    'vita': partial(VITA, model_path='/path/to/vita-1.5'),
    'vita_qwen2': partial(VITAQwen2, model_path='/path/to/vita-1.5'),
}

# Add to model registry
model_aliases = {
    'vita': 'vita',
    'vita_qwen2': 'vita_qwen2',
    # ... other models
}
```

### Judge Model Setup

#### Option 1: GPT-4 (Recommended)

```bash
# Set OpenAI API key
export OPENAI_API_KEY="sk-your-api-key"

# Configure in .env file
echo "OPENAI_API_KEY=sk-your-api-key" > .env
```

#### Option 2: Local Judge Model

```bash
# Start local judge server
CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server \
    /path/to/Qwen1.5-1.8B-Chat \
    --server-port 23333

# Configure .env file
cat > .env << EOF
OPENAI_API_KEY=sk-123456
OPENAI_API_BASE=http://0.0.0.0:23333/v1/chat/completions
LOCAL_LLM=/path/to/Qwen1.5-1.8B-Chat
EOF
```

### Running Evaluations

#### Single Benchmark

```bash
# MME evaluation
CUDA_VISIBLE_DEVICES=0 python run.py \
    --data MME \
    --model vita_qwen2 \
    --verbose

# MMBench evaluation
CUDA_VISIBLE_DEVICES=0 python run.py \
    --data MMBench_TEST_EN_V11 \
    --model vita_qwen2 \
    --verbose
```

#### Multiple Benchmarks

```bash
# Comprehensive evaluation
CUDA_VISIBLE_DEVICES=0 python run.py \
    --data MMBench_TEST_EN_V11 MMBench_TEST_CN_V11 MMStar MMMU_DEV_VAL MathVista_MINI HallusionBench AI2D_TEST OCRBench MMVet MME \
    --model vita_qwen2 \
    --verbose
```

#### Batch Evaluation

```bash
# Create evaluation script
cat > evaluate_vita.sh << 'EOF'
#!/bin/bash

BENCHMARKS=(
    "MME"
    "MMBench_TEST_EN_V11"
    "MMBench_TEST_CN_V11"
    "MMStar"
    "MMMU_DEV_VAL"
    "MathVista_MINI"
    "HallusionBench"
    "AI2D_TEST"
    "OCRBench"
    "MMVet"
)

for benchmark in "${BENCHMARKS[@]}"; do
    echo "Evaluating on $benchmark..."
    CUDA_VISIBLE_DEVICES=0 python run.py \
        --data $benchmark \
        --model vita_qwen2 \
        --verbose
done
EOF

chmod +x evaluate_vita.sh
./evaluate_vita.sh
```

## 🎥 Video-MME Evaluation

### Data Preparation

```bash
# Clone Video-MME repository
git clone https://github.com/BradyFU/Video-MME
cd Video-MME

# Download and extract dataset
# Follow instructions in Video-MME repository
```

### Frame Extraction

```python
# extract_frames.py
import cv2
import os
from pathlib import Path

def extract_frames(video_path, output_dir, max_frames=16):
    """Extract frames from video for evaluation"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices
    if frame_count <= max_frames:
        frame_indices = list(range(frame_count))
    else:
        frame_indices = [int(i * frame_count / max_frames) for i in range(max_frames)]
    
    # Extract frames
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{output_dir}/frame_{i:03d}.jpg", frame)
    
    cap.release()

# Extract frames for all videos
video_dir = "Video-MME/videos"
output_dir = "Video-MME-imgs"

for video_file in Path(video_dir).glob("*.mp4"):
    video_name = video_file.stem
    video_output_dir = Path(output_dir) / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    extract_frames(str(video_file), str(video_output_dir))
```

### Evaluation Scripts

#### Without Subtitles

```bash
cd ./videomme

# Set evaluation parameters
VIDEO_TYPE="s,m,l"  # short, medium, long videos
NAMES=(lyd jyg wzh wzz zcy by dyh lfy)  # evaluator names

# Run evaluation
for((i=0; i<${#NAMES[@]}; i++)) 
do
    CUDA_VISIBLE_DEVICES=6 python yt_video_inference_qa_imgs.py \
        --model-path /path/to/vita-1.5 \
        --model_type qwen2p5_instruct \
        --conv_mode qwen2p5_instruct \
        --responsible_man ${NAMES[i]} \
        --video_type $VIDEO_TYPE \
        --output_dir qa_wo_sub \
        --video_dir /path/to/Video-MME-imgs | tee logs/infer_${NAMES[i]}.log
done
```

#### With Subtitles

```bash
# Run evaluation with subtitles
for((i=0; i<${#NAMES[@]}; i++)) 
do
    CUDA_VISIBLE_DEVICES=7 python yt_video_inference_qa_imgs.py \
        --model-path /path/to/vita-1.5 \
        --model_type qwen2p5_instruct \
        --conv_mode qwen2p5_instruct \
        --responsible_man ${NAMES[i]} \
        --video_type $VIDEO_TYPE \
        --output_dir qa_w_sub \
        --video_dir /path/to/Video-MME-imgs \
        --use_subtitles | tee logs/infer_sub_${NAMES[i]}.log
done
```

### Results Parsing

```bash
# Parse results
python parse_answer.py --video_types "s,m,l" --result_dir qa_wo_sub
python parse_answer.py --video_types "s,m,l" --result_dir qa_w_sub

# Generate final scores
python calculate_scores.py --result_dir qa_wo_sub
python calculate_scores.py --result_dir qa_w_sub
```

## 🧪 Custom Benchmarks

### Creating Custom Evaluation

```python
# custom_evaluator.py
from vlmeval.api import VLMEvalAPI
from vita.model import VITA

class CustomVITAEvaluator:
    def __init__(self, model_path):
        self.model = VITA.from_pretrained(model_path)
    
    def evaluate_image_qa(self, image_path, question):
        """Evaluate image question answering"""
        response = self.model.generate(
            text=question,
            image_path=image_path,
            max_length=512
        )
        return response
    
    def evaluate_audio_qa(self, audio_path, question):
        """Evaluate audio question answering"""
        response = self.model.generate(
            text=question,
            audio_path=audio_path,
            max_length=512
        )
        return response
    
    def evaluate_multimodal_qa(self, image_path, audio_path, question):
        """Evaluate multimodal question answering"""
        response = self.model.generate(
            text=question,
            image_path=image_path,
            audio_path=audio_path,
            max_length=512
        )
        return response

# Usage
evaluator = CustomVITAEvaluator("/path/to/vita-1.5")

# Evaluate on custom dataset
results = []
for item in custom_dataset:
    result = evaluator.evaluate_image_qa(
        item['image_path'],
        item['question']
    )
    results.append({
        'question': item['question'],
        'predicted': result,
        'ground_truth': item['answer']
    })
```

### Benchmark Dataset Format

```json
{
    "dataset_name": "custom_benchmark",
    "version": "1.0",
    "description": "Custom evaluation dataset",
    "data": [
        {
            "id": "sample_001",
            "image": "path/to/image.jpg",
            "audio": "path/to/audio.wav",
            "question": "What do you see and hear?",
            "answer": "Expected answer",
            "category": "multimodal"
        }
    ]
}
```

## 📈 Performance Metrics

### Standard Metrics

#### Accuracy Metrics
- **Overall Accuracy**: Percentage of correct answers
- **Per-Category Accuracy**: Accuracy by question type
- **Per-Difficulty Accuracy**: Accuracy by difficulty level

#### Latency Metrics
- **Inference Time**: Time per sample
- **Throughput**: Samples per second
- **Memory Usage**: Peak GPU memory consumption

### VITA-1.5 Performance

#### Image Understanding Benchmarks

| Benchmark | VITA-1.0 | VITA-1.5 | Improvement |
|-----------|----------|----------|-------------|
| MME | 58.2 | 70.8 | +12.6 |
| MMBench | 61.4 | 72.1 | +10.7 |
| MathVista | 45.3 | 52.8 | +7.5 |
| MMStar | 38.7 | 45.2 | +6.5 |

#### Audio Processing Performance

| Metric | VITA-1.0 | VITA-1.5 | Improvement |
|--------|----------|----------|-------------|
| ASR WER | 18.4% | 7.5% | -10.9% |
| TTS Quality | 3.2/5.0 | 4.1/5.0 | +0.9 |
| Latency | 4.0s | 1.5s | -2.5s |

### Performance Analysis Script

```python
# performance_analysis.py
import json
import matplotlib.pyplot as plt
import pandas as pd

def analyze_results(results_file):
    """Analyze evaluation results"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Calculate metrics
    total_samples = len(results)
    correct_samples = sum(1 for r in results if r['correct'])
    accuracy = correct_samples / total_samples
    
    # Per-category analysis
    categories = {}
    for result in results:
        category = result.get('category', 'unknown')
        if category not in categories:
            categories[category] = {'total': 0, 'correct': 0}
        categories[category]['total'] += 1
        if result['correct']:
            categories[category]['correct'] += 1
    
    # Calculate category accuracies
    category_accuracies = {
        cat: data['correct'] / data['total']
        for cat, data in categories.items()
    }
    
    return {
        'overall_accuracy': accuracy,
        'category_accuracies': category_accuracies,
        'total_samples': total_samples
    }

def plot_results(analysis_results):
    """Plot evaluation results"""
    categories = list(analysis_results['category_accuracies'].keys())
    accuracies = list(analysis_results['category_accuracies'].values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(categories, accuracies)
    plt.title('VITA-1.5 Performance by Category')
    plt.xlabel('Category')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('performance_analysis.png')
    plt.show()

# Usage
results = analyze_results('evaluation_results.json')
plot_results(results)
```

## 📊 Results Interpretation

### Understanding Scores

#### MME (Multimodal Evaluation)
- **Perception**: Object detection, OCR, counting
- **Cognition**: Reasoning, spatial understanding
- **Overall**: Weighted average of all tasks

#### MMBench
- **English/Chinese**: Language-specific performance
- **Per-Category**: Performance by question type
- **Difficulty Levels**: Easy, Medium, Hard

#### Video-MME
- **Short/Medium/Long**: Video length categories
- **With/Without Subtitles**: Subtitle impact
- **Per-Evaluator**: Consistency across evaluators

### Statistical Significance

```python
# statistical_analysis.py
import scipy.stats as stats
import numpy as np

def compare_models(results1, results2):
    """Compare two model results statistically"""
    # Convert to arrays
    scores1 = np.array([r['score'] for r in results1])
    scores2 = np.array([r['score'] for r in results2])
    
    # Perform t-test
    t_stat, p_value = stats.ttest_rel(scores1, scores2)
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
    cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05
    }
```

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory During Evaluation

```python
# Solution: Reduce batch size
evaluation_config = {
    'batch_size': 1,
    'max_length': 256,
    'gradient_checkpointing': True
}
```

#### 2. Slow Evaluation

```bash
# Solution: Use multiple GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py \
    --data MME \
    --model vita_qwen2 \
    --num_gpus 4
```

#### 3. Inconsistent Results

```python
# Solution: Set random seeds
import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
```

### Debug Mode

```bash
# Enable debug logging
export VLMEVAL_DEBUG=1
python run.py --data MME --model vita_qwen2 --verbose
```

---

**Note**: Evaluation results may vary based on hardware configuration, random seeds, and model versions. Always report evaluation conditions for reproducibility.
