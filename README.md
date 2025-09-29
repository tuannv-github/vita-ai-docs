# VITA-1.5 Documentation

Welcome to the comprehensive documentation for VITA-1.5, an advanced Vision-Language-Audio-Action (VLAA) multimodal large language model that enables real-time interactive communication across multiple modalities.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Evaluation & Benchmarks](#evaluation--benchmarks)
- [Development](#development)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## 🌟 Overview

VITA-1.5 is a state-of-the-art multimodal large language model that supports real-time vision and speech interaction. Building upon VITA-1.0, this version introduces significant improvements in latency reduction, multimodal performance, and speech processing capabilities.

### What's New in VITA-1.5?

- **🚀 Reduced Interaction Latency**: End-to-end speech interaction latency reduced from ~4 seconds to ~1.5 seconds
- **📈 Enhanced Multimodal Performance**: Average performance on benchmarks increased from 59.8 to 70.8
- **🎤 Improved Speech Processing**: ASR WER reduced from 18.4 to 7.5 with end-to-end TTS module
- **🔄 Progressive Training Strategy**: Minimal impact on vision-language performance when adding speech
- **🌟 ModelScope Integration**: Official demo now available on ModelScope platform
- **📊 VLMEvalKit Support**: Full integration with OpenCompass evaluation framework

## ✨ Key Features

- **Multimodal Understanding**: Supports text, images, audio, and video inputs
- **Real-time Interaction**: Low-latency speech-to-speech communication
- **Bilingual Support**: Works with both English and Chinese
- **High Performance**: Competitive results on major MLLM benchmarks
- **Open Source**: Fully open-source with comprehensive documentation

## 🚀 Installation

### Prerequisites

- Python 3.10
- CUDA-compatible GPU
- Conda or virtual environment

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/VITA-MLLM/VITA
cd VITA

# Create conda environment
conda create -n vita python=3.10 -y
conda activate vita

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### Model Weights

Download the required model weights:
- [VITA-1.5 checkpoint](https://huggingface.co/VITA-MLLM/VITA-1.5/tree/main)
- [InternViT-300M-448px](https://huggingface.co/OpenGVLab/InternViT-300M-448px)
- [Audio encoder](https://huggingface.co/VITA-MLLM/VITA-1.5/tree/main/audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning)

## 🏃‍♂️ Quick Start

### Text Query Example

```bash
CUDA_VISIBLE_DEVICES=2 python video_audio_demo.py \
    --model_path [vita/path] \
    --image_path asset/vita_newlog.jpg \
    --model_type qwen2p5_instruct \
    --conv_mode qwen2p5_instruct \
    --question "Describe this image."
```

### Audio Query Example

```bash
CUDA_VISIBLE_DEVICES=4 python video_audio_demo.py \
    --model_path [vita/path] \
    --image_path asset/vita_newlog.png \
    --model_type qwen2p5_instruct \
    --conv_mode qwen2p5_instruct \
    --audio_path asset/q1.wav
```

### Noisy Audio Query Example

```bash
CUDA_VISIBLE_DEVICES=4 python video_audio_demo.py \
    --model_path [vita/path] \
    --image_path asset/vita_newlog.png \
    --model_type qwen2p5_instruct \
    --conv_mode qwen2p5_instruct \
    --audio_path asset/q2.wav
```

## 📖 Usage Examples

### Basic Demo

Run the basic web demo:

```bash
conda create -n vita_demo python==3.10
conda activate vita_demo
pip install -r web_demo/web_demo_requirements.txt

# Setup model weights
cp -rL VITA_ckpt/ demo_VITA_ckpt/
mv demo_VITA_ckpt/config.json demo_VITA_ckpt/origin_config.json

cd ./web_demo/vllm_tools
cp -rf qwen2p5_model_weight_file/* ../../demo_VITA_ckpt/
cp -rf vllm_file/* your_anaconda/envs/vita_demo/lib/python3.10/site-packages/vllm/model_executor/models/

# Run demo
python -m web_demo.web_ability_demo demo_VITA_ckpt/
```

**Note**: VITA has been accelerated using [vLLM](https://github.com/vllm-project/vllm). Since VITA has not yet been integrated into vLLM, you need to make some modifications to the vLLM code to adapt it for VITA.

### Real-Time Interactive Demo

For real-time interaction:

```bash
# Install additional dependencies
pip install flask==3.1.0 flask-socketio==5.5.0 cryptography==44.0.0 timm==1.0.12

# Download VAD models
# Place silero_vad.onnx and silero_vad.jit in ./web_demo/wakeup_and_vad/resource/

# Configure for real-time (set max_dynamic_patch to 1 in config.json)
python -m web_demo.server --model_path demo_VITA_ckpt --ip 0.0.0.0 --port 8081
```

**Important Notes**:
- Make sure you have executed the basic demo setup instructions first
- For better real-time interactive experience, set `max_dynamic_patch` to 1 in `demo_VITA_ckpt/config.json`
- When running the basic demo, you can set it to the default value of 12 to enhance the model's visual capabilities

## 🔧 API Reference

### Model Configuration

Key configuration parameters:

- `model_path`: Path to VITA-1.5 checkpoint
- `model_type`: Model type (qwen2p5_instruct)
- `conv_mode`: Conversation mode
- `vision_tower`: Vision encoder path
- `audio_encoder`: Audio encoder path

### Input Formats

- **Text**: Standard text input
- **Images**: JPG, PNG formats supported
- **Audio**: WAV format recommended
- **Video**: MP4 format with frame extraction

## 📊 Evaluation & Benchmarks

### Performance Metrics

VITA-1.5 achieves competitive performance on major benchmarks:

- **MME**: Comprehensive multimodal evaluation
- **MMBench**: Multimodal benchmark suite
- **MathVista**: Mathematical reasoning with visual inputs
- **Video-MME**: Video understanding evaluation

### Running Evaluations

Using VLMEvalKit:

```bash
# Configure model path in VLMEvalKit/vlmeval/config.py
vita_series = { 
    'vita': partial(VITA, model_path='/path/to/model'),
    'vita_qwen2': partial(VITAQwen2, model_path='/path/to/model'),
}

# Follow the instructions in VLMEvalKit to set the GPT as the judge model
# If OpenAI API is not available, you can use a local model as judge
# For example, Qwen1.5-1.8B-Chat works well compared to GPT-4, except in MM-Vet

# Run evaluation
CUDA_VISIBLE_DEVICES=0 python run.py --data MMBench_TEST_EN_V11 MMBench_TEST_CN_V11 MMStar MMMU_DEV_VAL MathVista_MINI HallusionBench AI2D_TEST OCRBench MMVet MME --model vita_qwen2 --verbose
```

## 🛠 Development

### Training

For continual training:

```bash
export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUTPUT_DIR=/path/to/output
bash script/train/finetuneTaskNeg_qwen_nodes.sh ${OUTPUT_DIR}
```

**Required Downloads**:
- [VITA-1.5 checkpoint](https://huggingface.co/VITA-MLLM/VITA-1.5/tree/main)
- [InternViT-300M-448px](https://huggingface.co/OpenGVLab/InternViT-300M-448px)
- [Audio encoder](https://huggingface.co/VITA-MLLM/VITA-1.5/tree/main/audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning)

### Data Preparation

Training data format:

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
                "value": "Response text here"
            }
        ],
        "image": "path/to/image.jpg",
        "audio": ["path/to/audio.wav"]
    }
]
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black vita/
isort vita/
```

## 📚 Citation

If you use VITA-1.5 in your research, please cite:

```bibtex
@article{fu2025vita,
  title={VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction},
  author={Fu, Chaoyou and Lin, Haojia and Wang, Xiong and Zhang, Yi-Fan and Shen, Yunhang and Liu, Xiaoyu and Li, Yangze and Long, Zuwei and Gao, Heting and Li, Ke and others},
  journal={arXiv preprint arXiv:2501.01957},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Paper**: [VITA-1.5 Technical Report](https://arxiv.org/pdf/2501.01957)
- **Demo**: [ModelScope Demo](https://modelscope.cn/studios/modelscope/VITA1.5_demo)
- **Video**: [VITA-1.5 Demo Show](https://youtu.be/tyi6SVFT5mM?si=fkMQCrwa5fVnmEe7)
- **VITA-1.0**: [Previous Version](https://vita-home.github.io/)
- **WeChat Group**: [Join Discussion](./asset/wechat-group.jpg)
- **GitHub Repository**: [VITA-MLLM/VITA](https://github.com/VITA-MLLM/VITA)
- **Hugging Face**: [VITA-MLLM/VITA-1.5](https://huggingface.co/VITA-MLLM/VITA-1.5)

## 🙏 Acknowledgments

VITA-1.5 builds upon excellent open-source projects including LLaVA-1.5, Bunny, ChatUnivi, InternVL, InternViT, Qwen-2.5, VLMEvalKit, and Mixtral 8*7B.

---

**Note**: VITA is trained on large-scale open-source corpus. Any content generated by VITA does not represent the views of the model developers. Please use responsibly.
