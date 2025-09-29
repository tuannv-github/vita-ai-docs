# VITA Dataset Documentation

This document provides comprehensive information about the datasets used in each training stage of the VITA (Vision, Audio, and Text) multimodal model, including download instructions, configuration, and usage in the codebase.

## 📋 Table of Contents

- [Overview](#overview)
- [Training Stages](#training-stages)
- [Stage 1: Vision-Language Alignment](#stage-1-vision-language-alignment)
- [Stage 2: Audio-Language Alignment](#stage-2-audio-language-alignment)
- [Stage 3: Multimodal Fine-tuning](#stage-3-multimodal-fine-tuning)
- [Dataset Download Instructions](#dataset-download-instructions)
- [Dataset Configuration](#dataset-configuration)
- [Code Usage](#code-usage)
- [Data Processing Tools](#data-processing-tools)

## 🌟 Overview

VITA-1.5 uses a progressive training strategy with three main stages, each requiring specific datasets:

1. **Stage 1**: Vision-Language Alignment (Pretrain MLP)
2. **Stage 2**: Audio-Language Alignment (Pretrain Audio MLP)  
3. **Stage 3**: End-to-End Multimodal Training (Fine-tune Task)

This approach ensures minimal impact on vision-language performance when adding speech capabilities.

## 🎯 What is "Pretrain"?

### **Pretraining in Machine Learning**

**Pretraining** is the initial training phase where a model learns fundamental representations and capabilities before being fine-tuned for specific tasks. In the context of VITA:

#### **General Pretraining Concept**
- **Purpose**: Learn basic multimodal understanding and alignment
- **Data**: Large-scale, diverse datasets (ShareGPT4V, COCO, LLaVA-Instruct)
- **Goal**: Establish foundational connections between modalities (vision, audio, text)
- **Approach**: Train specific components while keeping others frozen

#### **VITA's Pretraining Strategy**
VITA uses **progressive pretraining** - training different components sequentially to avoid catastrophic forgetting:

```
Base Model (Qwen2.5-7B) + Vision Encoder + Audio Encoder
    ↓
Stage 1: Pretrain Vision Projector (MLP)
    ↓
Stage 2: Pretrain Audio Projector (MLP)  
    ↓
Stage 3: Fine-tune All Components Together
```

### **VITA Pretraining Stages Explained**

#### **Stage 1: Vision-Language Pretraining**
```bash
# Training Script: pretrain_mlp_qwen.sh
--tune_mm_mlp_adapter True          # Train vision projector
--freeze_audio_encoder True         # Keep audio encoder frozen
--freeze_audio_encoder_adapter True # Keep audio adapter frozen
--learning_rate 5e-4                # Higher LR for projector training
```

**What happens**:
- **Vision Projector (MLP)**: Trained to align vision features with language model
- **Language Model**: Kept frozen (pretrained weights preserved)
- **Audio Components**: Completely frozen
- **Dataset**: ShareGPT4V, COCO, LLaVA-Instruct (vision-language data)

**Output**: Vision projector weights that can convert image features to language model embeddings

#### **Stage 2: Audio-Language Pretraining**
```bash
# Training Script: pretrain_audio_mlp_qwen.sh
--tune_audio_mlp_adapter True       # Train audio projector
--tune_mm_mlp_adapter False         # Keep vision projector frozen
--freeze_audio_encoder True         # Keep audio encoder frozen
--learning_rate 5e-4                # Higher LR for audio projector
```

**What happens**:
- **Audio Projector (MLP)**: Trained to align audio features with language model
- **Vision Projector**: Loaded from Stage 1, kept frozen
- **Language Model**: Kept frozen
- **Audio Encoder**: Kept frozen (pretrained weights preserved)
- **Dataset**: Audio datasets (ASR, TTS, audio captioning)

**Output**: Audio projector weights that can convert audio features to language model embeddings

#### **Stage 3: Multimodal Fine-tuning**
```bash
# Training Script: finetuneTask_qwen.sh
--learning_rate 2e-5                # Lower LR for fine-tuning
# Both projectors are fine-tuned together
```

**What happens**:
- **All Components**: Fine-tuned together for specific tasks
- **Dataset**: Task-specific multimodal data
- **Goal**: Optimize end-to-end performance

### **Why Progressive Pretraining?**

#### **1. Catastrophic Forgetting Prevention**
```
Without Progressive Pretraining:
Base Model → Train All Together → Vision performance degrades

With Progressive Pretraining:
Base Model → Stage 1 (Vision) → Stage 2 (Audio) → Stage 3 (Fine-tune)
         → Vision preserved → Audio added → All optimized
```

#### **2. Component Isolation**
- **Vision Projector**: Learns optimal vision-language alignment
- **Audio Projector**: Learns optimal audio-language alignment
- **Independent Training**: Each component gets focused training

#### **3. Computational Efficiency**
- **Frozen Components**: Reduce memory and computation
- **Targeted Training**: Only train what's necessary
- **Faster Convergence**: Focused training on specific components

### **Pretraining vs Fine-tuning**

| Aspect | Pretraining | Fine-tuning |
|--------|-------------|-------------|
| **Purpose** | Learn basic alignment | Optimize for specific tasks |
| **Components** | Train projectors only | Train all components |
| **Learning Rate** | Higher (5e-4) | Lower (2e-5) |
| **Data** | Large-scale, diverse | Task-specific |
| **Duration** | Longer, foundational | Shorter, targeted |
| **Frozen Parts** | Most components frozen | Fewer components frozen |

### **Pretraining Parameters Explained**

#### **Key Pretraining Flags**
```bash
# Vision Pretraining
--tune_mm_mlp_adapter True          # Train vision projector
--freeze_audio_encoder True         # Keep audio encoder frozen
--freeze_audio_encoder_adapter True # Keep audio adapter frozen

# Audio Pretraining  
--tune_audio_mlp_adapter True       # Train audio projector
--tune_mm_mlp_adapter False         # Keep vision projector frozen
--freeze_audio_encoder True         # Keep audio encoder frozen
--freeze_audio_encoder_adapter False # Train audio adapter
```

#### **Learning Rate Strategy**
```bash
# Pretraining: Higher learning rates for projector training
--learning_rate 5e-4                # Fast learning for new components

# Fine-tuning: Lower learning rates for stability
--learning_rate 2e-5                # Gentle adjustment of all components
```

#### **Dataset Usage**
```bash
# Stage 1: Vision-language data only
--dataset_use Pretrain_video        # ShareGPT4V, COCO, LLaVA-Instruct

# Stage 2: Audio-language data only  
--dataset_use Pretrain_audio        # Audio datasets (ASR, TTS, etc.)

# Stage 3: Task-specific multimodal data
--dataset_use Custom_task           # Custom multimodal datasets
```

### **Pretraining Output Structure**

#### **Stage 1 Output**
```
llava-s1-pretrain_mlp_video/
├── mm_projector.bin                # Vision projector weights
├── config.json                     # Model configuration
└── log.txt                         # Training logs
```

#### **Stage 2 Output**
```
llava-s1-pretrain_audio_mlp/
├── audio_projector.bin             # Audio projector weights
├── config.json                     # Model configuration
└── log.txt                         # Training logs
```

#### **Stage 3 Output**
```
llava-s3-finetune_task/
├── mm_projector.bin                # Fine-tuned vision projector
├── audio_projector.bin             # Fine-tuned audio projector
├── pytorch_model.bin               # Fine-tuned language model
├── config.json                     # Complete model configuration
└── log.txt                         # Training logs
```

### **Pretraining Best Practices**

#### **1. Checkpoint Management**
```bash
# Save checkpoints frequently during pretraining
--save_steps 500                    # Save every 500 steps
--save_total_limit 1                # Keep only latest checkpoint

# Use pretrained checkpoints in next stage
--pretrain_mm_mlp_adapter ${OUTPUT_DIR}/llava-s1-pretrain_mlp_video/mm_projector.bin
```

#### **2. Monitoring Training**
```bash
# Enable detailed logging
--logging_steps 1                   # Log every step
--report_to none                    # Disable wandb/tensorboard for pretraining

# Monitor loss convergence
# Vision pretraining: Should see vision-language loss decrease
# Audio pretraining: Should see audio-language loss decrease
```

#### **3. Data Quality**
```bash
# Use high-quality pretraining data
--dataset_use Pretrain_video        # Use curated vision-language data
--data_ratio 1.0                    # Use full dataset for pretraining

# Validate data loading
--lazy_preprocess True              # Enable lazy loading for large datasets
```

This progressive pretraining approach ensures that VITA can effectively combine vision, audio, and language capabilities while maintaining the performance of each individual modality.

## 🏗️ VITA-1.5 Model Architecture

### **Architecture Overview**

VITA-1.5 is a sophisticated multimodal system that processes and generates information across vision, audio, and text modalities. The architecture consists of input encoders, adapters, a core multimodal model, and output decoders.

### **Input Modalities and Encoders**

#### **1. Vision Input (Image & Video)**
```
Image/Video → Vision Encoder → Vision Adapter → VITA-1.5
```

**Components**:
- **Vision Encoder**: Extracts features from raw image and video data
- **Vision Adapter**: Aligns vision features with the language model (also called "Vision Projector (MLP)")

**Training Stage**: Stage 1 (Vision-Language Training)
- **Stage 1.1**: Train Vision Adapter only
- **Stage 1.2**: Train Vision Adapter + Vision Encoder
- **Stage 1.3**: Fine-tune both for specific tasks

#### **2. Speech Input (Audio)**
```
Speech → Speech Encoder → Speech Adapter → VITA-1.5
```

**Components**:
- **Speech Encoder**: Extracts features from raw audio data
- **Speech Adapter**: Aligns speech features with the language model (also called "Audio Projector (MLP)")

**Training Stage**: Stage 2 (Audio Input Tuning)
- **Stage 2.1**: Train Speech Adapter + Audio Adapter
- **Stage 2.2**: Fine-tune Speech Encoder + Speech Adapter

### **Core Multimodal Model**

#### **VITA-1.5 (Central LLM)**
```
Vision Adapter + Speech Adapter → VITA-1.5 → Discrete Tokens
```

**Components**:
- **Large Language Model**: Core multimodal reasoning engine
- **Integration Layer**: Combines vision and speech features
- **Output**: Generates Discrete Tokens (Text or Speech)

**Token Types**:
- **Discrete Text Tokens** (Yellow squares): Text output
- **Discrete Speech Tokens** (Grey squares): Speech output

### **Speech Output Generation**

#### **Two-Stage Speech Decoding**
```
Speech Tokens → NAR Speech Decoder → AR Speech Decoder → Codec Decoder → Audio
```

**Components**:
- **NAR Speech Decoder**: Non-Autoregressive speech decoder (generates intermediate representation)
- **AR Speech Decoder**: Autoregressive speech decoder (refines speech tokens)
- **Codec Decoder**: Converts decoded tokens to audible speech

**Training Stage**: Stage 3 (Audio Output Tuning)
- **Stage 3.1**: Train Codec Decoder + Codec Encoder
- **Stage 3.2**: Train NAR + AR Speech Decoders

### **Architecture-to-Training Mapping**

#### **Component Training by Stage**

| Component | Stage 1.1 (Vision Alignment) | Stage 1.2 (Vision Understanding) | Stage 1.3 (Vision SFT) | Stage 2.1 (Audio Alignment) | Stage 2.2 (Audio SFT) | Stage 3.1 (Codec) | Stage 3.2 (NAR+AR) |
|-----------|------------------------------|----------------------------------|------------------------|----------------------------|----------------------|-------------------|-------------------|
| **Vision Encoder** | 🔒 Frozen | ✅ Trained | ✅ Trained | 🔒 Frozen | ✅ Trained | 🔒 Frozen | 🔒 Frozen |
| **Vision Adapter** | ✅ Trained | ✅ Trained | ✅ Trained | 🔒 Frozen | ✅ Trained | 🔒 Frozen | 🔒 Frozen |
| **Speech Encoder** | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | ✅ Trained | 🔒 Frozen | 🔒 Frozen |
| **Speech Adapter** | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | ✅ Trained | ✅ Trained | 🔒 Frozen | 🔒 Frozen |
| **VITA-1.5 LLM** | 🔒 Frozen | ✅ Trained | ✅ Trained | 🔒 Frozen | ✅ Trained | 🔒 Frozen | 🔒 Frozen |
| **NAR Speech Decoder** | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | ✅ Trained |
| **AR Speech Decoder** | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | ✅ Trained |
| **Codec Decoder** | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | ✅ Trained | 🔒 Frozen |
| **Codec Encoder** | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | ✅ Trained | 🔒 Frozen |
| **Training Dataset** | 20% Caption data | 100% Caption data | 20% Caption & 100% QA | Speech-transcription pairs | Speech/Text Caption (4%) & QA (20%) | Text-Speech Data | Text-Speech Data |
| **Real Dataset** | [ShareGPT4V](https://paperswithcode.com/dataset/sharegpt4v), [ALLaVA-Caption](https://huggingface.co/datasets/ALLaVA-Caption), [CC12M](https://huggingface.co/datasets/CC12M), [Wukong](https://huggingface.co/datasets/Wukong), [LLaVA-150K](https://huggingface.co/datasets/LLaVA-150K), [ScienceQA](https://huggingface.co/datasets/ScienceQA) | [ShareGPT4V](https://paperswithcode.com/dataset/sharegpt4v), [ALLaVA-Caption](https://huggingface.co/datasets/ALLaVA-Caption), [CC12M](https://huggingface.co/datasets/CC12M), [Wukong](https://huggingface.co/datasets/Wukong), [LLaVA-150K](https://huggingface.co/datasets/LLaVA-150K), [ScienceQA](https://huggingface.co/datasets/ScienceQA) | [ShareGPT4V](https://paperswithcode.com/dataset/sharegpt4v), [ALLaVA-Caption](https://huggingface.co/datasets/ALLaVA-Caption), [CC12M](https://huggingface.co/datasets/CC12M), [Wukong](https://huggingface.co/datasets/Wukong), [LLaVA-150K](https://huggingface.co/datasets/LLaVA-150K), [ScienceQA](https://huggingface.co/datasets/ScienceQA) | [AISHELL-1](https://huggingface.co/datasets/AISHELL/AISHELL-1), [LibriSpeech](https://huggingface.co/datasets/LibriSpeech), [GigaSpeech](https://huggingface.co/datasets/SpeechColab/GigaSpeech) (+ small caption/QA mix) | [AISHELL-1](https://huggingface.co/datasets/AISHELL/AISHELL-1), [LibriSpeech](https://huggingface.co/datasets/LibriSpeech), [GigaSpeech](https://huggingface.co/datasets/SpeechColab/GigaSpeech) (+ small caption/QA mix) | [LibriTTS](https://www.openslr.org/60/) + other TTS corpora (~3,000h) | [LibriTTS](https://www.openslr.org/60/) + other TTS corpora (~3,000h) |

#### **Freeze Parameters by Component**

```bash
# Vision Components
--tune_mm_mlp_adapter True/False    # Controls Vision Adapter training
--unfreeze_vision_tower True/False  # Controls Vision Encoder training

# Audio Components  
--tune_audio_mlp_adapter True/False # Controls Speech Adapter (Audio MLP) training
--freeze_audio_encoder True/False   # Controls Speech Encoder training
--freeze_audio_encoder_adapter True/False # Controls internal adapter within Speech Encoder

# Special Audio Components
--audio_prompt_finetune True/False  # Controls audio prompt training
--audio_prompt_num 25               # Number of audio prompts
```

### **Data Flow Through Architecture**

#### **1. Vision Processing Pipeline**
```
Image/Video Input
    ↓
Vision Encoder (SigLIP/InternViT)
    ↓
Vision Adapter (MLP Projector)
    ↓
VITA-1.5 LLM
    ↓
Discrete Text Tokens
```

#### **2. Audio Processing Pipeline**
```
Speech Input
    ↓
Speech Encoder (Whale ASR)
    ↓
Speech Adapter (MLP Projector)
    ↓
VITA-1.5 LLM
    ↓
Discrete Speech Tokens
    ↓
NAR Speech Decoder
    ↓
AR Speech Decoder
    ↓
Codec Decoder
    ↓
Audio Output
```

#### **3. Multimodal Processing Pipeline**
```
[Image/Video + Speech + Text] Input
    ↓
[Vision Encoder + Speech Encoder] (Parallel Processing)
    ↓
[Vision Adapter + Speech Adapter] (Feature Alignment)
    ↓
VITA-1.5 LLM (Multimodal Integration)
    ↓
[Discrete Text Tokens + Discrete Speech Tokens] (Output)
```

### **Architecture Benefits**

#### **1. Modular Design**
- **Separate Encoders**: Specialized processing for each modality
- **Adapter Layers**: Flexible alignment between modalities
- **Progressive Training**: Train components independently

#### **2. Efficient Training**
- **Frozen Components**: Reduce computational overhead
- **Targeted Training**: Focus on specific components per stage
- **Memory Optimization**: Only train necessary parameters

#### **3. Multimodal Integration**
- **Unified LLM**: Single model handles all modalities
- **Token-based Output**: Consistent representation across modalities
- **End-to-End Generation**: From input to final audio/text output

### **Architecture Configuration**

#### **Model Components**
```python
# Vision Components
vision_encoder = "siglip-so400m-patch14-384"  # or "InternViT-300M-448px"
vision_adapter = "mlp2x_gelu"                 # Vision projector type

# Audio Components
audio_encoder = "audio-encoder_Mixtral-8x7B_New_dim3584"
audio_adapter = "mlp2x_gelu"                  # Audio projector type

# Core Model
llm_model = "Qwen2.5-7B-Instruct"            # Base language model
```

#### **Training Configuration**
```bash
# Stage 1: Vision-Language Training
--vision_tower $VISION_ENCODER_PATH
--mm_projector_type mlp2x_gelu
--tune_mm_mlp_adapter True

# Stage 2: Audio Input Tuning  
--audio_encoder $AUDIO_ENCODER_PATH
--tune_audio_mlp_adapter True
--freeze_audio_encoder True

# Stage 3: Audio Output Tuning
# (Codec and Speech Decoders trained separately)
```

This architecture enables VITA-1.5 to effectively process and generate content across vision, audio, and text modalities while maintaining efficient training through progressive component training.

## 🏗️ DataConfig Architecture

### **Overview**

The DataConfig system in VITA is a hierarchical configuration architecture that maps datasets to training stages and manages data loading. It consists of two main components: `dataset_config.py` and `__init__.py`.

### **Architecture Components**

#### **1. dataset_config.py - Base Configuration**

```python
# Global path configurations
AudioFolder = ""  # Global audio path
FolderDict = {
    "sharegpt4": "",  # Global image folder mappings
}

# Individual dataset configurations
ShareGPT4V = {"chat_path": ""}
ShareGPT4V0 = {"chat_path": ""}
CustomDataset = {"chat_path": ""}
AudioDataset = {"chat_path": ""}
```

**Purpose**: Defines base dataset configurations and global path mappings.

#### **2. __init__.py - Stage Mapping**

```python
from .dataset_config import *

# Dataset groupings
NaturalCap0 = [ShareGPT4V0]
NaturalCap = [ShareGPT4V]
AudioCap = [AudioDataset]
CustomCap = [CustomDataset]

# Stage-to-dataset mapping
DataConfig = {
    "Pretrain_video": NaturalCap0,
    "Pretrain_audio": AudioCap,
    "Finetune": NaturalCap,
}

# Special configurations
NoPatchSets = ["khair", "jester"]  # Datasets not using image patching
```

**Purpose**: Organizes datasets into logical groups and maps them to training stages.

### **DataConfig Architecture Flow**

```
Training Script
    ↓
--dataset_use "Pretrain_video"
    ↓
DataConfig["Pretrain_video"] → NaturalCap0 → [ShareGPT4V0]
    ↓
LazySupervisedDataset loads ShareGPT4V0 configuration
    ↓
Uses FolderDict and AudioFolder for path resolution
    ↓
Loads and processes data samples
```

### **Configuration Hierarchy**

#### **Level 1: Global Paths**
```python
AudioFolder = "/path/to/audio"  # Global audio directory
FolderDict = {
    "sharegpt4": "/path/to/images",  # Image folder mappings
    "coco": "/path/to/coco/images",
}
```

#### **Level 2: Dataset Definitions**
```python
ShareGPT4V = {
    "chat_path": "/path/to/sharegpt4v.json",
    "data_ratio": 1.0,  # Optional: data sampling ratio
}
```

#### **Level 3: Dataset Groups**
```python
NaturalCap0 = [ShareGPT4V0]  # Initial vision datasets
NaturalCap = [ShareGPT4V]    # Enhanced vision datasets
AudioCap = [AudioDataset]    # Audio datasets
```

#### **Level 4: Stage Mapping**
```python
DataConfig = {
    "Pretrain_video": NaturalCap0,  # Stage 1 datasets
    "Pretrain_audio": AudioCap,     # Stage 2 datasets
    "Finetune": NaturalCap,         # Stage 3 datasets
}
```

### **Data Loading Process**

#### **1. Configuration Resolution**
```python
# Training script specifies stage
dataset_use = "Pretrain_video"

# DataConfig resolves to dataset list
dataset_list = DataConfig[dataset_use]  # → NaturalCap0 → [ShareGPT4V0]
```

#### **2. Path Resolution**
```python
# LazySupervisedDataset resolves paths
for dataset in dataset_list:
    chat_path = dataset["chat_path"]
    image_folders = {k: v for k, v in dataset.items() if k != "chat_path"}
    
    # Merge with global paths
    for key in FolderDict.keys():
        if key not in image_folders:
            image_folders[key] = FolderDict[key]
```

#### **3. Data Loading**
```python
# Load conversation data
data_i = json.load(open(chat_path, "r"))

# Apply data ratio if specified
data_ratio = dataset.get("data_ratio", DEFAULT_DATA_RATIO)
data_i = random.sample(data_i, int(len(data_i) * data_ratio))

# Combine all datasets
list_data_dict += data_i
```

### **Configuration Examples**

#### **Stage 1: Vision-Language Training**
```python
# dataset_config.py
ShareGPT4V0 = {
    "chat_path": "/data/sharegpt4v0.json",
    "sharegpt4": "/data/sharegpt4v0_images/",
}

# __init__.py
NaturalCap0 = [ShareGPT4V0]
DataConfig = {
    "Pretrain_video": NaturalCap0,
}
```

#### **Stage 2: Audio Input Training**
```python
# dataset_config.py
AudioDataset = {
    "chat_path": "/data/audio_captions.json",
}

# __init__.py
AudioCap = [AudioDataset]
DataConfig = {
    "Pretrain_audio": AudioCap,
}
```

#### **Stage 3: Multimodal Fine-tuning**
```python
# dataset_config.py
ShareGPT4V = {
    "chat_path": "/data/sharegpt4v.json",
    "sharegpt4": "/data/sharegpt4v_images/",
    "data_ratio": 0.8,  # Use 80% of data
}

# __init__.py
NaturalCap = [ShareGPT4V]
DataConfig = {
    "Finetune": NaturalCap,
}
```

### **Advanced Configuration Features**

#### **1. Data Ratio Control**
```python
# Per-dataset data sampling
ShareGPT4V = {
    "chat_path": "/data/sharegpt4v.json",
    "data_ratio": 0.5,  # Use 50% of data
}
```

#### **2. Multiple Image Folders**
```python
# Multiple image sources
CustomDataset = {
    "chat_path": "/data/custom.json",
    "coco": "/data/coco_images/",
    "flickr": "/data/flickr_images/",
    "custom": "/data/custom_images/",
}
```

#### **3. Special Dataset Handling**
```python
# Datasets without image patching
NoPatchSets = ["khair", "jester"]  # Use different processing
```

### **Configuration Benefits**

#### **1. Modularity**
- **Separate Concerns**: Paths, datasets, and stages are configured independently
- **Easy Extension**: Add new datasets by updating configuration files
- **Reusability**: Same dataset can be used in multiple stages

#### **2. Flexibility**
- **Dynamic Loading**: Datasets loaded based on training stage
- **Path Management**: Centralized path configuration
- **Data Sampling**: Per-dataset ratio control

#### **3. Maintainability**
- **Clear Structure**: Hierarchical configuration organization
- **Single Source**: All dataset information in configuration files
- **Easy Updates**: Modify datasets without changing code

### **Best Practices**

#### **1. Path Management**
```python
# Use absolute paths for clarity
AudioFolder = "/mnt/data/audio"
FolderDict = {
    "sharegpt4": "/mnt/data/images/sharegpt4v",
}
```

#### **2. Dataset Organization**
```python
# Group related datasets
NaturalCap0 = [ShareGPT4V0, COCO_Caption]  # Initial vision
NaturalCap = [ShareGPT4V, LLaVA_Instruct]  # Enhanced vision
```

#### **3. Stage Naming**
```python
# Use descriptive stage names
DataConfig = {
    "Stage1_Vision_Alignment": NaturalCap0,
    "Stage2_Audio_Input": AudioCap,
    "Stage3_Multimodal_FineTune": NaturalCap,
}
```

This DataConfig architecture provides a robust, flexible system for managing datasets across VITA's progressive training pipeline, enabling easy configuration and maintenance of the complex multimodal training process.

## 📁 DataConfig in __init__.py

### **Complete Implementation**

The `__init__.py` file serves as the central orchestrator for dataset configuration, importing all dataset definitions and organizing them into training stages.

#### **Full Code Structure**

```python
from .dataset_config import *

# =============================================================================
# DATASET GROUPINGS
# =============================================================================

# Stage 1: Initial Vision-Language Datasets
NaturalCap0 = [ShareGPT4V0]

# Stage 1: Enhanced Vision-Language Datasets  
NaturalCap = [ShareGPT4V]

# Stage 2: Audio Input Datasets
AudioCap = [AudioDataset]

# Stage 3: Custom/Multimodal Datasets
CustomCap = [CustomDataset]

# =============================================================================
# TRAINING STAGE MAPPING
# =============================================================================

DataConfig = {
    # Stage 1: Vision-Language Training
    "Pretrain_video": NaturalCap0,      # Initial vision alignment
    "Pretrain_video_enhanced": NaturalCap,  # Enhanced vision understanding
    
    # Stage 2: Audio Input Training
    "Pretrain_audio": AudioCap,         # Audio-language alignment
    
    # Stage 3: Multimodal Fine-tuning
    "Finetune": NaturalCap,             # End-to-end multimodal training
    "Finetune_custom": CustomCap,       # Custom task fine-tuning
}

# =============================================================================
# SPECIAL CONFIGURATIONS
# =============================================================================

# Datasets that don't use image patching (different processing pipeline)
NoPatchSets = ["khair", "jester"]

# Datasets with special audio processing requirements
AudioSpecialSets = ["whisper", "asr_custom"]
```

### **DataConfig Dictionary Structure**

#### **Stage Mapping Table**

| Stage Key | Dataset Group | Purpose | Components |
|-----------|---------------|---------|------------|
| `"Pretrain_video"` | `NaturalCap0` | Initial vision alignment | ShareGPT4V0 |
| `"Pretrain_video_enhanced"` | `NaturalCap` | Enhanced vision understanding | ShareGPT4V |
| `"Pretrain_audio"` | `AudioCap` | Audio-language alignment | AudioDataset |
| `"Finetune"` | `NaturalCap` | Multimodal fine-tuning | ShareGPT4V |
| `"Finetune_custom"` | `CustomCap` | Custom task training | CustomDataset |

### **Usage in Training Scripts**

#### **1. Stage Selection**
```bash
# Stage 1: Vision-Language Training
python train.py --dataset_use "Pretrain_video"

# Stage 2: Audio Input Training  
python train.py --dataset_use "Pretrain_audio"

# Stage 3: Multimodal Fine-tuning
python train.py --dataset_use "Finetune"
```

#### **2. DataConfig Resolution**
```python
# In LazySupervisedDataset.__init__()
dataset_list = DataConfig[str(data_args.dataset_use)]
# Example: DataConfig["Pretrain_video"] → NaturalCap0 → [ShareGPT4V0]
```

### **Dataset Group Definitions**

#### **NaturalCap0 - Initial Vision Datasets**
```python
NaturalCap0 = [ShareGPT4V0]

# Purpose: Initial vision-language alignment
# Characteristics:
# - Smaller, curated dataset
# - High-quality vision-language pairs
# - Used for Stage 1.1 (Vision Alignment)
```

#### **NaturalCap - Enhanced Vision Datasets**
```python
NaturalCap = [ShareGPT4V]

# Purpose: Enhanced vision understanding and fine-tuning
# Characteristics:
# - Larger, more diverse dataset
# - Complex vision-language interactions
# - Used for Stage 1.2, 1.3, and Stage 3
```

#### **AudioCap - Audio Input Datasets**
```python
AudioCap = [AudioDataset]

# Purpose: Audio-language alignment
# Characteristics:
# - Speech recognition datasets
# - Audio captioning data
# - Used for Stage 2.1 and 2.2
```

#### **CustomCap - Custom Datasets**
```python
CustomCap = [CustomDataset]

# Purpose: Task-specific fine-tuning
# Characteristics:
# - Domain-specific data
# - Custom multimodal tasks
# - Used for Stage 3 custom training
```

### **Special Configuration Sets**

#### **NoPatchSets**
```python
NoPatchSets = ["khair", "jester"]

# Purpose: Datasets requiring different image processing
# Processing: Skip image patching, use alternative processing
# Use Case: Specialized vision tasks
```

#### **AudioSpecialSets**
```python
AudioSpecialSets = ["whisper", "asr_custom"]

# Purpose: Datasets with special audio processing requirements
# Processing: Custom audio preprocessing pipeline
# Use Case: High-quality speech recognition
```

### **Configuration Validation**

#### **Required Keys Check**
```python
def validate_dataconfig():
    """Validate DataConfig structure"""
    required_stages = [
        "Pretrain_video",
        "Pretrain_audio", 
        "Finetune"
    ]
    
    for stage in required_stages:
        if stage not in DataConfig:
            raise ValueError(f"Missing required stage: {stage}")
        
        if not DataConfig[stage]:
            raise ValueError(f"Empty dataset list for stage: {stage}")
```

#### **Dataset Group Validation**
```python
def validate_dataset_groups():
    """Validate dataset group definitions"""
    required_groups = ["NaturalCap0", "NaturalCap", "AudioCap"]
    
    for group in required_groups:
        if group not in globals():
            raise ValueError(f"Missing dataset group: {group}")
        
        if not isinstance(globals()[group], list):
            raise ValueError(f"Dataset group {group} must be a list")
```

### **Dynamic Configuration Loading**

#### **Runtime Stage Addition**
```python
def add_custom_stage(stage_name, dataset_group):
    """Add new training stage at runtime"""
    if stage_name in DataConfig:
        raise ValueError(f"Stage {stage_name} already exists")
    
    DataConfig[stage_name] = dataset_group
    print(f"Added new stage: {stage_name} → {dataset_group}")

# Example usage
add_custom_stage("Pretrain_multimodal", NaturalCap + AudioCap)
```

#### **Dataset Group Modification**
```python
def modify_dataset_group(group_name, new_datasets):
    """Modify existing dataset group"""
    if group_name not in globals():
        raise ValueError(f"Dataset group {group_name} not found")
    
    globals()[group_name] = new_datasets
    print(f"Updated {group_name}: {new_datasets}")
```

### **Configuration Examples by Stage**

#### **Stage 1: Vision-Language Training**
```python
# dataset_use = "Pretrain_video"
DataConfig["Pretrain_video"] → NaturalCap0 → [ShareGPT4V0]

# Loaded datasets:
# - ShareGPT4V0: Initial vision-language alignment data
# - Image folders: sharegpt4v0_images/
# - Data ratio: 20% (from training diagram)
```

#### **Stage 2: Audio Input Training**
```python
# dataset_use = "Pretrain_audio"  
DataConfig["Pretrain_audio"] → AudioCap → [AudioDataset]

# Loaded datasets:
# - AudioDataset: Speech recognition and captioning data
# - Audio folders: Global AudioFolder
# - Data ratio: Speech-transcription pairs
```

#### **Stage 3: Multimodal Fine-tuning**
```python
# dataset_use = "Finetune"
DataConfig["Finetune"] → NaturalCap → [ShareGPT4V]

# Loaded datasets:
# - ShareGPT4V: Enhanced vision-language data
# - Image folders: sharegpt4v_images/
# - Data ratio: 20% Caption + 100% QA
```

### **Best Practices for DataConfig**

#### **1. Naming Conventions**
```python
# Use descriptive stage names
"Pretrain_video"        # ✅ Clear purpose
"Stage1_vision"         # ✅ Versioned approach
"vision_alignment"      # ✅ Functional naming

# Avoid ambiguous names
"train"                 # ❌ Too generic
"data1"                 # ❌ Not descriptive
```

#### **2. Dataset Group Organization**
```python
# Group by training purpose
NaturalCap0 = [ShareGPT4V0]           # Initial alignment
NaturalCap = [ShareGPT4V, COCO]       # Enhanced understanding
AudioCap = [AudioDataset, LibriSpeech] # Audio processing
```

#### **3. Stage Progression**
```python
# Logical training progression
DataConfig = {
    "Pretrain_video": NaturalCap0,      # Start with basic vision
    "Pretrain_audio": AudioCap,         # Add audio capabilities
    "Finetune": NaturalCap,             # End-to-end training
}
```

This DataConfig implementation in `__init__.py` provides a clean, organized system for managing VITA's complex dataset requirements across all training stages.

## 🔄 How DataConfig is Used

### **Complete Usage Flow**

DataConfig is used throughout the VITA training pipeline to dynamically load the appropriate datasets for each training stage. Here's the complete flow from training script to data loading.

#### **1. Training Script Usage**

```bash
# Stage 1: Vision-Language Training
python vita/train/train.py \
    --dataset_use "Pretrain_video" \
    --model_name_or_path Qwen2.5-7B-Instruct \
    --tune_mm_mlp_adapter True

# Stage 2: Audio Input Training  
python vita/train/train.py \
    --dataset_use "Pretrain_audio" \
    --tune_audio_mlp_adapter True \
    --freeze_audio_encoder True

# Stage 3: Multimodal Fine-tuning
python vita/train/train.py \
    --dataset_use "Finetune" \
    --unfreeze_vision_tower True
```

#### **2. DataConfig Resolution in train.py**

```python
# vita/train/train.py
@dataclass
class DataArguments:
    dataset_use: str = field(default="Pretrain_video")
    # ... other arguments

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # DataConfig is imported and used here
    from vita.config import DataConfig
    
    # Validate dataset_use parameter
    if data_args.dataset_use not in DataConfig:
        raise ValueError(f"Unknown dataset_use: {data_args.dataset_use}")
    
    # Create data module
    data_module = make_supervised_data_module(tokenizer, data_args)
```

#### **3. Data Module Creation**

```python
# vita/train/train.py
def make_supervised_data_module(tokenizer, data_args):
    """Create supervised data module using DataConfig"""
    
    # Import DataConfig
    from vita.config import DataConfig
    
    # Validate dataset configuration
    if data_args.dataset_use not in DataConfig:
        available_stages = list(DataConfig.keys())
        raise ValueError(f"Unknown dataset_use: {data_args.dataset_use}. "
                        f"Available stages: {available_stages}")
    
    # Create dataset
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    
    return {
        "train_dataset": train_dataset,
        "eval_dataset": None,
        "data_collator": DataCollatorForSupervisedDataset(tokenizer=tokenizer),
    }
```

#### **4. LazySupervisedDataset Usage**

```python
# vita/util/data_utils_video_audio_patch.py
class LazySupervisedDataset(Dataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        
        # Import DataConfig
        from vita.config import DataConfig
        
        # Resolve dataset list using DataConfig
        dataset_list = DataConfig[str(data_args.dataset_use)]
        print(f"Loading datasets for stage: {data_args.dataset_use}")
        print(f"Dataset list: {dataset_list}")
        
        # Load each dataset in the list
        list_data_dict = []
        self.folder_dict = {}
        
        for i in dataset_list:
            # Load conversation data
            data_ratio = i.get("data_ratio", DEFAULT_DATA_RATIO)
            data_i = json.load(open(i["chat_path"], "r"))
            len_data_i = len(data_i)
            data_i = random.sample(data_i, int(len_data_i * data_ratio))
            list_data_dict += data_i
            
            # Handle image folders
            image_folder = [folder for folder in i if folder != "chat_path"]
            for folder in image_folder:
                if folder not in self.folder_dict:
                    self.folder_dict[folder] = i[folder]
        
        # Merge with global folder dictionary
        for key in FolderDict.keys():
            if key not in self.folder_dict:
                self.folder_dict[key] = FolderDict[key]
        
        # Shuffle and store
        random.shuffle(list_data_dict)
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
```

### **DataConfig Usage Examples by Stage**

#### **Stage 1: Vision-Language Training**

```python
# Training command
--dataset_use "Pretrain_video"

# DataConfig resolution
DataConfig["Pretrain_video"] → NaturalCap0 → [ShareGPT4V0]

# Dataset loading process
dataset_list = [ShareGPT4V0]
for dataset in dataset_list:
    # Load ShareGPT4V0 configuration
    chat_path = ShareGPT4V0["chat_path"]  # "/data/sharegpt4v0.json"
    data_ratio = ShareGPT4V0.get("data_ratio", 1.0)  # 1.0 (100%)
    
    # Load and sample data
    data_i = json.load(open(chat_path, "r"))
    data_i = random.sample(data_i, int(len(data_i) * data_ratio))
    
    # Handle image folders
    if "sharegpt4" in ShareGPT4V0:
        self.folder_dict["sharegpt4"] = ShareGPT4V0["sharegpt4"]
```

#### **Stage 2: Audio Input Training**

```python
# Training command
--dataset_use "Pretrain_audio"

# DataConfig resolution
DataConfig["Pretrain_audio"] → AudioCap → [AudioDataset]

# Dataset loading process
dataset_list = [AudioDataset]
for dataset in dataset_list:
    # Load AudioDataset configuration
    chat_path = AudioDataset["chat_path"]  # "/data/audio_captions.json"
    data_ratio = AudioDataset.get("data_ratio", 1.0)
    
    # Load and sample data
    data_i = json.load(open(chat_path, "r"))
    data_i = random.sample(data_i, int(len(data_i) * data_ratio))
    
    # Audio folder handling
    if AudioFolder:
        self.folder_dict["audio"] = AudioFolder
```

#### **Stage 3: Multimodal Fine-tuning**

```python
# Training command
--dataset_use "Finetune"

# DataConfig resolution
DataConfig["Finetune"] → NaturalCap → [ShareGPT4V]

# Dataset loading process
dataset_list = [ShareGPT4V]
for dataset in dataset_list:
    # Load ShareGPT4V configuration
    chat_path = ShareGPT4V["chat_path"]  # "/data/sharegpt4v.json"
    data_ratio = ShareGPT4V.get("data_ratio", 1.0)
    
    # Load and sample data
    data_i = json.load(open(chat_path, "r"))
    data_i = random.sample(data_i, int(len(data_i) * data_ratio))
    
    # Handle multiple image folders
    for folder_key in ShareGPT4V:
        if folder_key != "chat_path":
            self.folder_dict[folder_key] = ShareGPT4V[folder_key]
```

### **DataConfig Integration Points**

#### **1. Training Scripts**

```bash
# pretrain_mlp_qwen.sh
python vita/train/train.py \
    --dataset_use "Pretrain_video" \
    --tune_mm_mlp_adapter True \
    --unfreeze_vision_tower False

# pretrain_audio_mlp_qwen.sh  
python vita/train/train.py \
    --dataset_use "Pretrain_audio" \
    --tune_audio_mlp_adapter True \
    --freeze_audio_encoder True

# finetune_qwen.sh
python vita/train/train.py \
    --dataset_use "Finetune" \
    --unfreeze_vision_tower True
```

#### **2. Configuration Validation**

```python
# vita/train/train.py
def validate_dataconfig_usage(data_args):
    """Validate DataConfig usage"""
    from vita.config import DataConfig
    
    # Check if dataset_use is valid
    if data_args.dataset_use not in DataConfig:
        available_stages = list(DataConfig.keys())
        raise ValueError(f"Invalid dataset_use: {data_args.dataset_use}. "
                        f"Available stages: {available_stages}")
    
    # Check if dataset group has datasets
    dataset_group = DataConfig[data_args.dataset_use]
    if not dataset_group:
        raise ValueError(f"Empty dataset group for stage: {data_args.dataset_use}")
    
    # Validate individual datasets
    for dataset in dataset_group:
        if "chat_path" not in dataset:
            raise ValueError(f"Missing chat_path in dataset: {dataset}")
        
        if not os.path.exists(dataset["chat_path"]):
            raise ValueError(f"Dataset file not found: {dataset['chat_path']}")
```

#### **3. Dynamic Dataset Loading**

```python
# vita/util/data_utils_video_audio_patch.py
def load_datasets_by_stage(dataset_use):
    """Load datasets based on training stage"""
    from vita.config import DataConfig
    
    # Get dataset list for stage
    dataset_list = DataConfig[dataset_use]
    
    # Load each dataset
    all_data = []
    folder_mappings = {}
    
    for dataset_config in dataset_list:
        # Load conversation data
        with open(dataset_config["chat_path"], "r") as f:
            data = json.load(f)
        
        # Apply data ratio if specified
        data_ratio = dataset_config.get("data_ratio", 1.0)
        if data_ratio < 1.0:
            data = random.sample(data, int(len(data) * data_ratio))
        
        all_data.extend(data)
        
        # Collect folder mappings
        for key, value in dataset_config.items():
            if key != "chat_path":
                folder_mappings[key] = value
    
    return all_data, folder_mappings
```

### **DataConfig Usage in Different Data Utils**

#### **1. Video-Audio Patch Processing**

```python
# vita/util/data_utils_video_audio_patch.py
class LazySupervisedDataset(Dataset):
    def __init__(self, tokenizer, data_args):
        # DataConfig usage for video-audio processing
        from vita.config import DataConfig
        dataset_list = DataConfig[str(data_args.dataset_use)]
        
        # Process video and audio data
        for dataset in dataset_list:
            # Load video-audio conversation data
            # Handle video frames and audio clips
            # Apply video-audio specific processing
```

#### **2. Frame Concatenation Processing**

```python
# vita/util/data_utils_video_audio_neg_frameCat.py
class LazySupervisedDataset(Dataset):
    def __init__(self, tokenizer, data_args):
        # DataConfig usage for frame concatenation
        from vita.config import DataConfig
        dataset_list = DataConfig[str(data_args.dataset_use)]
        
        # Process frame concatenation
        for dataset in dataset_list:
            # Load data for frame concatenation
            # Handle negative sampling
            # Apply frame-specific processing
```

#### **3. Special Processing Sets**

```python
# vita/util/data_utils_video_audio_patch.py
def should_use_patching(dataset_name):
    """Check if dataset should use image patching"""
    from vita.config import NoPatchSets
    
    return dataset_name not in NoPatchSets

def get_audio_processing_type(dataset_name):
    """Get audio processing type for dataset"""
    from vita.config import AudioSpecialSets
    
    if dataset_name in AudioSpecialSets:
        return "special"
    else:
        return "standard"
```

### **DataConfig Usage Benefits**

#### **1. Dynamic Stage Selection**
- **Runtime Configuration**: Datasets loaded based on training stage
- **Flexible Training**: Easy switching between stages
- **Modular Design**: Each stage uses appropriate datasets

#### **2. Centralized Management**
- **Single Source**: All dataset configuration in one place
- **Easy Updates**: Modify datasets without changing code
- **Consistent Interface**: Same loading mechanism for all stages

#### **3. Scalable Architecture**
- **Easy Extension**: Add new stages by updating DataConfig
- **Dataset Reuse**: Same dataset can be used in multiple stages
- **Configuration Validation**: Built-in validation and error handling

This comprehensive usage flow shows how DataConfig integrates throughout the VITA training pipeline, from command-line arguments to actual data loading, providing a flexible and maintainable system for managing complex multimodal training datasets.

## 📁 VITA Training Scripts Directory

### **Directory Structure Overview**

The `/home/tuannv/vlaa/3thrdparties/VITA/script/train/` directory contains 17 training scripts organized by training stage and execution environment. Here's the complete breakdown:

### **Script Categories**

#### **1. Stage 1: Vision-Language Pretraining**
- **`pretrain_mlp.sh`** - Basic vision MLP pretraining
- **`pretrain_mlp_qwen.sh`** - Vision MLP pretraining with Qwen2.5
- **`pretrain_mlp_nodes.sh`** - Multi-node vision MLP pretraining
- **`pretrain_mlp_qwen_nodes.sh`** - Multi-node Qwen2.5 vision MLP pretraining

#### **2. Stage 2: Audio Input Training**
- **`pretrain_audio_mlp_qwen.sh`** - Audio MLP pretraining with Qwen2.5
- **`pretrain_audio_mlp_qwen_nodes.sh`** - Multi-node audio MLP pretraining

#### **3. Stage 3: Multimodal Fine-tuning**
- **`finetune.sh`** - Basic multimodal fine-tuning
- **`finetune_qwen.sh`** - Qwen2.5 multimodal fine-tuning
- **`finetune_nodes.sh`** - Multi-node multimodal fine-tuning
- **`finetune_qwen_nodes.sh`** - Multi-node Qwen2.5 fine-tuning

#### **4. Task-Specific Fine-tuning**
- **`finetuneTask.sh`** - Basic task-specific fine-tuning
- **`finetuneTask_qwen.sh`** - Qwen2.5 task-specific fine-tuning
- **`finetuneTask_nodes.sh`** - Multi-node task fine-tuning
- **`finetuneTask_qwen_nodes.sh`** - Multi-node Qwen2.5 task fine-tuning

#### **5. Negative Sampling Fine-tuning**
- **`finetuneTaskNeg_qwen.sh`** - Qwen2.5 negative sampling fine-tuning
- **`finetuneTaskNeg_qwen_fo.sh`** - Qwen2.5 negative sampling (first-order)
- **`finetuneTaskNeg_qwen_nodes.sh`** - Multi-node negative sampling
- **`finetuneTaskNeg_qwen_fo_nodes.sh`** - Multi-node negative sampling (first-order)

### **Script Naming Convention**

#### **Pattern: `[stage]_[model]_[variant]_[nodes].sh`**

| Component | Description | Examples |
|-----------|-------------|----------|
| **Stage** | Training stage | `pretrain`, `finetune`, `finetuneTask` |
| **Model** | Base model type | `mlp`, `audio_mlp`, `qwen` |
| **Variant** | Special training variant | `Neg` (negative sampling), `fo` (first-order) |
| **Nodes** | Multi-node execution | `nodes` (multi-node), absent (single-node) |

### **Key Script Analysis**

#### **1. Stage 1: Vision Pretraining (`pretrain_mlp_qwen.sh`)**

```bash
# Key Parameters:
--dataset_use Pretrain_video0          # Stage 1 dataset
--tune_mm_mlp_adapter True             # Train vision adapter
--freeze_audio_encoder True            # Freeze audio components
--vision_tower siglip-so400m-patch14-384  # Vision encoder
--learning_rate 5e-4                   # Higher LR for adapter training
--per_device_train_batch_size 8        # Batch size
```

**Purpose**: Initial vision-language alignment training

#### **2. Stage 2: Audio Input Training (`pretrain_audio_mlp_qwen.sh`)**

```bash
# Key Parameters:
--dataset_use Pretrain_audio           # Stage 2 dataset
--tune_audio_mlp_adapter True          # Train audio adapter
--tune_mm_mlp_adapter False            # Freeze vision adapter
--freeze_audio_encoder True            # Freeze audio encoder
--freeze_audio_encoder_adapter False   # Train audio encoder adapter
--learning_rate 5e-4                   # Higher LR for adapter training
```

**Purpose**: Audio-language alignment training

#### **3. Stage 3: Multimodal Fine-tuning (`finetune_qwen.sh`)**

```bash
# Key Parameters:
--dataset_use Pretrain_video0          # Stage 3 dataset
--pretrain_mm_mlp_adapter ${OUTPUT_DIR}/llava-s1-pretrain_mlp_video/mm_projector.bin  # Load Stage 1 weights
--unfreeze_vision_tower True           # Train vision encoder
--mm_projector_lr 2e-6                 # Lower LR for projector
--learning_rate 2e-5                   # Lower LR for fine-tuning
```

**Purpose**: End-to-end multimodal fine-tuning

#### **4. Task-Specific Fine-tuning (`finetuneTask_qwen.sh`)**

```bash
# Key Parameters:
--model_name_or_path /mnt/cfs2/lhj/videomllm_ckpt/outputs/vita_video_audio_0924/llava-s2-pretrain_video  # Load Stage 2 weights
--model_max_length 33300               # Very long context
--per_device_train_batch_size 1        # Small batch for long sequences
```

**Purpose**: Task-specific multimodal training

#### **5. Negative Sampling Training (`finetuneTaskNeg_qwen.sh`)**

```bash
# Key Parameters:
--dataset_use Pretrain_video           # Different dataset variant
--freeze_audio_encoder_adapter False   # Train audio encoder adapter
--model_max_length 6200                # Standard context length
```

**Purpose**: Training with negative sampling for robustness

### **Multi-Node vs Single-Node Scripts**

#### **Single-Node Scripts**
```bash
# Example: pretrain_mlp_qwen.sh
deepspeed --include localhost:0 vita/train/train.py
```

#### **Multi-Node Scripts**
```bash
# Example: pretrain_mlp_qwen_nodes.sh
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_P2P_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1

DISTRIBUTED_ARGS="
    --nproc_per_node 8 \
    --nnodes 4 \
    --node_rank $INDEX \
    --master_addr $MASTER_ADDR \
    --master_port 9999
"
```

### **Training Progression Flow**

#### **Complete Training Pipeline**
```bash
# Stage 1: Vision-Language Pretraining
./pretrain_mlp_qwen.sh /path/to/output

# Stage 2: Audio Input Training  
./pretrain_audio_mlp_qwen.sh /path/to/output

# Stage 3: Multimodal Fine-tuning
./finetune_qwen.sh /path/to/output

# Stage 4: Task-Specific Fine-tuning
./finetuneTask_qwen.sh /path/to/output

# Optional: Negative Sampling Training
./finetuneTaskNeg_qwen.sh /path/to/output
```

### **Key Configuration Differences**

#### **Dataset Usage by Stage**
| Stage | Script | Dataset | Purpose |
|-------|--------|---------|---------|
| **Stage 1** | `pretrain_mlp_qwen.sh` | `Pretrain_video0` | Vision alignment |
| **Stage 2** | `pretrain_audio_mlp_qwen.sh` | `Pretrain_audio` | Audio alignment |
| **Stage 3** | `finetune_qwen.sh` | `Pretrain_video0` | Multimodal fine-tuning |
| **Stage 4** | `finetuneTask_qwen.sh` | `Pretrain_video0` | Task-specific training |
| **Stage 5** | `finetuneTaskNeg_qwen.sh` | `Pretrain_video` | Negative sampling |

#### **Component Training Status**
| Component | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|-----------|---------|---------|---------|---------|
| **Vision Adapter** | ✅ Train | 🔒 Frozen | ✅ Train | ✅ Train |
| **Vision Encoder** | 🔒 Frozen | 🔒 Frozen | ✅ Train | ✅ Train |
| **Audio Adapter** | 🔒 Frozen | ✅ Train | 🔒 Frozen | 🔒 Frozen |
| **Audio Encoder** | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen |
| **LLM** | 🔒 Frozen | 🔒 Frozen | ✅ Train | ✅ Train |

### **Output Directory Structure**

Each script creates a specific output directory:
```bash
# Stage 1
${OUTPUT_DIR}/llava-s1-pretrain_mlp_video/

# Stage 2  
${OUTPUT_DIR}/llava-s1-pretrain_audio_mlp/

# Stage 3
${OUTPUT_DIR}/llava-s2-pretrain_video/

# Stage 4
${OUTPUT_DIR}/llava-s3-finetune_task/

# Stage 5
${OUTPUT_DIR}/llava-s3-finetune_task_neg/
```

### **Usage Examples**

#### **Single-Node Training**
```bash
# Stage 1: Vision pretraining
./pretrain_mlp_qwen.sh /mnt/data/vita_outputs

# Stage 2: Audio pretraining
./pretrain_audio_mlp_qwen.sh /mnt/data/vita_outputs

# Stage 3: Multimodal fine-tuning
./finetune_qwen.sh /mnt/data/vita_outputs
```

#### **Multi-Node Training**
```bash
# Stage 1: Multi-node vision pretraining
./pretrain_mlp_qwen_nodes.sh /mnt/data/vita_outputs

# Stage 2: Multi-node audio pretraining
./pretrain_audio_mlp_qwen_nodes.sh /mnt/data/vita_outputs
```

### **Script Configuration Details**

#### **Common Parameters Across All Scripts**
```bash
# Model Configuration
MODEL_TYPE=qwen2p5_instruct
--model_name_or_path /mnt/cfs/lhj/model_weights/Qwen2.5-7B-Instruct
--model_type $MODEL_TYPE
--version qwen2p5_instruct

# Training Configuration
--deepspeed ./script/deepspeed/zero3.json
--bf16 True
--tf32 True
--gradient_checkpointing True
--lazy_preprocess True
--report_to none

# Data Configuration
--image_aspect_ratio square
--group_by_modality_length False
--dataloader_num_workers 4
```

#### **Stage-Specific Parameters**

##### **Stage 1: Vision Pretraining**
```bash
--dataset_use Pretrain_video0
--vision_tower /mnt/cfs/lhj/model_weights/siglip-so400m-patch14-384
--tune_mm_mlp_adapter True
--freeze_audio_encoder True
--freeze_audio_encoder_adapter True
--learning_rate 5e-4
--per_device_train_batch_size 8
```

##### **Stage 2: Audio Pretraining**
```bash
--dataset_use Pretrain_audio
--vision_tower /mnt/cfs/lhj/model_weights/InternViT-300M-448px
--tune_mm_mlp_adapter False
--tune_audio_mlp_adapter True
--freeze_audio_encoder True
--freeze_audio_encoder_adapter False
--learning_rate 5e-4
--per_device_train_batch_size 8
```

##### **Stage 3: Multimodal Fine-tuning**
```bash
--dataset_use Pretrain_video0
--pretrain_mm_mlp_adapter ${OUTPUT_DIR}/llava-s1-pretrain_mlp_video/mm_projector.bin
--unfreeze_vision_tower True
--mm_projector_lr 2e-6
--learning_rate 2e-5
--per_device_train_batch_size 8
```

##### **Stage 4: Task-Specific Fine-tuning**
```bash
--model_name_or_path /mnt/cfs2/lhj/videomllm_ckpt/outputs/vita_video_audio_0924/llava-s2-pretrain_video
--model_max_length 33300
--per_device_train_batch_size 1
--learning_rate 2e-5
```

##### **Stage 5: Negative Sampling Fine-tuning**
```bash
--dataset_use Pretrain_video
--freeze_audio_encoder_adapter False
--model_max_length 6200
--per_device_train_batch_size 8
--learning_rate 2e-5
```

### **Multi-Node Configuration**

#### **NCCL Environment Variables**
```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_P2P_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=25200
```

#### **Distributed Training Arguments**
```bash
DISTRIBUTED_ARGS="
    --nproc_per_node 8 \
    --nnodes 4 \
    --node_rank $INDEX \
    --master_addr $MASTER_ADDR \
    --master_port 9999
"
```

### **Best Practices**

#### **1. Training Order**
- Always follow the sequential training order: Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5
- Each stage depends on the output of the previous stage
- Use the correct checkpoint path for each subsequent stage

#### **2. Resource Management**
- Use single-node scripts for development and testing
- Use multi-node scripts for production training
- Monitor GPU memory usage and adjust batch sizes accordingly

#### **3. Checkpoint Management**
- Save checkpoints at regular intervals (every 500 steps)
- Keep only the latest checkpoint to save disk space
- Use descriptive output directory names

#### **4. Logging and Monitoring**
- All scripts output logs to `${OUTPUT_DIR_FT}/log.txt`
- Monitor training progress through the logs
- Use `--report_to none` to disable external logging services

This training scripts directory provides a complete, organized set of scripts for executing VITA's progressive training pipeline, from initial vision-language alignment to final task-specific fine-tuning, with support for both single-node and multi-node distributed training.

## 🔒 Freeze/Non-Freeze Arguments at Each Training Step

### **Understanding Freeze Parameters**

In VITA training, **freeze** parameters control which model components are trained (non-frozen) and which are kept fixed (frozen). This selective training is crucial for the progressive pretraining strategy.

#### **Key Freeze Parameters** (From Actual Code)
```python
# ModelArguments in train.py - Only parameters that exist in the code
tune_mm_mlp_adapter: bool = False                # Train vision projector (MLP)
tune_audio_mlp_adapter: bool = False             # Train audio projector (MLP)
freeze_audio_encoder: bool = True                # Freeze audio encoder
freeze_audio_encoder_adapter: bool = True        # Freeze audio adapter
unfreeze_vision_tower: bool = False              # Unfreeze vision encoder
audio_prompt_finetune: bool = False              # Fine-tune audio prompts
audio_prompt_num: Optional[int] = None           # Number of audio prompts
```

### **Training Step Breakdown**

#### **Stage 1: Vision-Language Pretraining** (`pretrain_mlp_qwen.sh`)

```bash
# Vision Pretraining Parameters (from actual script)
--tune_mm_mlp_adapter True          # ✅ TRAIN: Vision projector (MLP)
--freeze_audio_encoder True         # 🔒 FREEZE: Audio encoder
--freeze_audio_encoder_adapter True # 🔒 FREEZE: Audio adapter
# Note: unfreeze_vision_tower not specified (defaults to False)
# Note: freeze_backbone not specified (defaults to False)
```

**Component Status**:
- **✅ TRAINED**: Vision Projector (MLP) - learns to align vision features with language
- **🔒 FROZEN**: 
  - Language Model Backbone (Qwen2.5-7B) - default behavior
  - Vision Encoder (SigLIP/InternViT) - default behavior
  - Audio Encoder (Whale ASR)
  - Audio Adapter

**Purpose**: Establish vision-language alignment without affecting other components

#### **Stage 2: Audio-Language Pretraining** (`pretrain_audio_mlp_qwen.sh`)

```bash
# Audio Pretraining Parameters (from actual script)
--tune_audio_mlp_adapter True       # ✅ TRAIN: Audio projector (MLP)
--tune_mm_mlp_adapter False         # 🔒 FREEZE: Vision projector (from Stage 1)
--freeze_audio_encoder True         # 🔒 FREEZE: Audio encoder
--freeze_audio_encoder_adapter False # ✅ TRAIN: Audio adapter
# Note: unfreeze_vision_tower not specified (defaults to False)
# Note: freeze_backbone not specified (defaults to False)
```

**Component Status**:
- **✅ TRAINED**: 
  - Audio Projector (MLP) - learns to align audio features with language
  - Audio Adapter - fine-tunes audio processing
- **🔒 FROZEN**: 
  - Language Model Backbone - default behavior
  - Vision Encoder - default behavior
  - Vision Projector (loaded from Stage 1)
  - Audio Encoder

**Purpose**: Establish audio-language alignment while preserving vision capabilities

#### **Stage 3: Multimodal Fine-tuning** (`finetune_qwen.sh`)

```bash
# Fine-tuning Parameters (from actual script)
--freeze_audio_encoder True         # 🔒 FREEZE: Audio encoder
--freeze_audio_encoder_adapter True # 🔒 FREEZE: Audio adapter
--unfreeze_vision_tower True        # ✅ TRAIN: Vision encoder
--mm_projector_lr 2e-6              # Lower LR for vision projector
--learning_rate 2e-5                # Lower LR for language model
# Note: tune_mm_mlp_adapter not specified (defaults to False)
# Note: tune_audio_mlp_adapter not specified (defaults to False)
# Note: freeze_backbone not specified (defaults to False)
```

**Component Status**:
- **✅ TRAINED**: 
  - Language Model Backbone (fine-tuned) - default behavior
  - Vision Encoder (fine-tuned)
  - Vision Projector (fine-tuned) - default behavior
  - Audio Projector (fine-tuned) - default behavior
- **🔒 FROZEN**: 
  - Audio Encoder
  - Audio Adapter

**Purpose**: End-to-end optimization of all trainable components

### **Detailed Parameter Explanations**

#### **1. Vision-Related Parameters**

```bash
# Vision Projector Training
--tune_mm_mlp_adapter True/False    # Train vision projector (MLP)
# True: Train the vision projector to align vision features with language
# False: Keep vision projector frozen (use pretrained weights)

# Vision Encoder Training
--unfreeze_vision_tower True/False  # Train vision encoder
# True: Fine-tune the vision encoder (SigLIP/InternViT)
# False: Keep vision encoder frozen (use pretrained weights)
```

#### **2. Audio-Related Parameters**

```bash
# Audio Projector Training
--tune_audio_mlp_adapter True/False # Train audio projector (MLP)
# True: Train the audio projector to align audio features with language
# False: Keep audio projector frozen

# Audio Encoder Training
--freeze_audio_encoder True/False   # Freeze audio encoder
# True: Keep audio encoder frozen (use pretrained weights)
# False: Train audio encoder (rarely used)

# Audio Adapter Training
--freeze_audio_encoder_adapter True/False # Freeze audio adapter
# True: Keep audio adapter frozen
# False: Train audio adapter (used in Stage 2)
```

#### **3. Language Model Parameters**

```bash
# Language Model Training
--freeze_backbone True/False        # Freeze language model backbone
# True: Keep language model completely frozen
# False: Allow language model fine-tuning (used in Stage 3)
```

### **Training Step Comparison Table**

| Component | Stage 1.1 (Vision Alignment) | Stage 1.2 (Vision Understanding) | Stage 1.3 (Vision SFT) | Stage 2.1 (Audio Alignment) | Stage 2.2 (Audio SFT) | Stage 3.1 (Codec) | Stage 3.2 (NAR+AR) |
|-----------|------------------------------|----------------------------------|------------------------|-----------------------------|----------------------|-------------------|-------------------|
| **Language Model** | 🔒 Frozen | ✅ Trained | ✅ Trained | 🔒 Frozen | ✅ Trained | 🔒 Frozen | 🔒 Frozen |
| **Vision Encoder** | 🔒 Frozen | ✅ Trained | ✅ Trained | 🔒 Frozen | ✅ Trained | 🔒 Frozen | 🔒 Frozen |
| **Vision Projector** | ✅ Trained | ✅ Trained | ✅ Trained | 🔒 Frozen | ✅ Trained | 🔒 Frozen | 🔒 Frozen |
| **Audio Encoder** | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | ✅ Trained | 🔒 Frozen | 🔒 Frozen |
| **Audio Adapter** | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | ✅ Trained | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen |
| **Audio Projector** | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | ✅ Trained | ✅ Trained | 🔒 Frozen | 🔒 Frozen |
| **Codec Decoder** | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | ✅ Trained | 🔒 Frozen |
| **Codec Encoder** | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | ✅ Trained | 🔒 Frozen |
| **AR Speech Decoder** | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | ✅ Trained |
| **NAR Speech Decoder** | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | 🔒 Frozen | ✅ Trained |

### **Learning Rate Strategy by Component**

#### **Stage 1: Vision Pretraining**
```bash
--learning_rate 5e-4                # High LR for vision projector training
# Only vision projector is trained, so high LR is safe
```

#### **Stage 2: Audio Pretraining**
```bash
--learning_rate 5e-4                # High LR for audio projector training
# Only audio projector and adapter are trained
```

#### **Stage 3: Fine-tuning**
```bash
--learning_rate 2e-5                # Lower LR for language model
--mm_projector_lr 2e-6              # Even lower LR for vision projector
# Multiple components trained, so lower LRs for stability
```

### **Advanced Freeze Configurations**

#### **Conservative Fine-tuning** (Most Common)
```bash
# Stage 3: Keep most components frozen
--tune_mm_mlp_adapter True          # Fine-tune vision projector
--tune_audio_mlp_adapter True       # Fine-tune audio projector
--freeze_audio_encoder True         # Keep audio encoder frozen
--freeze_audio_encoder_adapter True # Keep audio adapter frozen
--unfreeze_vision_tower False       # Keep vision encoder frozen
--freeze_backbone False             # Fine-tune language model
```

#### **Aggressive Fine-tuning** (Advanced)
```bash
# Stage 3: Train more components
--tune_mm_mlp_adapter True          # Fine-tune vision projector
--tune_audio_mlp_adapter True       # Fine-tune audio projector
--freeze_audio_encoder False        # Train audio encoder (risky)
--freeze_audio_encoder_adapter False # Train audio adapter
--unfreeze_vision_tower True        # Train vision encoder
--freeze_backbone False             # Fine-tune language model
```

#### **Audio-Only Training** (Special Case)
```bash
# Train only audio components
--tune_mm_mlp_adapter False         # Keep vision projector frozen
--tune_audio_mlp_adapter True       # Train audio projector
--freeze_audio_encoder True         # Keep audio encoder frozen
--freeze_audio_encoder_adapter False # Train audio adapter
--unfreeze_vision_tower False       # Keep vision encoder frozen
--freeze_backbone True              # Keep language model frozen
```

### **Freeze Parameter Best Practices**

#### **1. Progressive Unfreezing**
```bash
# Start with most components frozen, gradually unfreeze
Stage 1: Only vision projector trained
Stage 2: Only audio projector trained  
Stage 3: Language model + projectors trained
Stage 4: (Optional) All components trained
```

#### **2. Learning Rate Coordination**
```bash
# Coordinate learning rates with freeze settings
if tune_mm_mlp_adapter:
    --mm_projector_lr 2e-6          # Lower LR for fine-tuning
    
if not freeze_backbone:
    --learning_rate 2e-5             # Lower LR for language model
```

#### **3. Memory Management**
```bash
# More components trained = more memory usage
--freeze_audio_encoder True         # Keep large audio encoder frozen
--freeze_audio_encoder_adapter True # Keep audio adapter frozen
# This saves significant memory during training
```

#### **4. Stability Considerations**
```bash
# Avoid training too many components simultaneously
--unfreeze_vision_tower False       # Keep vision encoder frozen for stability
--freeze_audio_encoder True         # Keep audio encoder frozen
# Only train projectors and language model for stability
```

### **Common Freeze Configuration Mistakes**

#### **❌ Wrong: Training Everything at Once**
```bash
--tune_mm_mlp_adapter True
--tune_audio_mlp_adapter True
--freeze_audio_encoder False        # ❌ Don't train audio encoder
--freeze_audio_encoder_adapter False
--unfreeze_vision_tower True        # ❌ Don't train vision encoder
--freeze_backbone False
# This can cause instability and poor convergence
```

#### **✅ Correct: Progressive Training**
```bash
# Stage 1: Only vision projector
--tune_mm_mlp_adapter True
--freeze_audio_encoder True
--freeze_audio_encoder_adapter True

# Stage 2: Only audio projector
--tune_audio_mlp_adapter True
--tune_mm_mlp_adapter False

# Stage 3: Fine-tune key components
--tune_mm_mlp_adapter True
--tune_audio_mlp_adapter True
--freeze_audio_encoder True
--freeze_backbone False
```

This freeze/non-freeze strategy ensures stable, progressive training while maintaining the performance of each modality throughout the training process.

## 🎓 Training Stages

VITA uses a progressive training strategy with three main stages, each containing multiple child stages for specific training objectives.

### Stage 1: Vision-Language Training

#### **Stage 1.1: Vision Alignment** (`pretrain_mlp_qwen.sh`)
**Purpose**: Align visual features with the frozen Large Language Model (LLM).

**Key Parameters**:
- `tune_mm_mlp_adapter: True` - Train vision projector
- `freeze_audio_encoder: True` - Keep audio encoder frozen
- `freeze_audio_encoder_adapter: True` - Keep audio adapter frozen
- `learning_rate: 5e-4` - Higher learning rate for projector

**Component Status**:
- **✅ TRAINED**: Vision Adapter (MLP)
- **🔒 FROZEN**: LLM, Visual Encoder, Audio Encoder, Audio Adapter

**Data Used**: 20% Caption data (ShareGPT4V, COCO, LLaVA-Instruct)
**Output**: Discrete Text Tokens from the LLM

#### **Stage 1.2: Vision Understanding** (`finetune_qwen.sh`)
**Purpose**: Enhance the LLM's ability to understand and reason about visual content.

**Key Parameters**:
- `unfreeze_vision_tower: True` - Train vision encoder
- `freeze_audio_encoder: True` - Keep audio encoder frozen
- `freeze_audio_encoder_adapter: True` - Keep audio adapter frozen
- `learning_rate: 2e-5` - Lower learning rate for fine-tuning
- `mm_projector_lr: 2e-6` - Even lower LR for vision projector

**Component Status**:
- **✅ TRAINED**: LLM, Vision Adapter, Visual Encoder
- **🔒 FROZEN**: Audio Encoder, Audio Adapter

**Data Used**: 100% Caption data
**Output**: Discrete Text Tokens from the LLM

#### **Stage 1.3: Vision SFT (Supervised Fine-Tuning)** (`finetuneTask_qwen.sh`)
**Purpose**: Fine-tune the vision-language model on specific tasks like Question Answering (QA).

**Key Parameters**:
- `freeze_audio_encoder: True` - Keep audio encoder frozen
- `freeze_audio_encoder_adapter: True` - Keep audio adapter frozen
- `learning_rate: 2e-5` - Lower learning rate for fine-tuning

**Component Status**:
- **✅ TRAINED**: LLM, Vision Adapter, Visual Encoder
- **🔒 FROZEN**: Audio Encoder, Audio Adapter

**Data Used**: 20% Caption & 100% QA data
**Output**: Discrete Text Tokens from the LLM

### Stage 2: Audio Input Tuning

#### **Stage 2.1: Audio Alignment** (`pretrain_audio_mlp_qwen.sh`)
**Purpose**: Align speech features with the frozen LLM.

**Key Parameters**:
- `tune_audio_mlp_adapter: True` - Train audio projector
- `tune_mm_mlp_adapter: False` - Keep vision projector frozen
- `freeze_audio_encoder: True` - Keep audio encoder frozen
- `freeze_audio_encoder_adapter: False` - Train audio adapter
- `learning_rate: 5e-4` - Higher learning rate for audio projector

**Component Status**:
- **✅ TRAINED**: Speech Adapter (MLP), Audio Adapter
- **🔒 FROZEN**: LLM, Visual Encoder, Vision Adapter, Audio Encoder

**Data Used**: Speech-transcription pairs
**Output**: Discrete Text Tokens from the LLM

#### **Stage 2.2: Audio SFT (Supervised Fine-Tuning)** (`finetuneTask_qwen.sh`)
**Purpose**: Fine-tune the multimodal LLM to handle audio input alongside vision and text.

**Key Parameters**:
- `freeze_audio_encoder: True` - Keep audio encoder frozen
- `freeze_audio_encoder_adapter: True` - Keep audio adapter frozen
- `learning_rate: 2e-5` - Lower learning rate for fine-tuning

**Component Status**:
- **✅ TRAINED**: LLM, Vision Adapter, Visual Encoder, Speech Adapter, Speech Encoder
- **🔒 FROZEN**: Audio Encoder, Audio Adapter

**Data Used**: Speech/Text Caption (4%) & QA (20%)
**Output**: Discrete Text Tokens from the LLM

### Stage 3: Audio Output Tuning

#### **Stage 3.1: Codec Training**
**Purpose**: Train an audio codec for efficient speech representation and reconstruction.

**Component Status**:
- **✅ TRAINED**: Codec Decoder, Codec Encoder
- **🔒 FROZEN**: All other components

**Data Used**: Speech data
**Output**: Discrete Speech Tokens and reconstructed Speech

#### **Stage 3.2: NAR + AR Decoder Training**
**Purpose**: Train both Non-Autoregressive (NAR) and Autoregressive (AR) speech decoders.

**Component Status**:
- **✅ TRAINED**: AR Speech Decoder, NAR Speech Decoder
- **🔒 FROZEN**: LLM Embedding

**Data Used**: Text-Speech Data
**Output**: Discrete Speech Tokens from AR and NAR Speech Decoders

## 🖼️ Stage 1: Vision-Language Alignment

### Required Datasets

#### 1. ShareGPT4V Dataset
**Source**: [Lin-Chen/ShareGPT4V](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V) on Hugging Face

**Components**:
- `sharegpt4v_instruct_gpt4-vision_cap100k.json` (100K samples)
  - High-quality captions generated by GPT4-Vision
  - Used for supervised fine-tuning
  - Best quality but smaller size

- `share-captioner_coco_lcs_sam_1246k_1107.json` (1.2M samples)
  - Captions generated by Share-Captioner model
  - Trained on GPT4-Vision data
  - Large scale for pre-training

- `sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json` (665K samples)
  - Mixed dataset for fine-tuning
  - Combines multiple sources: COCO, LCS, SAM, DIV2K
  - Curated for optimal training

**Total**: ~2M samples for comprehensive vision-language training

#### 2. COCO Dataset
**Source**: [COCO Official Website](http://cocodataset.org/)

**Components**:
- **Images**: 
  - `train2017.zip` (~18GB) - Training images
  - `val2017.zip` (~1GB) - Validation images
- **Annotations**: 
  - `annotations_trainval2017.zip` (~250MB) - Object detection and caption annotations

**Usage**: Provides the actual image files referenced in ShareGPT4V conversations

#### 3. LLaVA-Instruct Dataset
**Source**: [liuhaotian/LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)

**Components**:
- `llava_v1_5_mix665k.json` - Instruction-following vision-language data
- Used for instruction tuning and conversation format training

### Dataset Configuration for Stage 1

```python
# In vita/config/dataset_config.py
ShareGPT4V = {"chat_path": "/data/vita/conversations/sharegpt4v_conversations.json"}
ShareGPT4V0 = {"chat_path": "/data/vita/conversations/sharegpt4v_conversations_v0.json"}

FolderDict = {
    "sharegpt4": "/data/vita/images/coco/train2017",
    "llava_instruct": "/data/vita/images/llava",
}

# In vita/config/__init__.py
NaturalCap0 = [ShareGPT4V0]
NaturalCap = [ShareGPT4V]

DataConfig = {
    "Pretrain_video": NaturalCap0,  # Used in Stage 1
}
```

## 🎵 Stage 2: Audio-Language Alignment

### Required Datasets

#### 1. VITA Audio Dataset
**Source**: [VITA-MLLM/VITA-Audio-Data](https://huggingface.co/datasets/VITA-MLLM/VITA-Audio-Data) on Hugging Face

**Components**:
- Various audio files for different tasks:
  - Speech Question Answering (QA)
  - Automatic Speech Recognition (ASR)
  - Text-to-Speech (TTS)
  - Audio captioning and understanding

**Directory Structure**:
```
new_value_dict_0717/output_wavs/
├── f61cf238b7872b4903e1fc15dcb5a50c.wav
├── a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6.wav
└── ...
```

#### 2. AudioCaps Dataset (Alternative)
**Source**: [AudioCaps Official Website](https://audiocaps.org/)

**Usage**: Alternative audio dataset for audio-language alignment training

### Dataset Configuration for Stage 2

```python
# In vita/config/dataset_config.py
AudioFolder = "/data/vita/audio_data"

AudioDataset = {"chat_path": "/data/vita/conversations/audio_conversations.json"}

# In vita/config/__init__.py
AudioCap = [AudioDataset]

DataConfig = {
    "Pretrain_audio": AudioCap,  # Used in Stage 2
}
```

## 🎯 Stage 3: Multimodal Fine-tuning

### Required Datasets

#### 1. Task-Specific Datasets
**Purpose**: Fine-tune the model for specific multimodal tasks

**Examples**:
- Video understanding tasks
- Audio-visual question answering
- Multimodal instruction following
- Custom domain-specific datasets

#### 2. Combined Multimodal Data
**Purpose**: End-to-end training with all modalities

**Components**:
- Vision-language data from Stage 1
- Audio-language data from Stage 2
- Additional task-specific multimodal samples

### Dataset Configuration for Stage 3

```python
# In vita/config/__init__.py
CustomCap = [CustomDataset]

DataConfig = {
    "Custom_task": CustomCap,  # Used in Stage 3
}
```

## 📥 Dataset Download Instructions

### Method 1: Using Hugging Face Hub (Recommended)

```bash
# Install and upgrade huggingface_hub
pip install --upgrade huggingface_hub

# Login to Hugging Face
hf login

# Create data directory
mkdir -p /data/vita/datasets
cd /data/vita/datasets

# Download ShareGPT4V dataset (conversation files only)
hf download Lin-Chen/ShareGPT4V --repo-type dataset --local-dir sharegpt4v

# Download LLaVA-Instruct dataset
hf download liuhaotian/LLaVA-Instruct-150K --repo-type dataset --local-dir llava_instruct

# Download VITA Audio dataset
hf download VITA-MLLM/VITA-Audio-Data --repo-type dataset --local-dir vita_audio_data
```

### Method 2: Complete Dataset Download Script

```bash
#!/bin/bash
# VITA Dataset Download Script
set -e

DATA_DIR="/data/vita/datasets"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "Downloading VITA training datasets..."

# Download ShareGPT4V dataset (3 JSON files with 1.2M+ samples)
echo "Downloading ShareGPT4V dataset..."
if [ ! -d "sharegpt4v" ]; then
    mkdir -p sharegpt4v
    wget -O sharegpt4v/sharegpt4v_instruct_gpt4-vision_cap100k.json \
        "https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/sharegpt4v_instruct_gpt4-vision_cap100k.json"
    wget -O sharegpt4v/share-captioner_coco_lcs_sam_1246k_1107.json \
        "https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/share-captioner_coco_lcs_sam_1246k_1107.json"
    wget -O sharegpt4v/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json \
        "https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json"
fi

# Download COCO images
echo "Downloading COCO train2017 images..."
if [ ! -d "coco/images/train2017" ]; then
    mkdir -p coco/images
    cd coco/images
    wget http://images.cocodataset.org/zips/train2017.zip
    unzip train2017.zip
    rm train2017.zip
    
    wget http://images.cocodataset.org/zips/val2017.zip
    unzip val2017.zip
    rm val2017.zip
    
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip annotations_trainval2017.zip
    rm annotations_trainval2017.zip
    cd ../..
fi

# Download LLaVA-Instruct
echo "Downloading LLaVA-Instruct dataset..."
if [ ! -d "llava_instruct" ]; then
    wget -O llava_instruct.zip \
        "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-158K/resolve/main/llava_v1_5_mix665k.json"
    mkdir -p llava_instruct
    unzip llava_instruct.zip -d llava_instruct/
    rm llava_instruct.zip
fi

# Download VITA Audio dataset
echo "Downloading VITA's audio dataset..."
if [ ! -d "vita_audio_data" ]; then
    git clone https://huggingface.co/datasets/VITA-MLLM/VITA-Audio-Data vita_audio_data
fi

echo "Dataset download completed!"
```

### Method 3: Using Git Clone

```bash
# Install git-lfs if not already installed
git lfs install

# Create data directory
mkdir -p /data/vita/datasets
cd /data/vita/datasets

# Clone datasets
git clone https://huggingface.co/datasets/Lin-Chen/ShareGPT4V sharegpt4v
git clone https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K llava_instruct
git clone https://huggingface.co/datasets/VITA-MLLM/VITA-Audio-Data vita_audio_data
```

## ⚙️ Dataset Configuration

### Configuration Files

#### 1. Dataset Configuration (`vita/config/dataset_config.py`)

This file defines the base dataset configurations and folder mappings. It serves as the foundation for all dataset-related configurations.

```python
# Audio data folder path - base directory for all audio files
AudioFolder = "/data/vita/audio_data"

# Image folder mappings for different datasets
FolderDict = {
    "sharegpt4": "/data/vita/images/coco/train2017",
    "custom_dataset": "/data/vita/images/custom",
    "llava_instruct": "/data/vita/images/llava",
}

# Dataset-specific conversation file paths
ShareGPT4V = {"chat_path": "/data/vita/conversations/sharegpt4v_conversations.json"}
ShareGPT4V0 = {"chat_path": "/data/vita/conversations/sharegpt4v_conversations_v0.json"}
CustomDataset = {"chat_path": "/data/vita/conversations/custom_conversations.json"}
AudioDataset = {"chat_path": "/data/vita/conversations/audio_conversations.json"}
```

**Purpose and Structure**:
- **`AudioFolder`**: Global base path for all audio files. Used by the system to locate audio files when processing audio-language data.
- **`FolderDict`**: Global mapping of folder keys to actual file system paths. These keys are referenced in conversation JSON files to locate images.
- **Individual Dataset Variables**: Each dataset (ShareGPT4V, AudioDataset, etc.) defines its specific configuration, primarily the path to its conversation JSON file.

**Key Features**:
- **Modular Design**: Each dataset is defined separately, making it easy to add/remove datasets
- **Path Flexibility**: Folder mappings can be overridden per dataset or use global defaults
- **Extensibility**: Easy to add new datasets by creating new variables

#### 2. Main Configuration (`vita/config/__init__.py`)

This file imports all dataset configurations and organizes them into logical groups for different training stages.

```python
from .dataset_config import *

# Dataset groupings for different training stages
NaturalCap0 = [ShareGPT4V0]  # Alternative ShareGPT4V configuration
NaturalCap = [ShareGPT4V]    # Main ShareGPT4V configuration
CustomCap = [CustomDataset]  # Custom dataset configuration
AudioCap = [AudioDataset]    # Audio-specific dataset

# Main data configuration mapping for different training stages
DataConfig = {
    "Pretrain_video": NaturalCap,      # Stage 1: Vision-Language Alignment
    "Pretrain_audio": AudioCap,        # Stage 2: Audio-Language Alignment  
    "Custom_task": CustomCap,          # Stage 3: Multimodal Fine-tuning
}

# Datasets that don't use image patching (for specific processing)
NoPatchSets = ["khair", "jester"]
```

**Purpose and Structure**:
- **Dataset Grouping**: Combines individual datasets into logical groups (NaturalCap, AudioCap, etc.)
- **Stage Mapping**: Maps dataset groups to specific training stages via `DataConfig`
- **Special Processing**: Defines datasets that require special handling (`NoPatchSets`)

**Key Features**:
- **Hierarchical Organization**: Individual datasets → Groups → Training stages
- **Training Stage Integration**: Direct mapping from training script parameters to dataset groups
- **Flexible Grouping**: Multiple datasets can be combined in a single group for multi-dataset training

### How Configuration Files Work Together

#### **File Relationship Flow**
```
dataset_config.py (Base Definitions)
    ↓ (imports via "from .dataset_config import *")
__init__.py (Organization & Grouping)
    ↓ (used by training scripts)
LazySupervisedDataset (Runtime Loading)
```

#### **Step-by-Step Configuration Process**

1. **Base Definitions** (`dataset_config.py`):
   ```python
   # Define individual dataset configurations
   ShareGPT4V = {"chat_path": "/data/vita/conversations/sharegpt4v.json"}
   FolderDict = {"sharegpt4": "/data/vita/images/coco/train2017"}
   ```

2. **Organization** (`__init__.py`):
   ```python
   # Group datasets logically
   NaturalCap = [ShareGPT4V]
   
   # Map groups to training stages
   DataConfig = {"Pretrain_video": NaturalCap}
   ```

3. **Runtime Usage** (Training Script):
   ```bash
   --dataset_use Pretrain_video  # Triggers DataConfig["Pretrain_video"]
   ```

4. **Dataset Loading** (`LazySupervisedDataset`):
   ```python
   # Resolves to: DataConfig["Pretrain_video"] → NaturalCap → [ShareGPT4V]
   dataset_list = DataConfig[str(data_args.dataset_use)]
   ```

### Configuration File Examples

#### **Current Minimal Configuration**
```python
# dataset_config.py
AudioFolder = ""
FolderDict = {
    "sharegpt4": "",
}
ShareGPT4V = {"chat_path": ""}
ShareGPT4V0 = {"chat_path": ""}

# __init__.py
NaturalCap0 = [ShareGPT4V0]
DataConfig = {
    "Pretrain_video": NaturalCap0,
}
```

**Issues with Current Config**:
- Empty paths will cause file not found errors
- Limited to only one dataset group
- No audio or custom dataset configurations

#### **Complete Working Configuration**
```python
# dataset_config.py
AudioFolder = "/data/vita/audio_data"

FolderDict = {
    "sharegpt4": "/data/vita/images/coco/train2017",
    "llava": "/data/vita/images/llava",
    "custom": "/data/vita/images/custom",
}

# Individual dataset configurations
ShareGPT4V = {
    "chat_path": "/data/vita/conversations/sharegpt4v.json",
    "data_ratio": 1.0  # Use 100% of data
}

ShareGPT4V0 = {
    "chat_path": "/data/vita/conversations/sharegpt4v_v0.json",
    "data_ratio": 0.8  # Use 80% of data
}

LLaVADataset = {
    "chat_path": "/data/vita/conversations/llava.json",
    "llava": "/data/vita/images/llava",  # Override global mapping
    "data_ratio": 0.6
}

AudioDataset = {
    "chat_path": "/data/vita/conversations/audio.json",
    "data_ratio": 0.5
}

CustomDataset = {
    "chat_path": "/data/vita/conversations/custom.json",
    "custom": "/data/vita/images/custom",
    "data_ratio": 1.0
}

# __init__.py
from .dataset_config import *

# Create dataset groups
NaturalCap0 = [ShareGPT4V0]  # Single dataset group
NaturalCap = [ShareGPT4V, LLaVADataset]  # Multi-dataset group
AudioCap = [AudioDataset]
CustomCap = [CustomDataset]

# Map groups to training stages
DataConfig = {
    "Pretrain_video": NaturalCap0,      # Stage 1: Single dataset
    "Pretrain_video_full": NaturalCap,  # Stage 1: Multiple datasets
    "Pretrain_audio": AudioCap,         # Stage 2: Audio dataset
    "Custom_task": CustomCap,           # Stage 3: Custom dataset
}

# Special processing datasets
NoPatchSets = ["khair", "jester"]
```

#### **Advanced Multi-Stage Configuration**
```python
# dataset_config.py
AudioFolder = "/data/vita/audio_data"

FolderDict = {
    "sharegpt4": "/data/vita/images/coco/train2017",
    "llava": "/data/vita/images/llava",
    "audio": "/data/vita/audio_data",
    "custom": "/data/vita/images/custom",
}

# Stage 1: Vision-Language datasets
ShareGPT4V_Small = {
    "chat_path": "/data/vita/conversations/sharegpt4v_small.json",
    "data_ratio": 0.3
}

ShareGPT4V_Full = {
    "chat_path": "/data/vita/conversations/sharegpt4v_full.json",
    "data_ratio": 1.0
}

LLaVA_Small = {
    "chat_path": "/data/vita/conversations/llava_small.json",
    "data_ratio": 0.2
}

# Stage 2: Audio-Language datasets
AudioDataset_ASR = {
    "chat_path": "/data/vita/conversations/audio_asr.json",
    "data_ratio": 0.8
}

AudioDataset_TTS = {
    "chat_path": "/data/vita/conversations/audio_tts.json",
    "data_ratio": 0.6
}

# Stage 3: Multimodal datasets
MultimodalDataset = {
    "chat_path": "/data/vita/conversations/multimodal.json",
    "custom": "/data/vita/images/custom",
    "data_ratio": 1.0
}

# __init__.py
from .dataset_config import *

# Stage 1: Vision-Language groups
VisionCap_Small = [ShareGPT4V_Small, LLaVA_Small]  # Small datasets for quick training
VisionCap_Full = [ShareGPT4V_Full]                 # Full dataset for final training

# Stage 2: Audio-Language groups
AudioCap_ASR = [AudioDataset_ASR]                   # ASR-focused training
AudioCap_TTS = [AudioDataset_TTS]                   # TTS-focused training
AudioCap_All = [AudioDataset_ASR, AudioDataset_TTS] # Combined audio training

# Stage 3: Multimodal groups
MultimodalCap = [MultimodalDataset]

# Complete training stage mapping
DataConfig = {
    # Stage 1: Vision-Language Alignment
    "Pretrain_video_small": VisionCap_Small,    # Quick vision training
    "Pretrain_video_full": VisionCap_Full,      # Full vision training
    
    # Stage 2: Audio-Language Alignment
    "Pretrain_audio_asr": AudioCap_ASR,         # ASR training
    "Pretrain_audio_tts": AudioCap_TTS,         # TTS training
    "Pretrain_audio_all": AudioCap_All,         # Combined audio training
    
    # Stage 3: Multimodal Fine-tuning
    "Multimodal_task": MultimodalCap,           # Final multimodal training
}

NoPatchSets = ["khair", "jester"]
```

### Configuration Best Practices

#### **1. Path Management**
```python
# Use absolute paths to avoid issues
AudioFolder = "/data/vita/audio_data"  # ✅ Good
AudioFolder = "./audio_data"           # ❌ Avoid relative paths

# Validate paths exist
import os
if not os.path.exists(AudioFolder):
    print(f"Warning: AudioFolder path does not exist: {AudioFolder}")
```

#### **2. Data Ratio Configuration**
```python
# Use appropriate data ratios for different stages
ShareGPT4V = {
    "chat_path": "/data/vita/conversations/sharegpt4v.json",
    "data_ratio": 0.1  # Use 10% for quick testing
}

# For production training
ShareGPT4V_Production = {
    "chat_path": "/data/vita/conversations/sharegpt4v.json",
    "data_ratio": 1.0  # Use 100% for full training
}
```

#### **3. Folder Mapping Strategy**
```python
# Global mappings for common datasets
FolderDict = {
    "sharegpt4": "/data/vita/images/coco/train2017",
    "llava": "/data/vita/images/llava",
}

# Override for specific datasets if needed
CustomDataset = {
    "chat_path": "/data/vita/conversations/custom.json",
    "custom_images": "/data/vita/images/custom",  # Override global mapping
}
```

#### **4. Multi-Dataset Grouping**
```python
# Combine related datasets
VisionCap = [ShareGPT4V, LLaVADataset, CustomVisionDataset]

# Keep datasets separate for fine-grained control
DataConfig = {
    "Pretrain_video_sharegpt4": [ShareGPT4V],
    "Pretrain_video_llava": [LLaVADataset],
    "Pretrain_video_all": VisionCap,  # All vision datasets
}
```

### Key Variables and Constants

#### 1. **Core Constants** (`vita/constants.py`)

```python
# Image/Video Processing Constants
MAX_IMAGE_LENGTH = 16        # Maximum number of image patches (8#16#32#64)
MIN_IMAGE_LENGTH = 4         # Minimum number of image patches
DEFAULT_DATA_RATIO = 1.0     # Default sampling ratio (0.124#0.5#0.2#1.0)

# Token Constants
DEFAULT_IMAGE_TOKEN = "<image>"   # Image placeholder token
DEFAULT_VIDEO_TOKEN = "<video>"   # Video placeholder token  
DEFAULT_AUDIO_TOKEN = "<audio>"   # Audio placeholder token
IGNORE_INDEX = -100               # Index for ignored tokens in loss calculation
IMAGE_TOKEN_INDEX = -200          # Special index for image tokens
AUDIO_TOKEN_INDEX = -500          # Special index for audio tokens

# System Constants
CONTROLLER_HEART_BEAT_EXPIRATION = 30  # Controller heartbeat timeout
WORKER_HEART_BEAT_INTERVAL = 15        # Worker heartbeat interval
LOGDIR = "gradio-logs"                 # Log directory
GLOBAL_WEIGHTS_PATH = "/path/to/model_weights"  # Global model weights path
```

#### 2. **DataArguments Parameters**

```python
@dataclass
class DataArguments:
    # Core Parameters
    dataset_use: str = "temp"                    # Dataset configuration key
    lazy_preprocess: bool = False                # Enable lazy loading for memory efficiency
    is_multimodal: bool = True                   # Enable multimodal processing
    
    # Image Processing Parameters
    image_folder: Optional[str] = None           # Specific image folder override
    image_aspect_ratio: str = "square"           # Image aspect ratio handling
    min_dynamic_patch: int = 1                   # Minimum dynamic image patches
    max_dynamic_patch: int = 12                  # Maximum dynamic image patches
    use_thumbnail: bool = True                   # Use thumbnail for large images
```

#### 3. **Dataset Configuration Variables**

```python
# Global Configuration Variables
AudioFolder = ""                    # Base path for all audio files
FolderDict = {}                     # Global image folder mappings
DataConfig = {}                     # Main dataset configuration mapping
NoPatchSets = []                    # Datasets that don't use image patching

# Individual Dataset Variables
ShareGPT4V = {}                     # Main ShareGPT4V dataset configuration
ShareGPT4V0 = {}                    # Alternative ShareGPT4V configuration
AudioDataset = {}                   # Audio dataset configuration
CustomDataset = {}                  # Custom dataset configuration

# Dataset Group Variables
NaturalCap = []                     # Natural caption dataset group
NaturalCap0 = []                    # Alternative natural caption group
AudioCap = []                       # Audio dataset group
CustomCap = []                      # Custom dataset group
```

#### 4. **Video Processing Parameters**

```python
# Video Decoding Parameters (in _get_rawvideo_dec function)
max_frames = 32                     # Maximum frames to extract from video
min_frames = 4                      # Minimum frames to extract from video
image_resolution = 384              # Target image resolution
video_framerate = 1                 # Video frame sampling rate
image_aspect_ratio = "pad"          # How to handle aspect ratios
```

#### 5. **Training Script Variables**

```bash
# Key Training Parameters
--dataset_use Pretrain_video        # Dataset configuration key
--tune_mm_mlp_adapter True          # Train vision projector
--tune_audio_mlp_adapter True       # Train audio projector
--freeze_audio_encoder True         # Freeze audio encoder weights
--freeze_audio_encoder_adapter True # Freeze audio adapter weights
--image_aspect_ratio square         # Image aspect ratio handling
--group_by_modality_length False    # Group samples by modality length
--model_max_length 6200             # Maximum sequence length
--lazy_preprocess True              # Enable lazy preprocessing
```

### Variable Flow and Data Loading Process

#### 1. **Configuration Flow**
```
Training Script (--dataset_use Pretrain_video)
    ↓
DataConfig["Pretrain_video"] → NaturalCap0 → [ShareGPT4V0]
    ↓
LazySupervisedDataset.__init__()
    ↓
Load conversation JSON from ShareGPT4V0["chat_path"]
    ↓
Apply data_ratio sampling (DEFAULT_DATA_RATIO = 1.0)
    ↓
Build folder_dict from dataset config + FolderDict
    ↓
Process multimodal data with constants (MAX_IMAGE_LENGTH, etc.)
```

#### 2. **Key Variable Interactions**

```python
# Step 1: Dataset Selection
dataset_use = "Pretrain_video"  # From training script
dataset_list = DataConfig[dataset_use]  # → [ShareGPT4V0]

# Step 2: Data Loading
for dataset_config in dataset_list:  # ShareGPT4V0
    chat_path = dataset_config["chat_path"]  # "/path/to/conversations.json"
    data_ratio = dataset_config.get("data_ratio", DEFAULT_DATA_RATIO)  # 1.0
    
    # Load and sample data
    raw_data = json.load(open(chat_path, "r"))
    sampled_data = random.sample(raw_data, int(len(raw_data) * data_ratio))
    
    # Build folder mappings
    for key, value in dataset_config.items():
        if key != "chat_path":
            folder_dict[key] = value

# Step 3: Apply Global Mappings
for key, value in FolderDict.items():
    if key not in folder_dict:
        folder_dict[key] = value

# Step 4: Process Each Sample
for sample in sampled_data:
    if "image" in sample:
        image_path = sample["image"]  # "COCO_train2017_000000000139.jpg"
        folder_key = "sharegpt4"  # From conversation format
        full_path = folder_dict[folder_key] + "/" + image_path
        # Process with MAX_IMAGE_LENGTH, MIN_IMAGE_LENGTH constants
```

#### 3. **Constants Usage in Processing**

```python
# Image Processing
if "image" in sample:
    # Use MAX_IMAGE_LENGTH = 16 for maximum patches
    # Use MIN_IMAGE_LENGTH = 4 for minimum patches
    # Use DEFAULT_IMAGE_TOKEN = "<image>" for tokenization

# Video Processing  
if "video" in sample:
    # Use max_frames = 32, min_frames = 4
    # Use image_resolution = 384
    # Use video_framerate = 1

# Audio Processing
if "audio" in sample:
    # Use DEFAULT_AUDIO_TOKEN = "<audio>" for tokenization
    # Use AudioFolder for base path
    # Use AUDIO_TOKEN_INDEX = -500 for special indexing

# Token Processing
# Use IGNORE_INDEX = -100 for loss calculation
# Use IMAGE_TOKEN_INDEX = -200 for image tokens
```

### Dataset Structure

Each dataset consists of three components:

1. **Conversation JSON files**: Contains the text conversations and file paths
2. **Image files**: The actual images referenced in conversations  
3. **Audio files**: The actual audio files referenced in conversations

**Important**: Hugging Face datasets only provide conversation files. You need to obtain images and audio files separately.

### Variable Configuration Examples

#### **Minimal Working Configuration**
```python
# dataset_config.py
AudioFolder = "/data/vita/audio_data"
FolderDict = {
    "sharegpt4": "/data/vita/images/coco/train2017",
}
ShareGPT4V0 = {
    "chat_path": "/data/vita/conversations/sharegpt4v.json",
    "data_ratio": 0.5  # Use 50% of data
}

# __init__.py
DataConfig = {
    "Pretrain_video": [ShareGPT4V0],
}
```

#### **Advanced Multi-Dataset Configuration**
```python
# dataset_config.py
AudioFolder = "/data/vita/audio_data"

FolderDict = {
    "sharegpt4": "/data/vita/images/coco/train2017",
    "llava": "/data/vita/images/llava",
    "custom": "/data/vita/images/custom",
}

ShareGPT4V = {
    "chat_path": "/data/vita/conversations/sharegpt4v.json",
    "sharegpt4": "/data/vita/images/coco/train2017",  # Override global mapping
    "data_ratio": 1.0
}

LLaVADataset = {
    "chat_path": "/data/vita/conversations/llava.json",
    "llava": "/data/vita/images/llava",
    "data_ratio": 0.8
}

AudioDataset = {
    "chat_path": "/data/vita/conversations/audio.json",
    "data_ratio": 0.6
}

# __init__.py
NaturalCap = [ShareGPT4V, LLaVADataset]  # Multiple datasets in one group
AudioCap = [AudioDataset]

DataConfig = {
    "Pretrain_video": NaturalCap,    # Uses both ShareGPT4V and LLaVA
    "Pretrain_audio": AudioCap,      # Uses only AudioDataset
}
```

## 💻 Code Usage

### Dataset Loading in Training Scripts

#### 1. Training Script Configuration

```bash
# Stage 1: Vision-Language Alignment
--dataset_use Pretrain_video
--tune_mm_mlp_adapter True
--freeze_audio_encoder True

# Stage 2: Audio-Language Alignment  
--dataset_use Pretrain_audio
--tune_audio_mlp_adapter True
--tune_mm_mlp_adapter False

# Stage 3: Multimodal Fine-tuning
--dataset_use Custom_task
# Both adapters are fine-tuned
```

#### 2. Dataset Class Implementation

The main dataset class is `LazySupervisedDataset` in `vita/util/data_utils_video_audio_patch.py`:

```python
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        
        # Load dataset configuration
        dataset_list = DataConfig[str(data_args.dataset_use)]
        
        # Load conversation data
        list_data_dict = []
        self.folder_dict = {}
        for i in dataset_list:
            data_ratio = i.get("data_ratio", DEFAULT_DATA_RATIO)
            data_i = json.load(open(i["chat_path"], "r"))
            len_data_i = len(data_i)
            data_i = random.sample(data_i, int(len_data_i * data_ratio))
            list_data_dict += data_i
            
            # Map image folders
            image_folder = [folder for folder in i if folder is not "chat_path"]
            for folder in image_folder:
                if folder not in self.folder_dict:
                    self.folder_dict[folder] = i[folder]
        
        # Apply global folder mappings
        for key in FolderDict.keys():
            if key not in self.folder_dict:
                self.folder_dict[key] = FolderDict[key]
        
        random.shuffle(list_data_dict)
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
```

#### 3. Data Processing Pipeline

The dataset processing includes:

1. **Conversation Loading**: JSON files are loaded and sampled based on `data_ratio`
2. **Folder Mapping**: Image and audio folders are mapped using `FolderDict`
3. **Modality Detection**: The system detects whether samples contain images, videos, or audio
4. **Tokenization**: Text conversations are tokenized using the model's tokenizer
5. **Multimodal Processing**: Images/videos are processed through vision encoders, audio through audio encoders

### Key Data Processing Functions

#### Image/Video Processing
```python
def _get_rawvideo_dec(video_path, image_processor, max_frames=32, min_frames=4, 
                     image_resolution=384, video_framerate=1, s=None, e=None, 
                     image_aspect_ratio="pad"):
    # Video decoding and frame extraction
    # Returns patch_images and video_mask
```

#### Audio Processing
```python
def process_audio(audio_path, audio_processor):
    # Audio loading and preprocessing
    # Returns audio features
```

## 🛠️ Data Processing Tools

### Available Tools

#### 1. Dataset Concatenation
```bash
# Concatenate multiple datasets
python data_tools/concat_data_patch.py \
    --input_dirs /path/to/dataset1 /path/to/dataset2 \
    --output_dir /path/to/combined_dataset
```

#### 2. Statistics and Token Counting
```bash
# Generate dataset statistics
python data_tools/statistics_token_num.py \
    --data_path /path/to/dataset \
    --output_path /path/to/statistics.json
```

#### 3. Data Validation
```bash
# Validate dataset integrity
python data_tools/validate_dataset.py \
    --dataset_path /path/to/dataset \
    --check_images True \
    --check_audio True
```

### Best Practices

1. **Use Absolute Paths**: Always use absolute paths to avoid issues with working directory changes
2. **Consistent Naming**: Use consistent naming conventions for your dataset keys in `FolderDict`
3. **Path Validation**: Ensure all paths in your configuration actually exist before training
4. **Backup Configurations**: Keep backup copies of working configurations
5. **Documentation**: Document your dataset structure and configuration choices
6. **Testing**: Test data loading with a small subset before full training

## 📊 Dataset Statistics

### ShareGPT4V Dataset
- **Total Samples**: ~2M
- **GPT4-Vision Captions**: 100K (high quality)
- **Share-Captioner Captions**: 1.2M (large scale)
- **Mixed Dataset**: 665K (curated)

### COCO Dataset
- **Training Images**: ~118K images (~18GB)
- **Validation Images**: ~5K images (~1GB)
- **Annotations**: Object detection and caption data

### LLaVA-Instruct Dataset
- **Samples**: 665K instruction-following conversations
- **Format**: Vision-language instruction tuning data

### VITA Audio Dataset
- **Audio Files**: Various formats (WAV, MP3)
- **Tasks**: ASR, TTS, QA, Audio Captioning
- **Languages**: Multiple languages supported

## 🔧 Troubleshooting

### Common Issues

1. **Missing Image Files**: Ensure COCO images are downloaded and paths are correctly configured
2. **Audio File Not Found**: Verify audio dataset download and folder structure
3. **Path Configuration**: Check that all paths in `FolderDict` and dataset configurations are correct
4. **Memory Issues**: Use data sampling (`data_ratio`) for large datasets
5. **Format Compatibility**: Ensure conversation JSON files match the expected format

### Variable-Related Issues

#### **1. Empty Configuration Variables**
**Problem**: Configuration variables are empty strings
```python
ShareGPT4V = {"chat_path": ""}  # Empty path
FolderDict = {"sharegpt4": ""}  # Empty folder
```

**Solution**: Set actual paths
```python
ShareGPT4V = {"chat_path": "/data/vita/conversations/sharegpt4v.json"}
FolderDict = {"sharegpt4": "/data/vita/images/coco/train2017"}
```

#### **2. Wrong DataConfig Key**
**Problem**: Training script uses non-existent dataset key
```bash
--dataset_use Pretrain_audio  # But DataConfig doesn't have this key
```

**Error**: `KeyError: 'Pretrain_audio'`

**Solution**: Add missing configuration
```python
DataConfig = {
    "Pretrain_video": NaturalCap0,
    "Pretrain_audio": AudioCap,  # Add this
}
```

#### **3. Incorrect Data Ratio**
**Problem**: Dataset sampling issues
```python
ShareGPT4V = {"data_ratio": 0.0}  # No data loaded
ShareGPT4V = {"data_ratio": 2.0}  # More than 100% (will cause error)
```

**Solution**: Use valid ratios (0.0 < ratio <= 1.0)
```python
ShareGPT4V = {"data_ratio": 0.5}  # Use 50% of data
```

#### **4. Missing Folder Mappings**
**Problem**: Images not found during training
```
FileNotFoundError: /data/vita/images/coco/train2017/COCO_train2017_000000000139.jpg
```

**Solution**: Ensure proper folder mapping
```python
# In conversation JSON, image field references "sharegpt4" folder
{"image": "COCO_train2017_000000000139.jpg", "conversations": [...]}

# Must have corresponding folder mapping
FolderDict = {
    "sharegpt4": "/data/vita/images/coco/train2017",  # Must match
}
```

#### **5. Constants Configuration Issues**
**Problem**: Image processing errors due to wrong constants
```python
MAX_IMAGE_LENGTH = 0  # Will cause division by zero
MIN_IMAGE_LENGTH = 10  # Greater than MAX_IMAGE_LENGTH
```

**Solution**: Use proper constant values
```python
MAX_IMAGE_LENGTH = 16  # Maximum patches
MIN_IMAGE_LENGTH = 4   # Minimum patches (must be < MAX_IMAGE_LENGTH)
```

#### **6. DataArguments Parameter Conflicts**
**Problem**: Conflicting parameters in training
```bash
--lazy_preprocess False --is_multimodal False  # Contradictory settings
```

**Solution**: Use consistent parameters
```bash
--lazy_preprocess True --is_multimodal True  # Enable both
```

### Validation Commands

#### **Check Configuration Variables**
```python
# Check dataset configuration
python -c "
from vita.config import DataConfig, FolderDict, AudioFolder
print('DataConfig:', DataConfig)
print('FolderDict:', FolderDict)
print('AudioFolder:', AudioFolder)
"

# Check constants
python -c "
from vita.constants import *
print('MAX_IMAGE_LENGTH:', MAX_IMAGE_LENGTH)
print('MIN_IMAGE_LENGTH:', MIN_IMAGE_LENGTH)
print('DEFAULT_DATA_RATIO:', DEFAULT_DATA_RATIO)
print('DEFAULT_IMAGE_TOKEN:', DEFAULT_IMAGE_TOKEN)
"
```

#### **Test Dataset Loading**
```python
# Test dataset loading with specific configuration
python -c "
from vita.util.data_utils_video_audio_patch import LazySupervisedDataset, DataArguments
from transformers import AutoTokenizer

# Create test configuration
data_args = DataArguments(dataset_use='Pretrain_video')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')

try:
    dataset = LazySupervisedDataset(tokenizer, data_args)
    print(f'Dataset loaded successfully: {len(dataset)} samples')
    print(f'Folder mappings: {dataset.folder_dict}')
except Exception as e:
    print(f'Error loading dataset: {e}')
"
```

#### **Validate File Paths**
```python
# Check if all configured paths exist
python -c "
import os
from vita.config import DataConfig, FolderDict

# Check conversation files
for stage, datasets in DataConfig.items():
    for dataset in datasets:
        chat_path = dataset.get('chat_path', '')
        if chat_path and not os.path.exists(chat_path):
            print(f'Missing conversation file: {chat_path}')

# Check folder mappings
for key, path in FolderDict.items():
    if path and not os.path.exists(path):
        print(f'Missing folder: {key} -> {path}')
"
```

#### **Test Data Processing**
```python
# Test data processing pipeline
python -c "
from vita.util.data_utils_video_audio_patch import LazySupervisedDataset, DataArguments
from vita.constants import MAX_IMAGE_LENGTH, MIN_IMAGE_LENGTH

# Test with minimal configuration
data_args = DataArguments(
    dataset_use='Pretrain_video',
    lazy_preprocess=True,
    is_multimodal=True
)

print(f'Using MAX_IMAGE_LENGTH: {MAX_IMAGE_LENGTH}')
print(f'Using MIN_IMAGE_LENGTH: {MIN_IMAGE_LENGTH}')
print(f'Data arguments: {data_args}')
"
```

This documentation provides a comprehensive guide to understanding, downloading, configuring, and using datasets in the VITA training pipeline. Each stage has specific dataset requirements that are essential for successful model training.
