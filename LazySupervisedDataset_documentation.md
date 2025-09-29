# LazySupervisedDataset Documentation

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Class Definition](#class-definition)
4. [Configuration](#configuration)
5. [Data Processing Pipeline](#data-processing-pipeline)
6. [Token Processing](#token-processing)
7. [Supported Modalities](#supported-modalities)
8. [API Reference](#api-reference)
9. [Usage Examples](#usage-examples)
10. [Performance Considerations](#performance-considerations)

## Overview

The `LazySupervisedDataset` class is a PyTorch Dataset implementation designed for supervised fine-tuning of multimodal models. It implements lazy loading to efficiently handle large datasets by processing data samples on-demand rather than loading everything into memory at once.

### Purpose
- **Multimodal Training**: Supports text, images, videos, and audio data
- **Memory Efficiency**: Lazy loading prevents memory overflow with large datasets
- **Flexible Processing**: Handles various data formats and aspect ratios
- **Production Ready**: Optimized for training large-scale multimodal models

## Key Features

| Feature | Description |
|---------|-------------|
| **Multimodal Support** | Handles text, images, videos, and audio data seamlessly |
| **Lazy Loading** | Processes data samples on-demand for memory efficiency |
| **Dynamic Image Patching** | Automatically handles variable aspect ratios with intelligent patching |
| **Video Processing** | Extracts and processes video frames using Decord |
| **Audio Processing** | Handles audio tokenization and feature extraction |
| **Configurable Sampling** | Supports data ratio sampling for balanced training |
| **Conversation Preprocessing** | Handles multimodal conversation formatting |

## Class Definition

```python
class LazySupervisedDataset(Dataset):
    """
    A lazy-loading dataset for supervised fine-tuning of multimodal models.
    
    This dataset supports multiple modalities including text, images, videos, and audio.
    It implements lazy loading to efficiently handle large datasets by processing
    data samples on-demand rather than loading everything into memory at once.
    
    Args:
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for text processing
        data_args (DataArguments): Configuration arguments for data processing
        
    Example:
        >>> from vita.util.data_utils_video_audio_neg_patch import LazySupervisedDataset, DataArguments
        >>> 
        >>> data_args = DataArguments(
        ...     dataset_use="Pretrain_video",
        ...     is_multimodal=True,
        ...     image_aspect_ratio="pad",
        ...     min_dynamic_patch=1,
        ...     max_dynamic_patch=12
        ... )
        >>> 
        >>> dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
        >>> sample = dataset[0]  # Get first sample with lazy processing
    """
    
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        # Implementation details...
```

## Configuration

### DataArguments Parameters

```python
@dataclass
class DataArguments:
    # Core Parameters
    lazy_preprocess: bool = False          # Enable lazy loading for memory efficiency
    is_multimodal: bool = True            # Enable multimodal processing
    dataset_use: str = "temp"             # Dataset configuration key
    
    # Image Processing Parameters
    image_folder: Optional[str] = None    # Specific image folder override
    image_aspect_ratio: str = None        # Image aspect ratio handling ("pad", "square")
    min_dynamic_patch: int = 1            # Minimum dynamic image patches
    max_dynamic_patch: int = 12           # Maximum dynamic image patches
    use_thumbnail: bool = True            # Use thumbnail for large images
```

### Parameter Descriptions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lazy_preprocess` | bool | False | Enables lazy loading for memory efficiency |
| `is_multimodal` | bool | True | Enables processing of images, videos, and audio |
| `dataset_use` | str | "temp" | Key to identify dataset configuration from `DataConfig` |
| `image_folder` | str | None | Override for specific image folder paths |
| `image_aspect_ratio` | str | None | How to handle image aspect ratios |
| `min_dynamic_patch` | int | 1 | Minimum number of patches for dynamic image processing |
| `max_dynamic_patch` | int | 12 | Maximum number of patches for dynamic image processing |
| `use_thumbnail` | bool | True | Whether to include thumbnail versions of large images |

## Data Processing Pipeline

### 1. Dataset Loading
```python
# Configuration resolution
dataset_list = DataConfig[str(data_args.dataset_use)]

# Data loading with sampling
for dataset_config in dataset_list:
    data_ratio = dataset_config.get("data_ratio", DEFAULT_DATA_RATIO)
    data = json.load(open(dataset_config["chat_path"], "r"))
    sampled_data = random.sample(data, int(len(data) * data_ratio))
    list_data_dict += sampled_data
```

### 2. Modality Detection
The dataset automatically detects which modalities are present in each sample:

```python
# Modality detection logic
if "image" in sources[0] and "audio" not in sources[0]:
    # Image-only processing
elif "image" in sources[0] and "audio" in sources[0]:
    # Image + Audio processing
elif "video" in sources[0] and "audio" not in sources[0]:
    # Video-only processing
elif "video" in sources[0] and "audio" in sources[0]:
    # Video + Audio processing
elif "audio" in sources[0]:
    # Audio-only processing
else:
    # Text-only processing
```

### 3. Media Processing

#### Image Processing
- **Loading**: PIL Image loading with RGB conversion
- **Aspect Ratio**: Padding to square or maintaining original ratio
- **Dynamic Patching**: Intelligent patch generation based on image complexity
- **Preprocessing**: Normalization and tensor conversion

#### Video Processing
- **Frame Extraction**: Using Decord for efficient video decoding
- **Frame Sampling**: Configurable min/max frames (4-32 frames)
- **Temporal Processing**: Frame rate adjustment and sampling
- **Tensor Conversion**: Frame-to-tensor conversion

#### Audio Processing
- **File Loading**: Audio file processing with error handling
- **Feature Extraction**: Audio feature extraction and tokenization
- **Length Tracking**: Audio length tracking for model processing

## Token Processing

### `__getitem__` Implementation Details

The `__getitem__` method is the core of the LazySupervisedDataset, handling the complete data processing pipeline from raw data to model-ready tensors.

#### Method Signature
```python
def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
```

#### Input Data Structure
The method expects `self.list_data_dict[i]` to contain a dictionary with the following structure:

```python
# Example input data structure
{
    "conversations": [
        {
            "from": "human",
            "value": "What do you see in this image? <image>"
        },
        {
            "from": "gpt", 
            "value": "I can see a beautiful landscape with mountains and a lake."
        }
    ],
    "image": "path/to/image.jpg",  # or ["img1.jpg", "img2.jpg"] for multiple images
    "video": "path/to/video.mp4",  # optional
    "audio": "path/to/audio.wav",  # optional
    "set": "dataset_name",         # dataset identifier
    "id": "unique_sample_id",      # optional
    "inserted_id": None,           # optional
    "end_tag": True               # optional
}
```

#### Processing Flow by Modality

**1. Image + Audio Processing**
```python
# Detection: "image" in sources[0] and "audio" in sources[0]
if "image" in sources[0] and "audio" in sources[0]:
    # Load and process images
    image_file = self.list_data_dict[i]["image"]
    set_id = self.list_data_dict[i].get("set", None)
    
    # Handle single or multiple images
    if type(image_file) is list:
        # Multiple images processing
        image = [Image.open(os.path.join(self.folder_dict[set_id[k]], file)).convert("RGB") 
                for k, file in enumerate(image_file)]
    else:
        # Single image processing
        image = Image.open(os.path.join(self.folder_dict[set_id], image_file)).convert("RGB")
    
    # Dynamic patching and preprocessing
    image_patches, patch_num = dynamic_preprocess(image, ...)
    image = [processor.preprocess(i, return_tensors="pt")["pixel_values"][0] 
             for i in image_patches]
    
    # Load and process audio
    audio_file = self.list_data_dict[i]["audio"]
    if type(audio_file) is list:
        # Multiple audio files
        audio = []
        audio_for_llm_lens = []
        audio_length = []
        for file in audio_file:
            a, a_llm = self.data_args.audio_processor.process(
                os.path.join(AudioFolder, "audio", file)
            )
            audio.append(a)
            audio_for_llm_lens.append(a_llm)
            audio_length.append(a.shape[0])
    else:
        # Single audio file
        audio, audio_for_llm_lens = self.data_args.audio_processor.process(
            os.path.join(AudioFolder, "audio", audio_file)
        )
        audio_length = audio.shape[0]
    
    # Preprocess multimodal conversation
    sources = preprocess_multimodal(
        copy.deepcopy([e["conversations"] for e in sources]),
        self.data_args,
        patch_num=patch_num,
        audio_lens=audio_for_llm_lens,
        inserted_id=inserted_id,
    )
    
    # Tokenize conversation
    data_dict = preprocess(
        sources,
        self.tokenizer,
        has_image=True,
        has_audio=True,
        end_tag=end_tag,
        modality="image",
    )
    data_dict["audio_lengths"] = audio_length
    data_dict["audio_lengths_for_llm"] = audio_for_llm_lens
```

**2. Video + Audio Processing**
```python
# Detection: "video" in sources[0] and "audio" in sources[0]
elif "video" in sources[0] and "audio" in sources[0]:
    # Extract video frames
    video_file = self.list_data_dict[i]["video"]
    video_id = self.list_data_dict[i]["id"]
    set_id = self.list_data_dict[i].get("set", None)
    
    # Video frame extraction using Decord
    image, image_token_num = _get_rawvideo_dec(
        os.path.join(self.folder_dict[set_id], video_file),
        processor,
        max_frames=MAX_IMAGE_LENGTH,  # 32 frames
        min_frames=MIN_IMAGE_LENGTH,  # 4 frames
        image_resolution=image_size,
        image_aspect_ratio=self.data_args.image_aspect_ratio,
    )
    
    # Process audio (same as image+audio)
    audio_file = self.list_data_dict[i]["audio"]
    # ... audio processing logic ...
    
    # Preprocess with video tokens
    sources = preprocess_multimodal(
        copy.deepcopy([e["conversations"] for e in sources]),
        self.data_args,
        image_token_num=image_token_num,
        audio_lens=audio_for_llm_lens,
        inserted_id=inserted_id,
    )
    
    # Tokenize with video modality
    data_dict = preprocess(
        sources,
        self.tokenizer,
        has_image=True,
        has_audio=True,
        end_tag=end_tag,
        modality="video",
    )
```

**3. Text-Only Processing**
```python
# Detection: No media modalities present
else:
    # Simple text processing
    sources = copy.deepcopy([e["conversations"] for e in sources])
    sources = preprocess_multimodal(
        sources,
        self.data_args,
        image_token_num=0,
    )
    
    # Tokenize text-only conversation
    data_dict = preprocess(sources, self.tokenizer, has_image=False, modality="lang")
```

#### Output Assembly
```python
# Extract single sample from batch format
if isinstance(i, int):
    if "audio" in self.list_data_dict[i]:
        data_dict = dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["labels"][0],
            audio_lengths=data_dict["audio_lengths"],
            audio_lengths_for_llm=data_dict["audio_lengths_for_llm"],
        )
    else:
        data_dict = dict(
            input_ids=data_dict["input_ids"][0], 
            labels=data_dict["labels"][0]
        )

# Add media tensors
if "image" in self.list_data_dict[i] or "video" in self.list_data_dict[i]:
    data_dict["image"] = image
elif self.data_args.is_multimodal:
    # Zero tensor for multimodal models without media
    crop_size = self.data_args.image_processor.crop_size
    data_dict["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])

if "audio" in self.list_data_dict[i]:
    data_dict["audio"] = audio
elif self.data_args.is_multimodal:
    # Zero tensor for multimodal models without audio
    data_dict["audio"] = torch.zeros(400, 80)
    data_dict["audio_lengths"] = 400
    data_dict["audio_lengths_for_llm"] = 60

return data_dict
```

#### Key Processing Steps

1. **Sample Retrieval**: `sources = self.list_data_dict[i]`
2. **Modality Detection**: Check for presence of image, video, audio keys
3. **Media Loading**: Load and process media files based on detected modalities
4. **Dynamic Processing**: Apply dynamic patching, frame extraction, audio processing
5. **Conversation Preprocessing**: Handle special tokens and conversation formatting
6. **Tokenization**: Convert text to token IDs with proper masking
7. **Output Assembly**: Combine all components into final dictionary

#### Error Handling
```python
# File loading with error handling
try:
    a, a_llm = self.data_args.audio_processor.process(
        os.path.join(audio_folder, "audio", file)
    )
except:
    print(f"File {os.path.join(audio_folder, 'audio', file)} not OK!!!!!")

# Validation checks
assert len(audio_file) > 0, "audio_file为列表时不能为空"
assert audio_file, "audio_file不能为空"
```

#### Performance Optimizations
- **Lazy Loading**: Media files loaded only when needed
- **Batch Processing**: Efficient tensor operations
- **Memory Management**: Proper tensor cleanup
- **Error Recovery**: Graceful handling of corrupted files

#### Numerical Output Examples

**Example 1: Image + Text Sample**
```python
# Input data
input_data = {
    "conversations": [
        {"from": "human", "value": "Describe this <image> in detail."},
        {"from": "gpt", "value": "This is a cat sitting on a windowsill."}
    ],
    "image": "cat_image.jpg",
    "set": "coco_dataset"
}

# Output tensor shapes and values
output = {
    "input_ids": torch.tensor([
        1,        # <s> token
        151644,   # "Describe"
        445,      # "this"
        151643,   # <image> token (patch 1)
        151643,   # <image> token (patch 2)
        151643,   # <image> token (patch 3)
        151643,   # <image> token (patch 4)
        338,      # "in"
        151644,   # "detail"
        2         # </s> token
    ]),  # Shape: [10]
    
    "labels": torch.tensor([
        -100,     # <s> token (masked)
        -100,     # "Describe" (masked)
        -100,     # "this" (masked)
        -100,     # <image> token (masked)
        -100,     # <image> token (masked)
        -100,     # <image> token (masked)
        -100,     # <image> token (masked)
        -100,     # "in" (masked)
        -100,     # "detail" (masked)
        2         # </s> token (not masked)
    ]),  # Shape: [10]
    
    "image": torch.tensor([
        # Patch 1: [3, 384, 384]
        [[[0.1, 0.2, 0.3, ...], [0.4, 0.5, 0.6, ...], ...], ...],
        # Patch 2: [3, 384, 384]
        [[[0.7, 0.8, 0.9, ...], [0.1, 0.2, 0.3, ...], ...], ...],
        # Patch 3: [3, 384, 384]
        [[[0.4, 0.5, 0.6, ...], [0.7, 0.8, 0.9, ...], ...], ...],
        # Patch 4: [3, 384, 384]
        [[[0.2, 0.3, 0.4, ...], [0.5, 0.6, 0.7, ...], ...], ...]
    ])  # Shape: [4, 3, 384, 384] - 4 patches, 3 channels, 384x384 pixels
}
```

**Example 2: Video + Audio Sample**
```python
# Input data
input_data = {
    "conversations": [
        {"from": "human", "value": "What's happening in this <video> with <audio>?"},
        {"from": "gpt", "value": "A person is playing guitar and singing."}
    ],
    "video": "music_video.mp4",
    "audio": "music_audio.wav",
    "set": "music_dataset"
}

# Output tensor shapes and values
output = {
    "input_ids": torch.tensor([
        1,        # <s> token
        151644,   # "What's"
        445,      # "happening"
        338,      # "in"
        151643,   # <image> token (frame 1)
        151643,   # <image> token (frame 2)
        # ... 14 more frame tokens ...
        151643,   # <image> token (frame 16)
        151643,   # <audio> token
        2         # </s> token
    ]),  # Shape: [20] - 3 text + 16 video + 1 audio + 2 special tokens
    
    "labels": torch.tensor([
        -100,     # <s> token (masked)
        -100,     # "What's" (masked)
        -100,     # "happening" (masked)
        -100,     # "in" (masked)
        -100,     # <image> token (masked)
        -100,     # <image> token (masked)
        # ... 14 more masked frame tokens ...
        -100,     # <image> token (masked)
        -100,     # <audio> token (masked)
        2         # </s> token (not masked)
    ]),  # Shape: [20]
    
    "image": torch.tensor([
        # Frame 1: [3, 384, 384]
        [[[0.1, 0.2, 0.3, ...], [0.4, 0.5, 0.6, ...], ...], ...],
        # Frame 2: [3, 384, 384]
        [[[0.7, 0.8, 0.9, ...], [0.1, 0.2, 0.3, ...], ...], ...],
        # ... 14 more frames ...
        # Frame 16: [3, 384, 384]
        [[[0.4, 0.5, 0.6, ...], [0.7, 0.8, 0.9, ...], ...], ...]
    ]),  # Shape: [16, 3, 384, 384] - 16 frames, 3 channels, 384x384 pixels
    
    "audio": torch.tensor([
        [0.1, 0.2, 0.3, ..., 0.8],  # Time step 1: 80 audio features
        [0.2, 0.3, 0.4, ..., 0.9],  # Time step 2: 80 audio features
        # ... 798 more time steps ...
        [0.3, 0.4, 0.5, ..., 0.1]   # Time step 800: 80 audio features
    ]),  # Shape: [800, 80] - 800 time steps, 80 audio features
    
    "audio_lengths": 800,           # Original audio length
    "audio_lengths_for_llm": 100    # Audio length for LLM processing
}
```

**Example 3: Text-Only Sample**
```python
# Input data
input_data = {
    "conversations": [
        {"from": "human", "value": "What is the capital of France?"},
        {"from": "gpt", "value": "The capital of France is Paris."}
    ]
}

# Output tensor shapes and values
output = {
    "input_ids": torch.tensor([
        1,        # <s> token
        151644,   # "What"
        445,      # "is"
        338,      # "the"
        151644,   # "capital"
        445,      # "of"
        151644,   # "France"
        2         # </s> token
    ]),  # Shape: [8] - 6 text tokens + 2 special tokens
    
    "labels": torch.tensor([
        -100,     # <s> token (masked)
        -100,     # "What" (masked)
        -100,     # "is" (masked)
        -100,     # "the" (masked)
        -100,     # "capital" (masked)
        -100,     # "of" (masked)
        -100,     # "France" (masked)
        2         # </s> token (not masked)
    ]),  # Shape: [8]
    
    "image": torch.zeros(3, 384, 384),  # Zero tensor for multimodal models
    "audio": torch.zeros(400, 80)       # Zero tensor for multimodal models
}
```

**Example 4: Multiple Images Sample**
```python
# Input data
input_data = {
    "conversations": [
        {"from": "human", "value": "Compare these <image> and <image>."},
        {"from": "gpt", "value": "The first image shows a cat, the second shows a dog."}
    ],
    "image": ["cat_image.jpg", "dog_image.jpg"],
    "set": ["coco_dataset", "coco_dataset"]
}

# Output tensor shapes and values
output = {
    "input_ids": torch.tensor([
        1,        # <s> token
        151644,   # "Compare"
        445,      # "these"
        151643,   # <image> token (cat image, patch 1)
        151643,   # <image> token (cat image, patch 2)
        151643,   # <image> token (dog image, patch 1)
        151643,   # <image> token (dog image, patch 2)
        151643,   # <image> token (dog image, patch 3)
        2         # </s> token
    ]),  # Shape: [9] - 2 text + 5 image tokens + 2 special tokens
    
    "labels": torch.tensor([
        -100,     # <s> token (masked)
        -100,     # "Compare" (masked)
        -100,     # "these" (masked)
        -100,     # <image> token (masked)
        -100,     # <image> token (masked)
        -100,     # <image> token (masked)
        -100,     # <image> token (masked)
        -100,     # <image> token (masked)
        2         # </s> token (not masked)
    ]),  # Shape: [9]
    
    "image": torch.tensor([
        # Cat image patches
        [[[0.1, 0.2, 0.3, ...], [0.4, 0.5, 0.6, ...], ...], ...],  # Patch 1
        [[[0.7, 0.8, 0.9, ...], [0.1, 0.2, 0.3, ...], ...], ...],  # Patch 2
        # Dog image patches
        [[[0.4, 0.5, 0.6, ...], [0.7, 0.8, 0.9, ...], ...], ...],  # Patch 1
        [[[0.2, 0.3, 0.4, ...], [0.5, 0.6, 0.7, ...], ...], ...],  # Patch 2
        [[[0.8, 0.9, 0.1, ...], [0.2, 0.3, 0.4, ...], ...], ...]   # Patch 3
    ])  # Shape: [5, 3, 384, 384] - 5 total patches (2+3), 3 channels, 384x384 pixels
}
```

**Example 5: Real Token Sequence (from data.json)**
```python
# Actual tokenized sequence from production data
output = {
    "input_ids": torch.tensor([
        # Text tokens (first 10)
        33975, 25, 3555, 653, 498, 1490, 12482, 304, 419, 2168,
        # ... more text tokens ...
        # Image tokens (151643 repeated 128 times)
        151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643,
        151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643,
        # ... 112 more image tokens ...
        151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643
    ]),  # Shape: [256] - 128 text + 128 image tokens
    
    "labels": torch.tensor([
        # Same structure as input_ids but with masking
        33975, 25, 3555, 653, 498, 1490, 12482, 304, 419, 2168,
        # ... more text tokens (not masked) ...
        # Image tokens (all masked with -100)
        -100, -100, -100, -100, -100, -100, -100, -100,
        -100, -100, -100, -100, -100, -100, -100, -100,
        # ... 112 more masked image tokens ...
        -100, -100, -100, -100, -100, -100, -100, -100
    ]),  # Shape: [256]
    
    "attention_mask": torch.tensor([
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        # ... 128 ones for text tokens ...
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        # ... 112 more ones for image tokens ...
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        # ... 128 zeros for padding ...
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ])  # Shape: [384] - 256 actual tokens + 128 padding
}
```

#### Tensor Shape Summary

| Modality | input_ids Shape | image Shape | audio Shape | audio_lengths |
|----------|----------------|-------------|-------------|---------------|
| **Text Only** | `[seq_len]` | `[3, 384, 384]` (zeros) | `[400, 80]` (zeros) | `400` |
| **Image Only** | `[seq_len]` | `[num_patches, 3, 384, 384]` | `[400, 80]` (zeros) | `400` |
| **Video Only** | `[seq_len]` | `[num_frames, 3, 384, 384]` | `[400, 80]` (zeros) | `400` |
| **Audio Only** | `[seq_len]` | `[3, 384, 384]` (zeros) | `[audio_len, 80]` | `audio_len` |
| **Image + Audio** | `[seq_len]` | `[num_patches, 3, 384, 384]` | `[audio_len, 80]` | `audio_len` |
| **Video + Audio** | `[seq_len]` | `[num_frames, 3, 384, 384]` | `[audio_len, 80]` | `audio_len` |

#### Key Numerical Insights

1. **Token ID 151643**: Used for all special tokens (`<image>`, `<video>`, `<audio>`)
2. **Image Patches**: Typically 1-12 patches per image based on aspect ratio
3. **Video Frames**: 4-32 frames per video (configurable)
4. **Audio Features**: 80-dimensional features per time step
5. **Label Masking**: Instruction and media tokens masked with -100
6. **Zero Tensors**: Used for missing modalities in multimodal models

### Special Token Constants
```python
# From vita/constants.py
DEFAULT_IMAGE_TOKEN = "<image>"    # Image placeholder token
DEFAULT_VIDEO_TOKEN = "<video>"    # Video placeholder token  
DEFAULT_AUDIO_TOKEN = "<audio>"    # Audio placeholder token
IMAGE_TOKEN_INDEX = -200           # Special index for image tokens
AUDIO_TOKEN_INDEX = -500           # Special index for audio tokens
IGNORE_INDEX = -100                # Index for ignored tokens in loss calculation
```

### Token Processing Flow

#### 1. Token Detection and Normalization
```python
# Input conversation with tokens
conversation = [
    {"from": "human", "value": "What do you see in this <image>?"},
    {"from": "gpt", "value": "I can see a cat sitting on a windowsill."}
]

# After preprocessing_multimodal()
conversation = [
    {"from": "human", "value": "<image>\nWhat do you see in this?"},
    {"from": "gpt", "value": "I can see a cat sitting on a windowsill."}
]
```

#### 2. Dynamic Token Replacement
The system replaces placeholder tokens with actual token sequences based on media content:

**Image Token Processing:**
```python
# Single image with dynamic patching
patch_num = [4]  # 4 patches generated from image
replace_token = "<image>" * 4  # "<image><image><image><image>"

# Multiple images
patch_num = [2, 3]  # 2 patches for first image, 3 for second
```

**Video Token Processing:**
```python
# Video with 16 frames
image_token_num = 16
vid_replace_token = "<image>" * 16  # 16 image tokens for video frames
```

**Audio Token Processing:**
```python
# Audio with specific length
audio_lens = 100  # 100 audio tokens
audio_replace_token = "<audio>"  # Single audio token placeholder
```

#### 3. Token Index Mapping
During tokenization, special tokens are mapped to specific indices:

```python
# Token ID mapping
token_mapping = {
    "<image>": 151643,  # Actual token ID for image tokens
    "<video>": 151643,  # Same as image (video uses image tokens)
    "<audio>": 151643,  # Same as image (audio uses image tokens)
    "☞": 151643,        # Special prefix for audio responses
    "☟": 151643,        # Special prefix for inserted responses  
    "☜": 151643,        # Special prefix for regular responses
}
```

#### 4. Label Masking Strategy
Labels are masked to prevent the model from learning to predict instruction tokens:

```python
# Example with image tokens
input_ids = [1, 151644, 445, 338, 151643, 151643, 151643, 151643, 2]
labels =    [-100, -100, -100, -100, -100, -100, -100, -100, 2]

# Only the response tokens are used for loss calculation
# Instruction tokens and image tokens are masked with -100
```

## Supported Modalities

### 1. Text-Only Data
- Pure text conversations without any media
- Processed using standard tokenization
- Minimal memory footprint

### 2. Image Data
- Single or multiple images per sample
- Dynamic patching for variable aspect ratios
- Support for different image processing modes
- Aspect ratio handling (pad, square, original)

### 3. Video Data
- Video frame extraction using Decord
- Configurable frame sampling (min/max frames)
- Video token processing with temporal information
- Frame rate adjustment and sampling

### 4. Audio Data
- Audio file processing and tokenization
- Support for multiple audio files per sample
- Audio length tracking for model processing
- Feature extraction and normalization

### 5. Multimodal Combinations
- Image + Audio
- Video + Audio
- Any combination of the above modalities
- Intelligent processing based on detected modalities

## API Reference

### Constructor
```python
def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
    """
    Initialize the LazySupervisedDataset.
    
    Args:
        tokenizer: Tokenizer for text processing
        data_args: Configuration arguments for data processing
    """
```

### Methods

#### `__len__(self) -> int`
Returns the total number of samples in the dataset.

#### `__getitem__(self, i: int) -> Dict[str, torch.Tensor]`
Retrieves and processes a single data sample.

**Parameters:**
- `i` (int): Index of the sample to retrieve

**Returns:**
```python
{
    "input_ids": torch.Tensor,           # Shape: [seq_len]
    "labels": torch.Tensor,              # Shape: [seq_len]
    "image": torch.Tensor,               # Shape: [num_patches, 3, H, W] (if present)
    "video": torch.Tensor,               # Shape: [num_frames, 3, H, W] (if present)
    "audio": torch.Tensor,               # Shape: [audio_len, 80] (if present)
    "audio_lengths": int,                # Original audio length (if present)
    "audio_lengths_for_llm": int,        # Audio length for LLM processing (if present)
}
```

#### `modality_lengths` (property)
Returns a list of sequence lengths for each sample, with special handling for multimodal samples:
- Positive values for samples with images/videos
- Negative values for text-only samples

## Usage Examples

### Basic Usage
```python
from vita.util.data_utils_video_audio_neg_patch import LazySupervisedDataset, DataArguments

# Configure data arguments
data_args = DataArguments(
    dataset_use="Pretrain_video",
    is_multimodal=True,
    image_aspect_ratio="pad",
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    use_thumbnail=True
)

# Create dataset
dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)

# Get a sample
sample = dataset[0]
print(f"Input IDs shape: {sample['input_ids'].shape}")
print(f"Labels shape: {sample['labels'].shape}")
if 'image' in sample:
    print(f"Image shape: {sample['image'].shape}")
```

### Training Integration
```python
from torch.utils.data import DataLoader
from vita.util.data_utils_video_audio_neg_patch import DataCollatorForSupervisedDataset

# Create data collator
data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

# Create data loader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=data_collator
)

# Training loop
for batch in dataloader:
    # batch contains batched input_ids, labels, images, etc.
    pass
```

### Configuration Examples

#### Video Training Configuration
```python
data_args = DataArguments(
    dataset_use="VideoTraining",
    is_multimodal=True,
    image_aspect_ratio="pad",
    min_dynamic_patch=1,
    max_dynamic_patch=8,
    use_thumbnail=False
)
```

#### Audio-Visual Training Configuration
```python
data_args = DataArguments(
    dataset_use="AudioVisual",
    is_multimodal=True,
    image_aspect_ratio="square",
    min_dynamic_patch=2,
    max_dynamic_patch=6,
    use_thumbnail=True
)
```

## Performance Considerations

### Memory Efficiency
- **Lazy Loading**: Prevents loading all data into memory at once
- **Media Processing**: Files are processed on-demand
- **Dynamic Patching**: Reduces memory usage for large images
- **Batch Processing**: Efficient batching with DataCollator

### Processing Speed
- **Video Processing**: Can be slow due to frame extraction
- **Audio Processing**: Depends on file size and format
- **Image Processing**: Generally fast with dynamic patching
- **Tokenization**: Optimized for large sequences

### Best Practices
1. **Batch Size**: Use appropriate batch sizes based on available memory
2. **Data Workers**: Consider using data workers for parallel processing
3. **Memory Monitoring**: Monitor memory usage during training
4. **Thumbnail Mode**: Use thumbnail mode for very large images when appropriate
5. **Dataset Sampling**: Use data_ratio for balanced training

### Error Handling
The dataset includes error handling for:
- Missing media files
- Corrupted data samples
- Invalid file formats
- Tokenization errors

Common error messages:
- `FileNotFoundError`: Missing media files
- `WARNING: tokenization mismatch`: Tokenization length mismatches
- `audio_file为列表时不能为空`: Empty audio file list

## Dependencies

Required packages:
- `torch`
- `transformers`
- `PIL` (Pillow)
- `numpy`
- `decord` (for video processing)
- `vita` (internal package)

## Related Classes

- `DataCollatorForSupervisedDataset`: Handles batching of dataset samples
- `DataArguments`: Configuration dataclass
- `preprocess_multimodal()`: Multimodal conversation preprocessing
- `preprocess()`: Text tokenization and formatting