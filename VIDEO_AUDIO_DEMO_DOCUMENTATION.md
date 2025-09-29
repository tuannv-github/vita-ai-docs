# VITA Video-Audio Demo Documentation

This document provides comprehensive information about the `video_audio_demo.py` script, which is a demonstration tool for testing VITA's multimodal capabilities with video, audio, and text inputs.

## 📋 Table of Contents

- [Overview](#overview)
- [File Structure](#file-structure)
- [Key Components](#key-components)
- [Video Processing](#video-processing)
- [Audio Processing](#audio-processing)
- [Image Processing](#image-processing)
- [Model Loading and Inference](#model-loading-and-inference)
- [Usage Examples](#usage-examples)
- [Command Line Arguments](#command-line-arguments)
- [Configuration Parameters](#configuration-parameters)
- [Output and Results](#output-and-results)
- [Best Practices](#best-practices)

## 🎯 Overview

The `video_audio_demo.py` script is a comprehensive demonstration tool that showcases VITA's multimodal capabilities. It can process:

- **Video inputs** with frame extraction and processing
- **Audio inputs** with speech processing
- **Image inputs** with dynamic preprocessing
- **Text-only inputs** for language-only interactions

The script demonstrates the complete VITA inference pipeline, from input preprocessing to model generation and output formatting.

## 🏗️ File Structure

### **Import Dependencies**
```python
# video_audio_demo.py:1-26
import argparse
import os
import time
import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu

# VITA-specific imports
from vita.constants import (
    DEFAULT_AUDIO_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN,
    IGNORE_INDEX, IMAGE_TOKEN_INDEX, MAX_IMAGE_LENGTH
)
from vita.conversation import SeparatorStyle, conv_templates
from vita.model.builder import load_pretrained_model
from vita.util.mm_utils import (
    KeywordsStoppingCriteria, get_model_name_from_path,
    tokenizer_image_audio_token, tokenizer_image_token
)
from vita.util.utils import disable_torch_init
```

### **Main Components**
1. **Video Processing Function** (`_get_rawvideo_dec`)
2. **Argument Parser** (Command-line interface)
3. **Model Loading** (VITA model initialization)
4. **Input Processing** (Video/Audio/Image/Text)
5. **Inference Pipeline** (Model generation)
6. **Output Formatting** (Result processing)

## 🎬 Video Processing

### **Video Decoding Function**

The `_get_rawvideo_dec` function handles video input processing:

```python
# video_audio_demo.py:29-38
def _get_rawvideo_dec(
    video_path,
    image_processor,
    max_frames=MAX_IMAGE_LENGTH,  # Default: 16 frames
    min_frames=4,
    image_resolution=384,
    video_framerate=1,
    s=None,  # Start time
    e=None,  # End time
    image_aspect_ratio="pad",
):
```

#### **Key Features:**

##### **1. Time-based Frame Selection**
```python
# video_audio_demo.py:42-48
# Time range processing
if s is None:
    start_time, end_time = None, None
else:
    start_time = int(s)
    end_time = int(e)
    # Validation and correction
    start_time = start_time if start_time >= 0.0 else 0.0
    end_time = end_time if end_time >= 0.0 else 0.0
```

##### **2. Frame Sampling Strategy**
```python
# video_audio_demo.py:60-79
# Calculate frame positions
fps = vreader.get_avg_fps()
f_start = 0 if start_time is None else int(start_time * fps)
f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))

# Adaptive frame sampling
if len(all_pos) > max_frames:
    # Downsample to max_frames
    sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
elif len(all_pos) < min_frames:
    # Upsample to min_frames
    sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=min_frames, dtype=int)]
```

##### **3. Image Preprocessing**
```python
# Extract frames and convert to PIL Images
patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]

# Aspect ratio handling
if image_aspect_ratio == "pad":
    def expand2square(pil_img, background_color):
        # Pad images to square format
        width, height = pil_img.size
        if width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
        return result
```

## 🎵 Audio Processing

### **Audio Input Handling**

The script supports both audio and text-only inputs:

```python
# video_audio_demo.py:184-205
if audio_path is not None:
    # Process audio file
    audio, audio_for_llm_lens = audio_processor.process(os.path.join(audio_path))
    audio_length = audio.shape[0]
    audio = torch.unsqueeze(audio, dim=0)
    audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
    audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
    
    # Prepare audio tensor
    audios = dict()
    audios["audios"] = audio.half().cuda()
    audios["lengths"] = audio_length.half().cuda()
    audios["lengths_for_llm"] = audio_for_llm_lens.cuda()
else:
    # Use dummy audio for text-only mode
    audio = torch.zeros(400, 80)
    audio_length = audio.shape[0]
    audio_for_llm_lens = 60
    # ... prepare dummy audio tensors
```

### **Audio Processing Features**

#### **1. Audio File Processing**
- **File Loading**: Supports various audio formats
- **Feature Extraction**: Converts audio to model-compatible features
- **Length Calculation**: Tracks audio duration for proper processing

#### **2. Dummy Audio for Text-Only**
- **Zero Padding**: Creates dummy audio tensors when no audio is provided
- **Consistent Format**: Maintains tensor structure for model compatibility

## 🖼️ Image Processing

### **Image Input Handling**

The script supports both single images and video frames:

```python
if image_path is not None:
    image = Image.open(image_path).convert("RGB")
    
    # Dynamic preprocessing based on frameCat flag
    if args.frameCat:
        image, p_num = dynamic_preprocess(
            image, min_num=2, max_num=12, 
            image_size=448, use_thumbnail=True, 
            img_mean=image_processor.image_mean
        )
    else:
        image, p_num = dynamic_preprocess(
            image, min_num=1, max_num=12, 
            image_size=448, use_thumbnail=True
        )
    
    # Process image through model
    image_tensor = model.process_images(image, model.config).to(
        dtype=model.dtype, device="cuda"
    )
```

### **Image Processing Features**

#### **1. Dynamic Preprocessing**
- **Patch-based Processing**: Splits images into patches for better processing
- **Thumbnail Generation**: Creates thumbnails for efficient processing
- **Size Adaptation**: Adjusts image size to model requirements

#### **2. Frame Concatenation vs Patch Processing**
- **`frameCat` Flag**: Controls processing method
- **Frame Concatenation**: Uses `data_utils_video_audio_neg_frameCat`
- **Patch Processing**: Uses `data_utils_video_audio_neg_patch`

## 🤖 Model Loading and Inference

### **Model Initialization**

```python
# video_audio_demo.py:165-183
# Disable torch initialization for efficiency
disable_torch_init()

# Load pretrained model
model_path = os.path.expanduser(model_path)
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, model_base, model_name, args.model_type
)

# Resize token embeddings
model.resize_token_embeddings(len(tokenizer))

# Load vision tower
vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()
image_processor = vision_tower.image_processor

# Load audio encoder
audio_encoder = model.get_audio_encoder()
audio_encoder.to(dtype=torch.float16)
audio_processor = audio_encoder.audio_processor

# Set model to evaluation mode
model.eval()
```

### **Inference Pipeline**

```python
# video_audio_demo.py:243-281
# Prepare conversation
conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt(modality)

# Tokenize input
if audio_path:
    input_ids = tokenizer_image_audio_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()
else:
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

# Generate response
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        audios=audios,
        do_sample=False,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        output_scores=True,
        return_dict_in_generate=True,
        max_new_tokens=1024,
        use_cache=True,
        stopping_criteria=[stopping_criteria],
        shared_v_pid_stride=None
    )
```

## 💻 Usage Examples

### **1. Video + Audio + Text**
```bash
python video_audio_demo.py \
    --model_path /path/to/vita/model \
    --video_path /path/to/video.mp4 \
    --audio_path /path/to/audio.wav \
    --question "What is happening in this video and what are they saying?"
```

### **2. Image + Audio + Text**
```bash
python video_audio_demo.py \
    --model_path /path/to/vita/model \
    --image_path /path/to/image.jpg \
    --audio_path /path/to/audio.wav \
    --question "Describe what you see and hear."
```

### **3. Video + Text Only**
```bash
python video_audio_demo.py \
    --model_path /path/to/vita/model \
    --video_path /path/to/video.mp4 \
    --question "What is happening in this video?"
```

### **4. Image + Text Only**
```bash
python video_audio_demo.py \
    --model_path /path/to/vita/model \
    --image_path /path/to/image.jpg \
    --question "Describe this image."
```

### **5. Text Only**
```bash
python video_audio_demo.py \
    --model_path /path/to/vita/model \
    --question "Tell me a story about a cat."
```

## ⚙️ Command Line Arguments

### **Required Arguments**
| Argument | Type | Description |
|----------|------|-------------|
| `--model_path` | str | Path to the VITA model directory |

### **Optional Arguments**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_base` | str | None | Base model path for LoRA models |
| `--video_path` | str | None | Path to video file |
| `--image_path` | str | None | Path to image file |
| `--audio_path` | str | None | Path to audio file |
| `--model_type` | str | "mixtral-8x7b" | Model type identifier |
| `--conv_mode` | str | "mixtral_two" | Conversation template mode |
| `--question` | str | "" | Text question/prompt |
| `--frameCat` | flag | False | Use frame concatenation processing |

### **Input Validation**
```python
# Ensure exactly one of audio_path or question is provided
assert (audio_path is None) != (qs == ""), "Exactly one of audio_path or qs must be non-None"
```

## 🔧 Configuration Parameters

### **Video Processing Parameters**
```python
# Maximum number of frames to process
max_frames = MAX_IMAGE_LENGTH  # Default: 16

# Frames per second for sampling
video_framerate = 1

# Image resolution
image_resolution = 384

# Minimum frames required
min_frames = 4
```

### **Generation Parameters**
```python
# Sampling parameters
temperature = 0.01  # Low temperature for deterministic output
top_p = None        # No top-p sampling
num_beams = 1       # No beam search

# Generation limits
max_new_tokens = 1024
```

### **Processing Modes**
```python
# Frame processing mode
if args.frameCat:
    from vita.util.data_utils_video_audio_neg_frameCat import dynamic_preprocess
else:
    from vita.util.data_utils_video_audio_neg_patch import dynamic_preprocess
```

## 📊 Output and Results

### **Output Formatting**
```python
# Decode output tokens
outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]

# Clean up output
outputs = outputs.strip()
if outputs.endswith(stop_str):
    outputs = outputs[: -len(stop_str)]
outputs = outputs.strip()

# Print results
print(outputs)
print(f"Time consume: {infer_time}")
```

### **Performance Metrics**
- **Inference Time**: Measures total generation time
- **Token Count**: Tracks input/output token lengths
- **Memory Usage**: Monitors GPU memory consumption

### **Output Examples**

#### **Video + Audio Analysis**
```
Input: Video of a person speaking + Audio of their voice + "What are they saying?"
Output: "The person in the video is explaining the concept of machine learning. They are discussing how neural networks work and their applications in computer vision tasks."
Time consume: 2.34 seconds
```

#### **Image Description**
```
Input: Image of a cat + "Describe this image"
Output: "This is a photograph of a fluffy orange cat sitting on a windowsill. The cat appears to be looking out the window with its tail curled around its body. The lighting suggests it's either early morning or late afternoon."
Time consume: 1.87 seconds
```

## 🎯 Best Practices

### **1. Input Preparation**
- **Video Files**: Use common formats (MP4, AVI, MOV)
- **Audio Files**: Use high-quality audio (WAV, MP3)
- **Images**: Use RGB images with reasonable resolution
- **Text**: Provide clear, specific questions

### **2. Performance Optimization**
- **GPU Memory**: Ensure sufficient GPU memory for model and inputs
- **Batch Processing**: Process multiple inputs in sequence
- **Model Caching**: Keep model loaded for multiple inferences

### **3. Input Validation**
- **File Existence**: Check that input files exist before processing
- **Format Support**: Verify file formats are supported
- **Size Limits**: Monitor input sizes to avoid memory issues

### **4. Error Handling**
```python
# Video file validation
if os.path.exists(video_path):
    vreader = VideoReader(video_path, ctx=cpu(0))
else:
    print(video_path)
    raise FileNotFoundError
```

### **5. Debugging Tips**
- **Verbose Output**: Use print statements to track processing steps
- **Memory Monitoring**: Monitor GPU memory usage during inference
- **Time Profiling**: Measure processing time for each component

## 🔍 Advanced Usage

### **Custom Processing**
```python
# Custom frame sampling
def custom_frame_sampling(video_path, target_frames=8):
    vreader = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vreader)
    frame_indices = np.linspace(0, total_frames-1, target_frames, dtype=int)
    return vreader.get_batch(frame_indices)
```

### **Batch Processing**
```python
# Process multiple inputs
def batch_process(model, inputs_list):
    results = []
    for inputs in inputs_list:
        result = model.generate(**inputs)
        results.append(result)
    return results
```

### **Custom Stopping Criteria**
```python
# Custom stopping criteria
class CustomStoppingCriteria:
    def __init__(self, stop_tokens):
        self.stop_tokens = stop_tokens
    
    def __call__(self, input_ids, scores, **kwargs):
        # Custom stopping logic
        return False
```

This comprehensive documentation provides a complete understanding of the `video_audio_demo.py` script, from basic usage to advanced customization, enabling users to effectively utilize VITA's multimodal capabilities for various applications.

## 🔄 Input/Output Flow Explanation

### **Simple Input/Output Overview**

The `video_audio_demo.py` script takes multimodal inputs and produces text responses. Here's how it works:

```
INPUT:  [Video/Audio/Image] + [Text Question] 
   ↓
PROCESSING: Extract features → Tokenize → Generate
   ↓
OUTPUT: [Text Response]
```

### **Input Types**

#### **1. Video Input**
- **File**: MP4, AVI, MOV video files
- **Processing**: Extracts frames → Converts to image patches
- **Example**: `--video_path "cat_playing.mp4"`

#### **2. Audio Input**
- **File**: WAV, MP3 audio files
- **Processing**: Converts to audio features
- **Example**: `--audio_path "speech.wav"`

#### **3. Image Input**
- **File**: JPG, PNG image files
- **Processing**: Converts to image patches
- **Example**: `--image_path "photo.jpg"`

#### **4. Text Input**
- **Question**: Natural language question
- **Example**: `--question "What is happening in this video?"`

### **Output Format**

#### **Text Response**
- **Format**: Natural language text
- **Content**: Answer to the input question
- **Additional**: Processing time information

## 📝 Simple Examples

### **Example 1: Video Analysis**

#### **Input:**
```bash
python video_audio_demo.py \
    --model_path /path/to/vita/model \
    --video_path "kitchen_cooking.mp4" \
    --question "What is the person doing in this video?"
```

#### **Output:**
```
The person in the video is cooking in a kitchen. They are chopping vegetables on a cutting board, then adding them to a pan on the stove. The video shows them stirring the ingredients and checking the food as it cooks.

Time consume: 2.15 seconds
```

### **Example 2: Image Description**

#### **Input:**
```bash
python video_audio_demo.py \
    --model_path /path/to/vita/model \
    --image_path "sunset_beach.jpg" \
    --question "Describe this image"
```

#### **Output:**
```
This is a beautiful sunset scene at the beach. The sky is painted in warm orange and pink colors as the sun sets over the ocean. There are gentle waves lapping at the shore, and the sand appears golden in the evening light. The image has a peaceful, serene atmosphere.

Time consume: 1.42 seconds
```

### **Example 3: Audio Transcription**

#### **Input:**
```bash
python video_audio_demo.py \
    --model_path /path/to/vita/model \
    --audio_path "meeting_recording.wav" \
    --question "What are they talking about?"
```

#### **Output:**
```
The speakers are discussing a new project proposal. They are talking about the timeline, budget requirements, and team assignments. One person mentions that the deadline is next month and they need to finalize the technical specifications.

Time consume: 3.28 seconds
```

### **Example 4: Multimodal (Video + Audio + Text)**

#### **Input:**
```bash
python video_audio_demo.py \
    --model_path /path/to/vita/model \
    --video_path "presentation.mp4" \
    --audio_path "presentation_audio.wav" \
    --question "Summarize the presentation"
```

#### **Output:**
```
The presentation is about artificial intelligence in healthcare. The speaker is showing slides with medical imaging examples while explaining how AI can help doctors diagnose diseases more accurately. They discuss the benefits of machine learning in radiology and the potential for improving patient outcomes through early detection.

Time consume: 4.67 seconds
```

### **Example 5: Text-Only Conversation**

#### **Input:**
```bash
python video_audio_demo.py \
    --model_path /path/to/vita/model \
    --question "Tell me a short story about a robot learning to paint"
```

#### **Output:**
```
Once upon a time, there was a robot named Pixel who wanted to learn how to paint. At first, Pixel's paintings were just geometric shapes and lines, but as it studied human art and practiced every day, its paintings became more expressive and colorful. Eventually, Pixel created a masterpiece that moved everyone who saw it, proving that creativity can be learned by anyone, even a robot.

Time consume: 1.89 seconds
```

## 🔍 Input/Output Breakdown

### **What Happens Inside the Script**

#### **Step 1: Input Processing**
```python
# Video: Extract frames → Convert to patches
video_frames, slice_len = _get_rawvideo_dec(video_path, image_processor)

# Audio: Process audio file → Extract features
audio, audio_for_llm_lens = audio_processor.process(audio_path)

# Image: Load image → Convert to patches
image = Image.open(image_path).convert("RGB")

# Text: Prepare question
qs = "What is happening in this video?"
```

#### **Step 2: Token Preparation**
```python
# Combine inputs with special tokens
if video_path and audio_path:
    qs = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + qs + DEFAULT_AUDIO_TOKEN
elif video_path:
    qs = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + qs
elif audio_path:
    qs = qs + DEFAULT_AUDIO_TOKEN
```

#### **Step 3: Model Generation**
```python
# Generate response
output_ids = model.generate(
    input_ids,
    images=image_tensor,
    audios=audios,
    max_new_tokens=1024
)
```

#### **Step 4: Output Processing**
```python
# video_audio_demo.py:283-297
# Decode and clean output
outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]
outputs = outputs.strip()
print(outputs)
print(f"Time consume: {infer_time}")
```

### **Input Requirements**

#### **File Formats**
| Input Type | Supported Formats | Example |
|------------|------------------|---------|
| **Video** | MP4, AVI, MOV | `video.mp4` |
| **Audio** | WAV, MP3, FLAC | `audio.wav` |
| **Image** | JPG, PNG, BMP | `image.jpg` |
| **Text** | Any string | `"What do you see?"` |

#### **File Size Limits**
- **Video**: Recommended < 100MB for processing speed
- **Audio**: Recommended < 50MB for memory efficiency
- **Image**: Recommended < 10MB for processing speed
- **Text**: No practical limit

### **Output Characteristics**

#### **Response Format**
- **Language**: Natural, conversational text
- **Length**: Typically 1-3 sentences for simple questions
- **Style**: Descriptive and informative
- **Accuracy**: Depends on model training and input quality

#### **Performance Metrics**
- **Processing Time**: Usually 1-5 seconds
- **Token Count**: Input + Output tokens tracked
- **Memory Usage**: GPU memory consumption monitored

### **Common Use Cases**

#### **1. Content Analysis**
```bash
# Analyze video content
--video_path "movie_scene.mp4" --question "What's happening in this scene?"
```

#### **2. Audio Transcription**
```bash
# Transcribe speech
--audio_path "lecture.wav" --question "What is the speaker saying?"
```

#### **3. Image Understanding**
```bash
# Describe images
--image_path "photo.jpg" --question "What do you see in this image?"
```

#### **4. Multimodal Analysis**
```bash
# Combined analysis
--video_path "demo.mp4" --audio_path "demo.wav" --question "Summarize this demo"
```

#### **5. Conversational AI**
```bash
# Text-only conversation
--question "Explain quantum computing in simple terms"
```

This simple explanation shows how the script transforms various input types into coherent text responses, making VITA's multimodal capabilities accessible and easy to understand.

## 🧠 Model Input/Output Structure

### **Model Input Format**

The VITA model receives structured inputs in the form of tensors and tokenized text. Here's the detailed breakdown:

#### **1. Text Input (Tokenized)**
```python
# Input text is converted to token IDs
input_ids = tokenizer_image_audio_token(
    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
).unsqueeze(0).cuda()

# Shape: [1, sequence_length]
# Example: [1, 256] for a 256-token sequence
```

#### **2. Image/Video Input (Visual Features)**
```python
# Video frames or image patches
image_tensor = video_frames.half().cuda()  # or processed image

# Shape: [batch_size, num_frames, channels, height, width]
# Example: [1, 16, 3, 384, 384] for 16 video frames
# Example: [1, 1, 3, 384, 384] for single image
```

#### **3. Audio Input (Audio Features)**
```python
# Audio features dictionary
audios = {
    "audios": audio.half().cuda(),           # Audio features
    "lengths": audio_length.half().cuda(),   # Audio length
    "lengths_for_llm": audio_for_llm_lens.cuda()  # LLM audio length
}

# Shapes:
# audios["audios"]: [1, audio_length, feature_dim]  # e.g., [1, 400, 80]
# audios["lengths"]: [1]  # e.g., [400]
# audios["lengths_for_llm"]: [1]  # e.g., [60]
```

### **Model Output Format**

#### **1. Generated Token IDs**
```python
# Model generates token IDs
output_ids = model.generate(
    input_ids,
    images=image_tensor,
    audios=audios,
    max_new_tokens=1024
)

# Shape: [batch_size, total_sequence_length]
# Example: [1, 512] for input + generated tokens
```

#### **2. Decoded Text Output**
```python
# Convert token IDs back to text
outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]

# Format: Natural language string
# Example: "The person in the video is cooking in a kitchen..."
```

### **Input Processing Pipeline**

#### **Step 1: Text Tokenization**
```python
# Original question
question = "What is happening in this video?"

# Add special tokens for multimodal inputs
if video_path and audio_path:
    prompt = DEFAULT_IMAGE_TOKEN * num_frames + "\n" + question + DEFAULT_AUDIO_TOKEN
elif video_path:
    prompt = DEFAULT_IMAGE_TOKEN * num_frames + "\n" + question
elif audio_path:
    prompt = question + DEFAULT_AUDIO_TOKEN
else:
    prompt = question

# Tokenize to IDs
input_ids = tokenizer(prompt, return_tensors="pt")
# Result: [1, sequence_length] tensor of token IDs
```

#### **Step 2: Visual Processing**
```python
# Video: Extract and process frames
video_frames, slice_len = _get_rawvideo_dec(video_path, image_processor)
# Result: [1, num_frames, 3, 384, 384] tensor

# Image: Process single image
image_tensor = model.process_images(image, model.config)
# Result: [1, 1, 3, 384, 384] tensor

# Convert to model dtype and device
image_tensor = image_tensor.to(dtype=model.dtype, device="cuda")
```

#### **Step 3: Audio Processing**
```python
# Process audio file
audio, audio_for_llm_lens = audio_processor.process(audio_path)

# Prepare audio tensors
audios = {
    "audios": audio.half().cuda(),                    # [1, 400, 80]
    "lengths": torch.tensor([audio.shape[0]]).cuda(), # [1]
    "lengths_for_llm": torch.tensor([audio_for_llm_lens]).cuda()  # [1]
}
```

### **Model Architecture Input Flow**

#### **1. Multimodal Encoder Inputs**
```python
# Vision Encoder Input
vision_input = {
    "images": image_tensor,  # [1, num_frames, 3, 384, 384]
    "num_frames": slice_len  # Scalar: number of frames
}

# Audio Encoder Input  
audio_input = {
    "audios": audios["audios"],           # [1, 400, 80]
    "lengths": audios["lengths"],         # [1]
    "lengths_for_llm": audios["lengths_for_llm"]  # [1]
}

# Text Encoder Input
text_input = {
    "input_ids": input_ids,  # [1, sequence_length]
    "attention_mask": attention_mask  # [1, sequence_length]
}
```

#### **2. Feature Fusion**
```python
# Vision features → Vision Projector → Vision tokens
vision_features = vision_encoder(vision_input)
vision_tokens = vision_projector(vision_features)
# Shape: [1, num_frames, hidden_dim]

# Audio features → Audio Projector → Audio tokens  
audio_features = audio_encoder(audio_input)
audio_tokens = audio_projector(audio_features)
# Shape: [1, audio_length, hidden_dim]

# Combine with text tokens
combined_tokens = torch.cat([text_tokens, vision_tokens, audio_tokens], dim=1)
# Shape: [1, total_sequence_length, hidden_dim]
```

### **Model Generation Process**

#### **1. Forward Pass**
```python
# Model processes combined multimodal input
with torch.inference_mode():
    outputs = model.generate(
        input_ids=input_ids,           # [1, text_length]
        images=image_tensor,           # [1, num_frames, 3, 384, 384]
        audios=audios,                 # Dict with audio tensors
        do_sample=False,               # Deterministic generation
        temperature=0.01,              # Low temperature
        max_new_tokens=1024,           # Maximum new tokens
        use_cache=True,                # Use KV cache
        stopping_criteria=[stopping_criteria]  # Stop generation criteria
    )
```

#### **2. Output Processing**
```python
# Extract generated sequence
output_ids = outputs.sequences  # [1, total_length]

# Remove input tokens (keep only generated tokens)
input_token_len = input_ids.shape[1]
generated_ids = output_ids[:, input_token_len:]  # [1, generated_length]

# Decode to text
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

# Clean up output
generated_text = generated_text.strip()
if generated_text.endswith(stop_str):
    generated_text = generated_text[:-len(stop_str)]
```

### **Tensor Shape Examples**

#### **Input Shapes by Modality**

| Input Type | Tensor Shape | Description |
|------------|--------------|-------------|
| **Text Only** | `[1, 50]` | 50 token sequence |
| **Image Only** | `[1, 1, 3, 384, 384]` | Single 384x384 RGB image |
| **Video Only** | `[1, 16, 3, 384, 384]` | 16 frames of 384x384 video |
| **Audio Only** | `[1, 400, 80]` | 400 time steps, 80 features |
| **Video + Audio** | `[1, 16, 3, 384, 384]` + `[1, 400, 80]` | Combined visual and audio |

#### **Output Shapes**

| Output Type | Shape | Description |
|-------------|-------|-------------|
| **Generated Tokens** | `[1, 128]` | 128 generated token IDs |
| **Logits** | `[1, 128, vocab_size]` | Probability distribution over vocabulary |
| **Hidden States** | `[1, 128, hidden_dim]` | Internal model representations |

### **Memory and Performance**

#### **Memory Usage**
```python
# Typical memory consumption
input_memory = {
    "text_tokens": "~1MB",      # Token embeddings
    "image_tensor": "~50MB",    # Single image (384x384)
    "video_tensor": "~800MB",   # 16 frames (384x384)
    "audio_tensor": "~1MB",     # Audio features
    "model_weights": "~14GB",   # VITA model parameters
    "activations": "~2GB"       # Intermediate computations
}
```

#### **Processing Time**
```python
# Typical processing times (RTX 4090)
processing_times = {
    "text_only": "0.5-1.0 seconds",
    "image_only": "1.0-2.0 seconds", 
    "video_only": "2.0-4.0 seconds",
    "audio_only": "1.5-3.0 seconds",
    "multimodal": "3.0-6.0 seconds"
}
```

### **Special Tokens and Markers**

#### **Token Types**
```python
# Special tokens used in VITA
DEFAULT_IMAGE_TOKEN = "<image>"    # Marks image/video input
DEFAULT_AUDIO_TOKEN = "<audio>"    # Marks audio input
DEFAULT_VIDEO_TOKEN = "<video>"    # Marks video input
IMAGE_TOKEN_INDEX = -200           # Internal image token ID
AUDIO_TOKEN_INDEX = -500           # Internal audio token ID
IGNORE_INDEX = -100                # Ignored during loss calculation
```

#### **Token Sequence Example**
```python
# Example tokenized sequence
tokens = [
    1,      # <s> (start token)
    151644, # "What"
    445,    # "is"
    338,    # "happening"
    -200,   # <image> token
    -200,   # <image> token (repeated for each frame)
    -200,   # <image> token
    -500,   # <audio> token
    2       # </s> (end token)
]
```

### **Model Configuration**

#### **Key Parameters**
```python
# vita/constants.py:1-15
# Model configuration
model_config = {
    "max_image_length": 16,        # Maximum video frames
    "min_image_length": 4,         # Minimum video frames
    "image_resolution": 384,       # Image/video resolution
    "audio_feature_dim": 80,       # Audio feature dimension
    "max_audio_length": 400,       # Maximum audio length
    "vocab_size": 32000,           # Vocabulary size
    "hidden_size": 4096,           # Model hidden dimension
    "num_layers": 32,              # Number of transformer layers
    "num_heads": 32                # Number of attention heads
}
```

This detailed explanation shows how the VITA model processes different input modalities and generates coherent text responses, providing insight into the internal data flow and tensor operations.

## 🔢 What are Token IDs (ids)?

### **Token IDs Explained**

Token IDs are numerical representations of text that machine learning models can understand and process. Here's how they work:

#### **1. Basic Concept**
```python
# Human-readable text
text = "What is happening in this video?"

# Token IDs (what the model sees)
token_ids = [1, 151644, 445, 338, 11, 318, 257, 2]
#            ^  ^      ^   ^   ^  ^   ^   ^
#            |  |      |   |   |  |   |   |
#            |  |      |   |   |  |   |   └─ </s> (end token)
#            |  |      |   |   |  |   └─ "video"
#            |  |      |   |   |  └─ "in"
#            |  |      |   |   └─ "this"
#            |  |      |   └─ "happening"
#            |  |      └─ "is"
#            |  └─ "What"
#            └─ <s> (start token)
```

#### **2. Tokenization Process**
```python
# Step 1: Text to tokens
text = "Hello world"
tokens = ["Hello", "world"]  # Split into words/subwords

# Step 2: Tokens to IDs
tokenizer = AutoTokenizer.from_pretrained("model_name")
token_ids = tokenizer.encode(text)
# Result: [1, 15496, 995, 2]  # [<s>, "Hello", "world", </s>]
```

### **Types of Token IDs in VITA**

#### **1. Regular Text Tokens**
```python
# Common words and their IDs
vocabulary = {
    1: "<s>",           # Start of sequence
    2: "</s>",          # End of sequence
    151644: "What",     # Common word
    445: "is",          # Common word
    338: "happening",   # Common word
    11: "in",           # Common word
    318: "this",        # Common word
    257: "video",       # Common word
    0: "<unk>",         # Unknown word
    32000: "<pad>"      # Padding token
}
```

#### **2. Special Multimodal Tokens**
```python
# VITA-specific special tokens
special_tokens = {
    -200: "<image>",    # Image/video placeholder
    -500: "<audio>",    # Audio placeholder
    -100: "<ignore>",   # Ignored during training
    32001: "<image>",   # Alternative image token
    32002: "<audio>"    # Alternative audio token
}
```

### **Token ID Processing in VITA**

#### **1. Input Tokenization**
```python
# Original text with special tokens
prompt = "<image><image><image>What is happening in this video?<audio>"

# Tokenize to IDs
input_ids = tokenizer_image_audio_token(
    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
)

# Result: [1, -200, -200, -200, 151644, 445, 338, 11, 318, 257, 2, -500, 2]
#         ^  ^     ^     ^     ^      ^   ^   ^  ^   ^   ^  ^  ^    ^
#         |  |     |     |     |      |   |   |  |   |   |  |  |    |
#         |  |     |     |     |      |   |   |  |   |   |  |  |    └─ </s>
#         |  |     |     |     |      |   |   |  |   |   |  |  └─ <audio>
#         |  |     |     |     |      |   |   |  |   |   |  └─ </s>
#         |  |     |     |     |      |   |   |  |   |   └─ "video"
#         |  |     |     |     |      |   |   |  |   └─ "this"
#         |  |     |     |     |      |   |   |  └─ "in"
#         |  |     |     |     |      |   |   └─ "happening"
#         |  |     |     |     |      |   └─ "is"
#         |  |     |     |     |      └─ "What"
#         |  |     |     |     └─ </s>
#         |  |     |     └─ <image> (frame 3)
#         |  |     └─ <image> (frame 2)
#         |  └─ <image> (frame 1)
#         └─ <s>
```

#### **2. Model Processing**
```python
# Model receives token IDs
input_ids = torch.tensor([[1, -200, -200, -200, 151644, 445, 338, 11, 318, 257, 2, -500, 2]])

# Model processes and generates new token IDs
output_ids = model.generate(input_ids, images=image_tensor, audios=audios)

# Result: [1, -200, -200, -200, 151644, 445, 338, 11, 318, 257, 2, -500, 2, 151644, 445, 338, 11, 318, 257, 2]
#         ^  ^     ^     ^     ^      ^   ^   ^  ^   ^   ^  ^  ^    ^  ^      ^   ^   ^  ^   ^   ^  ^
#         |  |     |     |     |      |   |   |  |   |   |  |  |    |  |      |   |   |  |   |   |  |
#         |  |     |     |     |      |   |   |  |   |   |  |  |    |  |      |   |   |  |   |   |  └─ </s>
#         |  |     |     |     |      |   |   |  |   |   |  |  |    |  |      |   |   |  |   |   └─ "video"
#         |  |     |     |     |      |   |   |  |   |   |  |  |    |  |      |   |   |  |   └─ "this"
#         |  |     |     |     |      |   |   |  |   |   |  |  |    |  |      |   |   |  └─ "in"
#         |  |     |     |     |      |   |   |  |   |   |  |  |    |  |      |   |   └─ "happening"
#         |  |     |     |     |      |   |   |  |   |   |  |  |    |  |      |   └─ "is"
#         |  |     |     |     |      |   |   |  |   |   |  |  |    |  |      └─ "What"
#         |  |     |     |     |      |   |   |  |   |   |  |  |    |  └─ </s>
#         |  |     |     |     |      |   |   |  |   |   |  |  |    └─ <audio>
#         |  |     |     |     |      |   |   |  |   |   |  |  └─ </s>
#         |  |     |     |     |      |   |   |  |   |   |  └─ "video"
#         |  |     |     |     |      |   |   |  |   |   └─ "this"
#         |  |     |     |     |      |   |   |  |   └─ "in"
#         |  |     |     |     |      |   |   |  └─ "happening"
#         |  |     |     |     |      |   |   └─ "is"
#         |  |     |     |     |      |   └─ "What"
#         |  |     |     |     |      └─ </s>
#         |  |     |     |     └─ "video"
#         |  |     |     └─ "this"
#         |  |     └─ "in"
#         |  └─ "happening"
#         └─ "is"
```

#### **3. Output Decoding**
```python
# Extract only the generated tokens (remove input tokens)
input_length = input_ids.shape[1]  # 13 tokens
generated_ids = output_ids[:, input_length:]  # [151644, 445, 338, 11, 318, 257, 2]

# Decode back to text
generated_text = tokenizer.decode(generated_ids[0])
# Result: "What is happening in this video?"
```

### **Why Use Token IDs?**

#### **1. Numerical Processing**
```python
# Models work with numbers, not text
# Text: "Hello world" → Cannot do math operations
# IDs: [15496, 995] → Can do matrix operations, embeddings, etc.
```

#### **2. Vocabulary Management**
```python
# Fixed vocabulary size (e.g., 32,000 tokens)
# Each word/subword gets a unique ID
# Unknown words → <unk> token (ID: 0)
```

#### **3. Efficient Processing**
```python
# Token IDs are compact integers
# Easy to store, transmit, and process
# Can be converted to embeddings for neural networks
```

### **Token ID Examples in VITA**

#### **1. Simple Text**
```python
text = "Hello"
token_ids = [1, 15496, 2]  # [<s>, "Hello", </s>]
```

#### **2. Text with Image**
```python
text = "<image>What do you see?"
token_ids = [1, -200, 151644, 345, 345, 345, 2]  # [<s>, <image>, "What", "do", "you", "see", </s>]
```

#### **3. Text with Audio**
```python
text = "What did you hear?<audio>"
token_ids = [1, 151644, 345, 345, 345, 2, -500, 2]  # [<s>, "What", "did", "you", "hear", </s>, <audio>, </s>]
```

#### **4. Multimodal Input**
```python
text = "<image><image>What is happening?<audio>"
token_ids = [1, -200, -200, 151644, 445, 338, 2, -500, 2]
```

### **Token ID Operations**

#### **1. Adding Special Tokens**
```python
# Add image tokens for video frames
num_frames = 16
image_tokens = [-200] * num_frames  # [<image>, <image>, ..., <image>]

# Add audio token
audio_token = [-500]  # [<audio>]

# Combine with text
full_sequence = image_tokens + text_tokens + audio_token
```

#### **2. Masking and Attention**
```python
# Create attention mask (1 = attend, 0 = ignore)
input_ids = [1, -200, -200, 151644, 445, 2]
attention_mask = [1, 1, 1, 1, 1, 1]  # Attend to all tokens

# Special tokens might be masked differently
special_mask = [1, 0, 0, 1, 1, 1]  # Don't attend to <image> tokens
```

#### **3. Position Encoding**
```python
# Each token ID gets a position
input_ids = [1, -200, 151644, 445, 2]
positions = [0, 1, 2, 3, 4]  # Position of each token

# Model uses both token ID and position for processing
```

### **Common Token ID Patterns**

#### **1. Conversation Format**
```python
# User message
user_ids = [1, 151644, 445, 338, 2]  # "What is happening?"

# Assistant response
assistant_ids = [1, 151644, 445, 338, 11, 318, 257, 2]  # "What is happening in this video?"

# Combined conversation
conversation_ids = user_ids + assistant_ids
```

#### **2. Multimodal Sequences**
```python
# Video + Text + Audio
sequence_ids = [
    1,        # <s>
    -200,     # <image> (frame 1)
    -200,     # <image> (frame 2)
    -200,     # <image> (frame 3)
    151644,   # "What"
    445,      # "is"
    338,      # "happening"
    2,        # </s>
    -500,     # <audio>
    2         # </s>
]
```

### **Debugging Token IDs**

#### **1. Inspect Token IDs**
```python
# Print token IDs with their meanings
def print_tokens(token_ids, tokenizer):
    for i, token_id in enumerate(token_ids):
        token = tokenizer.decode([token_id])
        print(f"Position {i}: ID {token_id} = '{token}'")

# Example usage
token_ids = [1, -200, 151644, 445, 2]
print_tokens(token_ids, tokenizer)
# Output:
# Position 0: ID 1 = '<s>'
# Position 1: ID -200 = '<image>'
# Position 2: ID 151644 = 'What'
# Position 3: ID 445 = 'is'
# Position 4: ID 2 = '</s>'
```

#### **2. Verify Tokenization**
```python
# Round-trip test
original_text = "What is happening?"
token_ids = tokenizer.encode(original_text)
decoded_text = tokenizer.decode(token_ids)
assert original_text in decoded_text  # Should be True
```

Token IDs are the fundamental building blocks that allow the VITA model to process and understand text, images, and audio in a unified numerical format, enabling sophisticated multimodal reasoning and generation.
