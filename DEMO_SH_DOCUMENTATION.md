# VITA Demo.sh Script Documentation

## 📋 Overview

The `demo.sh` script is a simple bash wrapper that demonstrates VITA's multimodal capabilities by running the `video_audio_demo.py` script with predefined parameters. It showcases VITA's ability to process and describe images using the VITA-1.5 model.

## 🎯 Purpose

The demo script serves as a quick test to verify that:
- The VITA model is properly loaded and functional
- The container environment is correctly configured
- The multimodal inference pipeline works as expected
- The model can generate meaningful descriptions of visual content

## 📁 Script Content

```bash
#!/bin/bash

python /workspace/3thrdparties/VITA/video_audio_demo.py \
--model_path ~/models/VITA-1.5 \
--image_path /workspace/3thrdparties/VITA/asset/vita_newlog.jpg \
--model_type qwen2p5_instruct \
--conv_mode qwen2p5_instruct \
--question "Describe this images."
```

## 🔧 Parameters Explained

### **Model Configuration**
- `--model_path ~/models/VITA-1.5`: Specifies the path to the VITA-1.5 model directory
  - The model contains approximately 341M whale encoder parameters and 289M vision encoder parameters
  - Located in the user's home directory under `~/models/VITA-1.5`

### **Input Configuration**
- `--image_path /workspace/3thrdparties/VITA/asset/vita_newlog.jpg`: Path to the input image
  - Uses the VITA logo image as a test case
  - The image is processed through dynamic preprocessing with 1-12 patches
  - Final image tensor shape: `torch.Size([5, 3, 448, 448])` (5 patches, 3 channels, 448x448 resolution)

### **Model Type Configuration**
- `--model_type qwen2p5_instruct`: Specifies the underlying language model architecture
  - Uses Qwen2.5 Instruct as the base language model
  - Provides instruction-following capabilities

### **Conversation Mode**
- `--conv_mode qwen2p5_instruct`: Sets the conversation template
  - Matches the model type for consistent behavior
  - Uses the Qwen2.5 Instruct conversation format

### **Question/Prompt**
- `--question "Describe this images."`: The user's query to the model
  - Simple instruction to describe the provided image
  - The model will analyze the visual content and generate a detailed description

## 🚀 Execution Flow

### **1. Script Initialization**
```bash
#!/bin/bash
```
- Sets the script to run with bash interpreter
- Ensures proper shell environment

### **2. Python Script Execution**
The script calls the main VITA demo with the following process:

#### **Input Processing**
- **Image Loading**: Opens and converts the image to RGB format
- **Dynamic Preprocessing**: Applies intelligent patch extraction (1-12 patches based on content)
- **Token Processing**: Converts the question into token format with image tokens
- **Audio Processing**: Initializes empty audio tensors (no audio input in this demo)

#### **Model Inference**
- **Input Shape**: `torch.Size([1, 160])` - 160 tokens including system prompt, user question, and image tokens
- **Image Tensor**: `torch.Size([5, 3, 448, 448])` - 5 image patches processed
- **Audio Tensor**: Empty audio data structure (all zeros)

#### **Generation Process**
- **Temperature**: 0.01 (very low for deterministic output)
- **Max Tokens**: 1024 new tokens
- **Beam Search**: Single beam (num_beams=1)
- **Sampling**: Disabled (do_sample=False)

### **3. Output Generation**
The model generates a detailed description of the VITA logo, including:
- Visual analysis of the logo design
- Color scheme description
- Typography analysis
- Brand messaging interpretation
- Contextual information about the project

## 🔬 Detailed Implementation Analysis

### **video_audio_demo.py Architecture**

The `video_audio_demo.py` script implements a comprehensive multimodal inference pipeline. Here's the detailed breakdown:

#### **1. Import Dependencies and Setup**
```python
# Core libraries
import argparse, os, time
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
```

#### **2. Argument Parsing and Configuration**
```python
parser = argparse.ArgumentParser(description="Process model and video paths.")
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--model_base", type=str, default=None)
parser.add_argument("--video_path", type=str, default=None)
parser.add_argument("--image_path", type=str, default=None)
parser.add_argument("--audio_path", type=str, default=None)
parser.add_argument("--model_type", type=str, default="mixtral-8x7b")
parser.add_argument("--conv_mode", type=str, default="mixtral_two")
parser.add_argument("--question", type=str, default="")
parser.add_argument("--frameCat", action='store_true')
```

**Key Configuration Parameters:**
- `max_frames = MAX_IMAGE_LENGTH` (16 frames for video)
- `video_framerate = 1` (1 frame per second)
- `temperature = 0.01` (deterministic generation)
- `top_p = None` (no nucleus sampling)
- `num_beams = 1` (greedy decoding)

#### **3. Model Loading Process**
```python
disable_torch_init()  # Optimize PyTorch initialization
model_path = os.path.expanduser(model_path)
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, model_base, model_name, args.model_type
)
```

**Model Loading Details:**
- **Whale Encoder**: 341.4M parameters for multimodal fusion
- **Vision Encoder**: 289.9M parameters for image processing
- **Language Model**: Qwen2.5 Instruct base
- **Total Parameters**: ~631M parameters
- **Precision**: FP16 (half precision) for memory efficiency

#### **4. Input Processing Pipeline**

##### **A. Image Processing (for demo.sh)**
```python
if image_path is not None:
    image = Image.open(image_path).convert("RGB")
    if args.frameCat:
        image, p_num = dynamic_preprocess(image, min_num=2, max_num=12, 
                                        image_size=448, use_thumbnail=True, 
                                        img_mean=image_processor.image_mean)
    else:
        image, p_num = dynamic_preprocess(image, min_num=1, max_num=12, 
                                        image_size=448, use_thumbnail=True)
    
    image_tensor = model.process_images(image, model.config).to(
        dtype=model.dtype, device="cuda"
    )
    qs = DEFAULT_IMAGE_TOKEN * p_num[0] + "\n" + qs
    modality = "image"
```

**Dynamic Preprocessing Algorithm:**
1. **Aspect Ratio Analysis**: Calculates original image aspect ratio
2. **Patch Grid Calculation**: Determines optimal grid layout (1x1 to 12x12)
3. **Target Ratio Selection**: Finds closest aspect ratio to original
4. **Image Resizing**: Resizes to target dimensions
5. **Patch Extraction**: Splits image into uniform patches
6. **Patch Processing**: Each patch becomes 448x448 pixels

**Example for VITA Logo:**
- Original: 2798×770 pixels (aspect ratio: 3.634)
- Processed: 5 patches (4×1 grid + 1 thumbnail)
- Each patch: 448×448×3 (RGB channels)
- Final tensor: `torch.Size([5, 3, 448, 448])`

**Why 5 Patches?**
The VITA logo's wide aspect ratio (3.634) triggers the dynamic preprocessing algorithm:

1. **Aspect Ratio Analysis**: Original image is 2798×770 pixels
2. **Grid Selection**: Algorithm finds closest match is 4×1 grid (aspect ratio: 4.000, difference: 0.366)
3. **Patch Extraction**: Creates 4 patches from the 4×1 grid layout
4. **Thumbnail Addition**: Since `use_thumbnail=True` and patches > 1, adds 1 thumbnail
5. **Final Result**: 4 grid patches + 1 thumbnail = 5 total patches

This design provides both detailed views (4 patches) and global context (1 thumbnail) for comprehensive image understanding.

##### **B. Audio Processing (Empty for demo.sh)**
```python
if audio_path is not None:
    # Audio processing code (not used in demo.sh)
    audios = dict()
    audios["audios"] = audio.half().cuda()
    audios["lengths"] = audio_length.half().cuda()
    audios["lengths_for_llm"] = audio_for_llm_lens.cuda()
else:
    audios = None
```

**For demo.sh**: Audio tensors are initialized as empty (all zeros):
```python
audios = {
    'audios': tensor([[[0., 0., 0., ..., 0., 0., 0.]]], device='cuda:0'),
    'lengths': tensor([400.], device='cuda:0'),
    'lengths_for_llm': tensor([60], device='cuda:0')
}
```

##### **C. Video Processing (Not used in demo.sh)**
```python
def _get_rawvideo_dec(video_path, image_processor, max_frames=MAX_IMAGE_LENGTH, 
                     min_frames=4, image_resolution=384, video_framerate=1, 
                     s=None, e=None, image_aspect_ratio="pad"):
    # Video decoding and frame extraction logic
    # Uses decord library for efficient video processing
```

#### **5. Conversation Template Processing**
```python
conv = conv_templates[conv_mode].copy()  # qwen2p5_instruct template
conv.append_message(conv.roles[0], qs)   # User message with image tokens
conv.append_message(conv.roles[1], None) # Assistant response (empty)
prompt = conv.get_prompt(modality)       # Generate final prompt
```

**Qwen2.5 Instruct Template Structure:**
```
<|im_start|>system
You are an AI robot and your name is VITA.
- You are a multimodal large language model developed by the open source community.
- Your aim is to be helpful, honest and harmless.
- You support the ability to communicate fluently and answer user questions in multiple languages.
- If the user corrects the wrong answer you generated, you will apologize and discuss the correct answer.
- You must answer the question strictly according to the content of the image given by the user.
<|im_end|>
<|im_start|>user
<image><image><image><image><image>
Describe this images.
<|im_end|>
<|im_start|>assistant
```

#### **6. Tokenization Process**
```python
if audio_path:
    input_ids = tokenizer_image_audio_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, 
                                          return_tensors="pt").unsqueeze(0).cuda()
else:
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, 
                                    return_tensors="pt").unsqueeze(0).cuda()
```

**Tokenization Details:**
- **Image Tokens**: Each `<image>` becomes `IMAGE_TOKEN_INDEX` (-200)
- **Special Tokens**: System/user/assistant tokens from Qwen2.5
- **Final Shape**: `torch.Size([1, 160])` - 160 tokens total
- **Token Types**: Text tokens + 5 image tokens + special tokens

**Why Exactly 5 `<image>` Tokens?**
The 5 image tokens correspond to the dynamic preprocessing result:

```
VITA Logo (2798×770, aspect ratio: 3.634)
                    ↓
            Dynamic Preprocessing
                    ↓
4×1 Grid Layout + Thumbnail
┌─────┬─────┬─────┬─────┐  ┌─────┐
│  1  │  2  │  3  │  4  │  │  T  │
│     │     │     │     │  │     │
└─────┴─────┴─────┴─────┘  └─────┘
  4 patches (448×448 each)  1 thumbnail
                    ↓
            Token Generation
                    ↓
<image><image><image><image><image>
```

**Algorithm Logic:**
1. **Aspect Ratio Matching**: Finds 4×1 grid as closest to original 3.634 ratio
2. **Grid Processing**: Creates 4 patches from horizontal layout
3. **Thumbnail Addition**: Adds 1 thumbnail when `use_thumbnail=True` and patches > 1
4. **Token Mapping**: Each patch becomes one `<image>` token
5. **Final Count**: 4 grid tokens + 1 thumbnail token = 5 total tokens

#### **7. Stopping Criteria Setup**
```python
stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
keywords = [stop_str]  # ["<|im_end|>"]
stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
```

#### **8. Model Generation Process**
```python
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor,           # [5, 3, 448, 448]
        audios=audios,                 # Empty audio data
        do_sample=False,               # Deterministic generation
        temperature=temperature,       # 0.01
        top_p=top_p,                   # None
        num_beams=num_beams,           # 1
        output_scores=True,
        return_dict_in_generate=True,
        max_new_tokens=1024,           # Maximum response length
        use_cache=True,                # Enable KV cache
        stopping_criteria=[stopping_criteria],
        shared_v_pid_stride=None
    )
```

**Generation Parameters Explained:**
- **`do_sample=False`**: Greedy decoding (most likely token)
- **`temperature=0.01`**: Near-deterministic (very low randomness)
- **`num_beams=1`**: No beam search (faster, single path)
- **`max_new_tokens=1024`**: Maximum 1024 new tokens in response
- **`use_cache=True`**: KV cache for faster generation
- **`stopping_criteria`**: Stop at conversation end tokens

#### **9. Output Processing**
```python
output_ids = output_ids.sequences
input_token_len = input_ids.shape[1]
outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]

# Clean up output
outputs = outputs.strip()
if outputs.endswith(stop_str):
    outputs = outputs[: -len(stop_str)]
outputs = outputs.strip()
```

**Output Processing Steps:**
1. **Extract Sequences**: Get generated token sequences
2. **Decode Tokens**: Convert token IDs back to text
3. **Remove Stop Tokens**: Clean up conversation markers
4. **Format Output**: Final text response

#### **10. Performance Monitoring**
```python
start_time = time.time()
# ... generation process ...
infer_time = time.time() - start_time
print(f"Time consume: {infer_time}")
```

### **Memory and Computational Flow**

#### **GPU Memory Usage:**
- **Model Weights**: ~2.5GB (FP16 precision)
- **Image Tensors**: ~5 × 3 × 448 × 448 × 2 bytes = ~6MB
- **Audio Tensors**: ~400 × 2 bytes = ~800 bytes
- **KV Cache**: Variable based on sequence length
- **Total VRAM**: ~3-4GB for inference

#### **Computational Complexity:**
- **Image Processing**: O(patch_count × 448²)
- **Token Processing**: O(sequence_length)
- **Generation**: O(sequence_length × model_parameters)
- **Total Time**: ~15 seconds for 1024 tokens

### **Error Handling and Robustness**

#### **Input Validation:**
```python
assert (audio_path is None) != (qs == ""), "Exactly one of audio_path or qs must be non-None"
assert len(p_num) == 1  # Single image processing
```

#### **Model Compatibility:**
- **Supported Types**: `mixtral-8x7b`, `nemo`, `qwen2p5_instruct`, `qwen2p5_fo_instruct`
- **Conversation Modes**: Matching templates for each model type
- **Precision Support**: FP16, INT8, INT4 quantization options

#### **Resource Management:**
- **CUDA Memory**: Automatic device placement
- **Torch Inference Mode**: Disables gradient computation
- **Context Length**: Configurable based on model capabilities

## 🔍 Dynamic Preprocessing Algorithm Deep Dive

### **Understanding the 5 Image Tokens**

The VITA demo produces exactly 5 `<image>` tokens due to a sophisticated dynamic preprocessing algorithm that adapts to different image aspect ratios. Here's the complete breakdown:

#### **Algorithm Overview**
```python
def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    # 1. Analyze original image aspect ratio
    # 2. Find optimal grid layout (1x1 to 12x12)
    # 3. Extract patches from grid
    # 4. Add thumbnail if conditions are met
    # 5. Return processed images and count
```

#### **Step-by-Step Process for VITA Logo**

**Step 1: Image Analysis**
- **Original**: 2798×770 pixels
- **Aspect Ratio**: 3.634 (very wide, landscape)
- **Target**: Find grid layout that best preserves this ratio

**Step 2: Grid Selection**
The algorithm evaluates all possible grid layouts:

| Grid Layout | Aspect Ratio | Difference | Patches | Selection |
|-------------|--------------|------------|---------|-----------|
| 4×1         | 4.000        | **0.366**  | 4       | ✅ **Winner** |
| 3×1         | 3.000        | 0.634      | 3       | ❌ |
| 6×2         | 3.000        | 0.634      | 12      | ❌ Too many |
| 5×2         | 2.500        | 1.134      | 10      | ❌ Too many |

**Step 3: Patch Extraction**
```
Original VITA Logo (2798×770)
┌─────────────────────────────────────────────────────────┐
│                    VITA LOGO                            │
└─────────────────────────────────────────────────────────┘
                              ↓
                    4×1 Grid Layout (1792×448)
┌─────┬─────┬─────┬─────┐
│  1  │  2  │  3  │  4  │  ← 4 patches, each 448×448
│     │     │     │     │
└─────┴─────┴─────┴─────┘
```

**Step 4: Thumbnail Addition**
```python
if use_thumbnail and len(processed_images) != 1:
    thumbnail_img = image.resize((image_size, image_size))
    processed_images.append(thumbnail_img)
```

Since `use_thumbnail=True` and we have 4 patches (not 1):
```
4×1 Grid + Thumbnail
┌─────┬─────┬─────┬─────┐  ┌─────┐
│  1  │  2  │  3  │  4  │  │  T  │  ← Thumbnail (448×448)
│     │     │     │     │  │     │
└─────┴─────┴─────┴─────┘  └─────┘
  4 grid patches             1 thumbnail
```

**Step 5: Token Generation**
```python
qs = DEFAULT_IMAGE_TOKEN * p_num[0] + "\n" + qs
# p_num[0] = 5 (4 grid + 1 thumbnail)
# Result: "<image><image><image><image><image>\nDescribe this images."
```

#### **Why This Design is Effective**

1. **Aspect Ratio Preservation**: 4×1 grid maintains the wide format of the VITA logo
2. **Detail Capture**: Each 448×448 patch captures fine details from different sections
3. **Global Context**: Thumbnail provides complete image overview
4. **Computational Efficiency**: 5 patches balance detail vs. processing cost
5. **Adaptive Processing**: Algorithm works for any image aspect ratio

#### **Alternative Scenarios**

**Square Image (1:1 ratio):**
- Would likely use 1×1 grid (1 patch)
- No thumbnail added (since patches = 1)
- Result: 1 `<image>` token

**Tall Image (1:2 ratio):**
- Would use 1×2 grid (2 patches)
- Thumbnail added (since patches > 1)
- Result: 3 `<image>` tokens (2 grid + 1 thumbnail)

**Very Wide Image (5:1 ratio):**
- Would use 5×1 grid (5 patches)
- Thumbnail added
- Result: 6 `<image>` tokens (5 grid + 1 thumbnail)

#### **Code Implementation**
```python
# In video_audio_demo.py
if args.frameCat:
    # Uses frameCat preprocessing (different algorithm)
    image, p_num = dynamic_preprocess(image, min_num=2, max_num=12, 
                                    image_size=448, use_thumbnail=True, 
                                    img_mean=image_processor.image_mean)
else:
    # Uses standard patch preprocessing (used in demo.sh)
    image, p_num = dynamic_preprocess(image, min_num=1, max_num=12, 
                                    image_size=448, use_thumbnail=True)

# p_num[0] contains the number of patches
qs = DEFAULT_IMAGE_TOKEN * p_num[0] + "\n" + qs
```

This dynamic approach ensures that VITA can effectively process images of any aspect ratio while maintaining optimal detail capture and computational efficiency.

## 📊 Expected Output

### **Model Loading Information**
```
the number of whale encoder params: 341.3681640625M
Loading checkpoint shards: 100%|██████████| 4/4 [00:02<00:00, 1.53it/s]
the number of vision encoder params: 289.9287109375M
```

### **Input Processing Details**
```
qs:
----------
<image><image><image><image><image>
Describe this images.
----------

prompt:
----------
<|im_start|>system
You are an AI robot and your name is VITA...
<|im_start|>user
<image><image><image><image><image>
Describe this images.<|im_end|>
<|im_start|>assistant

----------

input_ids.shape:
----------
torch.Size([1, 160])
----------

image_tensor.shape:
----------
torch.Size([5, 3, 448, 448])
----------

audios:
----------
{'audios': tensor([[[0., 0., 0., ..., 0., 0., 0.]]], device='cuda:0', dtype=torch.float16), 
 'lengths': tensor([400.], device='cuda:0', dtype=torch.float16), 
 'lengths_for_llm': tensor([60], device='cuda:0')}
----------
```

### **Generated Response**
A detailed description of the VITA logo including:
- Logo design analysis
- Color scheme description
- Typography evaluation
- Brand messaging interpretation
- Project context explanation

### **Performance Metrics**
```
Time consume: 14.884526252746582
```

## 🛠️ Technical Details

### **Model Architecture**
- **Base Model**: Qwen2.5 Instruct
- **Vision Encoder**: 289.9M parameters
- **Whale Encoder**: 341.4M parameters
- **Total Parameters**: ~631M parameters

### **Processing Pipeline**
1. **Image Preprocessing**: Dynamic patch extraction with intelligent cropping
2. **Token Processing**: Multi-modal token integration
3. **Model Inference**: GPU-accelerated generation
4. **Output Decoding**: Token-to-text conversion

### **Hardware Requirements**
- **GPU**: NVIDIA GPU with CUDA support
- **Memory**: Sufficient VRAM for model loading (~8GB+ recommended)
- **Storage**: Model files and dependencies

## 🔧 Customization Options

### **Changing the Input Image**
```bash
--image_path /path/to/your/image.jpg
```

### **Modifying the Question**
```bash
--question "What do you see in this image?"
```

### **Using Different Model Types**
```bash
--model_type mixtral-8x7b
--conv_mode mixtral_two
```

### **Adding Video Input**
```bash
--video_path /path/to/video.mp4
```

### **Adding Audio Input**
```bash
--audio_path /path/to/audio.wav
```

## 🚨 Troubleshooting

### **Common Issues**

1. **Model Not Found**
   - Ensure the model path `~/models/VITA-1.5` exists
   - Check model file permissions

2. **Image Not Found**
   - Verify the image path `/workspace/3thrdparties/VITA/asset/vita_newlog.jpg`
   - Check file permissions

3. **CUDA Out of Memory**
   - Reduce batch size or use smaller model
   - Ensure sufficient GPU memory

4. **Import Errors**
   - Verify all dependencies are installed
   - Check Python path configuration

### **Debug Information**
The script provides detailed debug output including:
- Input tensor shapes
- Processing parameters
- Model loading status
- Generation timing

## 📈 Performance Optimization

### **Speed Improvements**
- Use GPU acceleration (CUDA)
- Optimize image preprocessing
- Adjust generation parameters

### **Memory Optimization**
- Use mixed precision (FP16)
- Implement gradient checkpointing
- Optimize batch processing

## 🔗 Related Files

- **Main Script**: `/workspace/3thrdparties/VITA/video_audio_demo.py`
- **Model Files**: `~/models/VITA-1.5/`
- **Test Image**: `/workspace/3thrdparties/VITA/asset/vita_newlog.jpg`
- **Documentation**: `/workspace/3thrdparties/vita-ai-docs/VIDEO_AUDIO_DEMO_DOCUMENTATION.md`

## 📝 Usage Examples

### **Basic Usage**
```bash
cd /workspace/verl/vita
chmod +x demo.sh
./demo.sh
```

### **Docker Container Usage**
```bash
docker exec vita-retrain bash -c "cd /workspace/verl/vita && ./demo.sh"
```

### **Custom Parameters**
```bash
python /workspace/3thrdparties/VITA/video_audio_demo.py \
--model_path ~/models/VITA-1.5 \
--image_path /path/to/custom/image.jpg \
--model_type qwen2p5_instruct \
--conv_mode qwen2p5_instruct \
--question "Analyze this image in detail."
```

## 🎯 Success Criteria

The demo is considered successful when:
- ✅ Model loads without errors
- ✅ Image processing completes successfully
- ✅ Generation produces meaningful output
- ✅ Response time is reasonable (< 30 seconds)
- ✅ No CUDA or memory errors occur

## 📚 Additional Resources

- [VITA GitHub Repository](https://github.com/vita-ai/VITA)
- [VITA Documentation](https://vita-ai.github.io/)
- [Qwen2.5 Model Documentation](https://huggingface.co/Qwen/Qwen2.5)
- [Multimodal AI Best Practices](https://docs.vita-ai.org/)

---

*This documentation provides a comprehensive guide to understanding and using the VITA demo.sh script for multimodal AI demonstrations.*
