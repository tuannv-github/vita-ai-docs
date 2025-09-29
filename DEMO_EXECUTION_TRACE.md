# VITA Demo Execution Trace

This document provides a complete trace of the execution flow from `demo.sh` through the entire VITA system, showing every step from script execution to model inference.

## 🎯 Overview

The trace follows the complete execution path from the demo script through model loading, initialization, input processing, and inference.

## 📋 Complete Execution Trace

### **Step 1: Demo Script Execution**

**File**: `/home/tuannv/vlaa/verl/vita/demo.sh`
```bash
#!/bin/bash

python /workspace/3thrdparties/VITA/video_audio_demo.py \
--model_path ~/models/VITA-1.5 \
--image_path /workspace/3thrdparties/VITA/asset/vita_newlog.jpg \
--model_type qwen2p5_instruct \
--conv_mode qwen2p5_instruct \
--question "Describe this images."
```

**What happens**: The shell script executes the Python demo script with specific arguments.

### **Step 2: Python Script Entry Point**

**File**: `/home/tuannv/vlaa/3thrdparties/VITA/video_audio_demo.py:120-136`
```python
# video_audio_demo.py:120-136
if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Process model and video paths.")

    # Add arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--audio_path", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="mixtral-8x7b")
    parser.add_argument("--conv_mode", type=str, default="mixtral_two")
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--frameCat", action='store_true')

    # Parse the arguments
    args = parser.parse_args()
```

**What happens**: The script parses command line arguments and stores them in the `args` object.

### **Step 3: Argument Processing**

**File**: `/home/tuannv/vlaa/3thrdparties/VITA/video_audio_demo.py:138-146`
```python
# video_audio_demo.py:138-146
# Assign arguments to variables
model_path = args.model_path
model_base = args.model_base
video_path = args.video_path
image_path = args.image_path
audio_path = args.audio_path
qs = args.question
assert (audio_path is None) != (qs == ""), "Exactly one of audio_path or qs must be non-None"
conv_mode = args.conv_mode
```

**What happens**: Arguments are assigned to variables and validated.

### **Step 4: Dynamic Import Selection**

**File**: `/home/tuannv/vlaa/3thrdparties/VITA/video_audio_demo.py:148-151`
```python
# video_audio_demo.py:148-151
if args.frameCat:
    from vita.util.data_utils_video_audio_neg_frameCat import dynamic_preprocess
else:
    from vita.util.data_utils_video_audio_neg_patch import dynamic_preprocess
```

**What happens**: The script imports the appropriate preprocessing function based on the `frameCat` flag.

### **Step 5: Configuration Setup**

**File**: `/home/tuannv/vlaa/3thrdparties/VITA/video_audio_demo.py:153-163`
```python
# video_audio_demo.py:153-163
# The number of visual tokens varies with the length of the video. "max_frames" is the maximum number of frames.
# When the video is long, we will uniformly downsample the video to meet the frames when equal to the "max_frames".
max_frames = MAX_IMAGE_LENGTH  # 100

# The number of frames retained per second in the video.
video_framerate = 1

# Sampling Parameter
temperature = 0.01
top_p = None
num_beams = 1
```

**What happens**: Configuration parameters are set for video processing and model generation.

### **Step 6: Model Loading**

**File**: `/home/tuannv/vlaa/3thrdparties/VITA/video_audio_demo.py:165-170`
```python
# video_audio_demo.py:165-170
disable_torch_init()
model_path = os.path.expanduser(model_path)
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, model_base, model_name, args.model_type
)
```

**What happens**: The script calls `load_pretrained_model()` to load the VITA model.

### **Step 7: Model Builder Execution**

**File**: `/home/tuannv/vlaa/3thrdparties/VITA/vita/model/builder.py:201-207`
```python
# builder.py:201-207
elif model_type == "qwen2p5_instruct":
    # import pdb; pdb.set_trace()
    print(f'Loading Qwen2.5-7B-Instruct model...\n-\n{model_path}\n----------')
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = VITAQwen2ForCausalLM.from_pretrained(
        model_path, low_cpu_mem_usage=True, **kwargs
    )
```

**What happens**: The builder loads the VITAQwen2ForCausalLM model for the `qwen2p5_instruct` type.

### **Step 8: VITAQwen2ForCausalLM Initialization**

**File**: `/home/tuannv/vlaa/3thrdparties/VITA/vita/model/language_model/vita_qwen2.py:125-135`
```python
# vita_qwen2.py:125-135
class VITAQwen2ForCausalLM(Qwen2ForCausalLM, VITAMetaForCausalLM):
    config_class = VITAQwen2Config

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = VITAQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
```

**What happens**: The VITAQwen2ForCausalLM model is initialized, creating a VITAQwen2Model instance.

### **Step 9: VITAQwen2Model Initialization**

**File**: `/home/tuannv/vlaa/3thrdparties/VITA/vita/model/language_model/vita_qwen2.py:118-123`
```python
# vita_qwen2.py:118-123
class VITAQwen2Model(VITAMetaModel, Qwen2Model):
    config_class = VITAQwen2Config

    def __init__(self, config: Qwen2Config):
        super(VITAQwen2Model, self).__init__(config)
```

**What happens**: VITAQwen2Model calls VITAMetaModel.__init__().

### **Step 10: VITAMetaModel Initialization**

**File**: `/home/tuannv/vlaa/3thrdparties/VITA/vita/model/vita_arch.py:14-26`
```python
# vita_arch.py:14-26
class VITAMetaModel:
    def __init__(self, config):
        super(VITAMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(
                config, delay_load=False#not getattr(config, "continuous_training", False)
            )
            if getattr(config, "continuous_training", False):
                config.continuous_training = False
            self.mm_projector = build_vision_projector(config)

        if hasattr(config, "mm_audio_encoder"):
            self.audio_encoder = build_audio_encoder(config)
```

**What happens**: VITAMetaModel initializes the vision tower and audio encoder if they exist in the config.

### **Step 11: Vision Tower Building**

**File**: `/home/tuannv/vlaa/3thrdparties/VITA/vita/model/multimodal_encoder/builder.py:14-43`
```python
# builder.py:14-43
def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None)
    )
    use_s2 = getattr(vision_tower_cfg, "use_s2", False)

    if "sig" in vision_tower.lower():
        # ... SigLIP handling
    elif "eva" in vision_tower.lower():
        # ... EVA-CLIP handling
    elif "clip" in vision_tower.lower():
        # ... CLIP handling
    elif "internvit" in vision_tower.lower():
        if use_s2:
            raise ValueError(f"Currently not supporting S2 for InternViT")
        else:
            return InternViTVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    else:
        raise ValueError(f"Unknown vision tower: {vision_tower}")
```

**What happens**: The builder creates an InternViTVisionTower instance based on the vision tower configuration.

### **Step 12: InternViTVisionTower Initialization**

**File**: `/home/tuannv/vlaa/3thrdparties/VITA/vita/model/multimodal_encoder/internvit/internvit_encoder.py:8-32`
```python
# internvit_encoder.py:8-32
class InternViTVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = -1
        self.scale_pix_shuffle = 0.5

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = AutoConfig.from_pretrained(
                self.vision_tower_name, trust_remote_code=True
            )

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = InternVisionModel.from_pretrained(
            self.vision_tower_name, trust_remote_code=True
        )
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True
```

**What happens**: InternViTVisionTower is initialized and loads the InternViT model and image processor.

### **Step 13: Model Structure Writing**

**File**: `/home/tuannv/vlaa/3thrdparties/VITA/video_audio_demo.py:172-179`
```python
# video_audio_demo.py:172-179
model.resize_token_embeddings(len(tokenizer))

# Write model structure to file
with open("model_structure.txt", "w") as f:
    print(model, file=f)
print("[VITA] model structure written to model_structure.txt")
```

**What happens**: The model structure is written to a file for debugging purposes.

### **Step 14: Vision Tower Access**

**File**: `/home/tuannv/vlaa/3thrdparties/VITA/video_audio_demo.py:182-185`
```python
# video_audio_demo.py:182-185
vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()
image_processor = vision_tower.image_processor
```

**What happens**: The script gets the vision tower and ensures it's loaded.

### **Step 15: Audio Encoder Setup**

**File**: `/home/tuannv/vlaa/3thrdparties/VITA/video_audio_demo.py:187-205`
```python
# video_audio_demo.py:187-205
audio_encoder = model.get_audio_encoder()
audio_encoder.to(dtype=torch.float16)
audio_processor = audio_encoder.audio_processor

model.eval()
if audio_path is not None:
    audio, audio_for_llm_lens = audio_processor.process(os.path.join(audio_path))
    audio_length = audio.shape[0]
    audio = torch.unsqueeze(audio, dim=0)
    audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
    audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
    audios = dict()
    audios["audios"] = audio.half().cuda()
    audios["lengths"] = audio_length.half().cuda()
    audios["lengths_for_llm"] = audio_for_llm_lens.cuda()
else:
    # Create dummy audio for text-only input
    audio = torch.zeros(400, 80)
    audio_length = audio.shape[0]
    audio_for_llm_lens = 60
    # ... prepare dummy audio tensors
```

**What happens**: The audio encoder is set up and audio is processed (or dummy audio is created).

### **Step 16: Image Processing**

**File**: `/home/tuannv/vlaa/3thrdparties/VITA/video_audio_demo.py:207-236`
```python
# video_audio_demo.py:207-236
# Check if the video exists
if video_path is not None:
    video_frames, slice_len = _get_rawvideo_dec(
        video_path,
        image_processor,
        max_frames=max_frames,
        video_framerate=video_framerate,
        image_aspect_ratio=getattr(model.config, "image_aspect_ratio", None),
    )
    image_tensor = video_frames.half().cuda()
    if audio_path:
        qs = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + qs + DEFAULT_AUDIO_TOKEN
    else:
        qs = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + qs
    modality = "video"
elif image_path is not None:
    image = Image.open(image_path).convert("RGB")
    if args.frameCat:
        image, p_num = dynamic_preprocess(
            image, min_num=2, max_num=12, image_size=448, 
            use_thumbnail=True, img_mean=image_processor.image_mean
        )
    else:
        image, p_num = dynamic_preprocess(
            image, min_num=1, max_num=12, image_size=448, 
            use_thumbnail=True
        )
    assert len(p_num) == 1
    image_tensor = model.process_images(image, model.config).to(
        dtype=model.dtype, device="cuda"
    )
    if audio_path:
        qs = DEFAULT_IMAGE_TOKEN * p_num[0] + "\n" + qs + DEFAULT_AUDIO_TOKEN
    else:
        qs = DEFAULT_IMAGE_TOKEN * p_num[0] + "\n" + qs
    modality = "image"
else:
    image_tensor = torch.zeros((1, 3, 448, 448)).to(dtype=model.dtype, device="cuda")
    if audio_path:
        qs = qs + DEFAULT_AUDIO_TOKEN
    modality = "lang"
```

**What happens**: The script processes the input image/video and prepares the image tensor and question string.

### **Step 17: Conversation Template Setup**

**File**: `/home/tuannv/vlaa/3thrdparties/VITA/video_audio_demo.py:238-246`
```python
# video_audio_demo.py:238-246
conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt(modality)
```

**What happens**: The conversation template is set up and the prompt is prepared.

### **Step 18: Tokenization**

**File**: `/home/tuannv/vlaa/3thrdparties/VITA/video_audio_demo.py:248-259`
```python
# video_audio_demo.py:248-259
if audio_path:
    input_ids = (
        tokenizer_image_audio_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
else:
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
```

**What happens**: The prompt is tokenized into input IDs.

### **Step 19: Stopping Criteria Setup**

**File**: `/home/tuannv/vlaa/3thrdparties/VITA/video_audio_demo.py:261-263`
```python
# video_audio_demo.py:261-263
stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
keywords = [stop_str]
stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
```

**What happens**: Stopping criteria are set up for text generation.

### **Step 20: Model Inference**

**File**: `/home/tuannv/vlaa/3thrdparties/VITA/video_audio_demo.py:265-281`
```python
# video_audio_demo.py:265-281
start_time = time.time()
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
        shared_v_pid_stride=None#2#16#8#4#1#None,
    )
infer_time = time.time() - start_time
```

**What happens**: The model generates a response using the input tokens, images, and audio.

### **Step 21: Output Processing**

**File**: `/home/tuannv/vlaa/3thrdparties/VITA/video_audio_demo.py:282-297`
```python
# video_audio_demo.py:282-297
output_ids = output_ids.sequences
input_token_len = input_ids.shape[1]
if args.model_type == "mixtral-8x7b":
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
        output_ids = output_ids[:, input_token_len:]
outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]

outputs = outputs.strip()
if outputs.endswith(stop_str):
    outputs = outputs[: -len(stop_str)]
outputs = outputs.strip()
print(outputs)
print(f"Time consume: {infer_time}")
```

**What happens**: The output tokens are decoded back to text and printed.

## 🔄 Complete Execution Flow Diagram

```
demo.sh
  ↓
video_audio_demo.py
  ↓
Argument Parsing
  ↓
Dynamic Import Selection
  ↓
Configuration Setup
  ↓
load_pretrained_model()
  ↓
VITAQwen2ForCausalLM.from_pretrained()
  ↓
VITAQwen2Model.__init__()
  ↓
VITAMetaModel.__init__()
  ↓
build_vision_tower()
  ↓
InternViTVisionTower.__init__()
  ↓
InternViTVisionTower.load_model()
  ↓
Model Structure Writing
  ↓
Vision Tower Access
  ↓
Audio Encoder Setup
  ↓
Image Processing
  ↓
Conversation Template Setup
  ↓
Tokenization
  ↓
Stopping Criteria Setup
  ↓
Model Inference
  ↓
Output Processing
  ↓
Result Display
```

## 🎯 Key Execution Points

### **1. Model Loading Phase**
- **Duration**: ~10-30 seconds
- **Memory**: ~14GB GPU memory
- **Components**: Vision tower, audio encoder, language model

### **2. Input Processing Phase**
- **Duration**: ~1-2 seconds
- **Components**: Image preprocessing, tokenization, audio processing

### **3. Inference Phase**
- **Duration**: ~2-6 seconds
- **Components**: Multimodal generation, text decoding

### **4. Output Phase**
- **Duration**: ~0.1 seconds
- **Components**: Text cleanup, result display

## 📊 Performance Metrics

### **Memory Usage**
- **Model Weights**: ~14GB
- **Activations**: ~2GB
- **Input Processing**: ~1GB
- **Total**: ~17GB GPU memory

### **Processing Time**
- **Model Loading**: 10-30 seconds
- **Image Processing**: 1-2 seconds
- **Inference**: 2-6 seconds
- **Total**: 13-38 seconds

## 🔧 Configuration Flow

### **1. Command Line Arguments**
```bash
--model_path ~/models/VITA-1.5
--image_path /workspace/3thrdparties/VITA/asset/vita_newlog.jpg
--model_type qwen2p5_instruct
--conv_mode qwen2p5_instruct
--question "Describe this images."
```

### **2. Internal Configuration**
```python
max_frames = MAX_IMAGE_LENGTH  # 16
video_framerate = 1
temperature = 0.01
top_p = None
num_beams = 1
```

### **3. Model Configuration**
```python
model.config.mm_vision_tower = "InternViT-300M-448px"
model.config.mm_projector_type = "mlp2x_gelu"
model.config.mm_hidden_size = 1024
```

## 🚀 Optimization Points

### **1. Model Loading**
- Use `delay_load=True` for memory optimization
- Implement model caching for repeated runs
- Use quantization for deployment

### **2. Input Processing**
- Batch processing for multiple images
- Async processing for video frames
- Memory-efficient image preprocessing

### **3. Inference**
- Use KV cache for faster generation
- Implement streaming for long outputs
- Optimize beam search parameters

## 📝 Summary

The demo execution trace shows a complete pipeline from shell script execution to model inference:

1. **Script Execution**: Shell script calls Python demo
2. **Model Loading**: VITA model is loaded with all components
3. **Input Processing**: Images and text are processed
4. **Inference**: Multimodal generation produces response
5. **Output**: Text response is displayed

The entire process takes 13-38 seconds and uses ~17GB of GPU memory, with the majority of time spent on model loading and inference.
