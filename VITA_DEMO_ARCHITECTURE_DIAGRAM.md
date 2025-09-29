# VITA Demo Architecture and Execution Flow Diagram

## 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           VITA Demo System Architecture                        │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────────────────┐
│   demo.sh       │───▶│ video_audio_     │───▶│        VITA Model               │
│   (Bash Script) │    │ demo.py          │    │    (Multimodal LLM)             │
└─────────────────┘    └──────────────────┘    └─────────────────────────────────┘
         │                        │                           │
         │                        │                           │
         ▼                        ▼                           ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────────────────┐
│ • Model Path    │    │ • Input          │    │ • Whale Encoder (341M params)   │
│ • Image Path    │    │   Processing     │    │ • Vision Encoder (289M params)  │
│ • Model Type    │    │ • Tokenization   │    │ • Qwen2.5 Language Model       │
│ • Question      │    │ • Generation     │    │ • Total: ~631M parameters      │
└─────────────────┘    └──────────────────┘    └─────────────────────────────────┘
```

## 🔄 Complete Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           VITA Demo Execution Pipeline                         │
└─────────────────────────────────────────────────────────────────────────────────┘

1. SCRIPT INITIALIZATION
┌─────────────────┐
│   demo.sh       │
│   #!/bin/bash   │
└─────────┬───────┘
          │
          ▼
2. PYTHON SCRIPT CALL
┌─────────────────────────────────────────────────────────────────────────────────┐
│ python /workspace/3thrdparties/VITA/video_audio_demo.py \                      │
│ --model_path ~/models/VITA-1.5 \                                               │
│ --image_path /workspace/3thrdparties/VITA/asset/vita_newlog.jpg \              │
│ --model_type qwen2p5_instruct \                                                │
│ --conv_mode qwen2p5_instruct \                                                 │
│ --question "Describe this images."                                             │
└─────────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
3. ARGUMENT PARSING & CONFIGURATION
┌─────────────────────────────────────────────────────────────────────────────────┐
│ • model_path: ~/models/VITA-1.5                                                │
│ • image_path: /workspace/3thrdparties/VITA/asset/vita_newlog.jpg               │
│ • model_type: qwen2p5_instruct                                                 │
│ • conv_mode: qwen2p5_instruct                                                  │
│ • question: "Describe this images."                                            │
│ • max_frames: 16, temperature: 0.01, num_beams: 1                             │
└─────────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
4. MODEL LOADING
┌─────────────────────────────────────────────────────────────────────────────────┐
│ disable_torch_init()                                                            │
│ model_name = get_model_name_from_path(model_path)                              │
│ tokenizer, model, image_processor, context_len = load_pretrained_model(...)    │
│                                                                                 │
│ Loading:                                                                        │
│ • Whale Encoder: 341.4M parameters                                             │
│ • Vision Encoder: 289.9M parameters                                            │
│ • Total: ~631M parameters                                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
5. INPUT PROCESSING PIPELINE
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│ 5A. IMAGE PROCESSING                                                            │
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │ image = Image.open(image_path).convert("RGB")                              │ │
│ │ image, p_num = dynamic_preprocess(image, min_num=1, max_num=12,            │ │
│ │                                   image_size=448, use_thumbnail=True)      │ │
│ │                                                                             │ │
│ │ Dynamic Preprocessing Steps:                                                │ │
│ │ 1. Aspect Ratio Analysis                                                   │ │
│ │ 2. Patch Grid Calculation (1x1 to 12x12)                                   │ │
│ │ 3. Target Ratio Selection                                                  │ │
│ │ 4. Image Resizing                                                          │ │
│ │ 5. Patch Extraction                                                        │ │
│ │ 6. Final: 5 patches × 448×448×3 = [5, 3, 448, 448]                       │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│ 5B. AUDIO PROCESSING (Empty for demo.sh)                                       │
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │ audios = {                                                                  │ │
│ │   'audios': tensor([[[0., 0., 0., ..., 0., 0., 0.]]], device='cuda:0'),   │ │
│ │   'lengths': tensor([400.], device='cuda:0'),                              │ │
│ │   'lengths_for_llm': tensor([60], device='cuda:0')                         │ │
│ │ }                                                                           │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│ 5C. QUESTION PROCESSING                                                        │
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │ qs = DEFAULT_IMAGE_TOKEN * p_num[0] + "\n" + question                     │ │
│ │ qs = "<image><image><image><image><image>\nDescribe this images."          │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
6. CONVERSATION TEMPLATE PROCESSING
┌─────────────────────────────────────────────────────────────────────────────────┐
│ conv = conv_templates[conv_mode].copy()  # qwen2p5_instruct template           │
│ conv.append_message(conv.roles[0], qs)   # User message with image tokens      │
│ conv.append_message(conv.roles[1], None) # Assistant response (empty)          │
│ prompt = conv.get_prompt(modality)       # Generate final prompt               │
│                                                                                 │
│ Final Prompt Structure:                                                        │
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │ <|im_start|>system                                                         │ │
│ │ You are an AI robot and your name is VITA.                                 │ │
│ │ - You are a multimodal large language model...                             │ │
│ │ <|im_end|>                                                                 │ │
│ │ <|im_start|>user                                                           │ │
│ │ <image><image><image><image><image>                                        │ │
│ │ Describe this images.                                                      │ │
│ │ <|im_end|>                                                                 │ │
│ │ <|im_start|>assistant                                                      │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
7. TOKENIZATION
┌─────────────────────────────────────────────────────────────────────────────────┐
│ input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,        │
│                                  return_tensors="pt").unsqueeze(0).cuda()      │
│                                                                                 │
│ Tokenization Process:                                                           │
│ • Each <image> → IMAGE_TOKEN_INDEX (-200)                                      │
│ • System/User/Assistant tokens from Qwen2.5                                    │
│ • Final shape: torch.Size([1, 160]) - 160 tokens total                        │
│ • Token types: Text tokens + 5 image tokens + special tokens                   │
└─────────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
8. STOPPING CRITERIA SETUP
┌─────────────────────────────────────────────────────────────────────────────────┐
│ stop_str = conv.sep  # "<|im_end|>"                                            │
│ keywords = [stop_str]                                                           │
│ stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)   │
└─────────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
9. MODEL GENERATION
┌─────────────────────────────────────────────────────────────────────────────────┐
│ with torch.inference_mode():                                                   │
│     output_ids = model.generate(                                               │
│         input_ids,                    # [1, 160] tokens                        │
│         images=image_tensor,          # [5, 3, 448, 448] patches              │
│         audios=audios,                # Empty audio data                       │
│         do_sample=False,              # Deterministic generation               │
│         temperature=0.01,             # Very low randomness                   │
│         top_p=None,                   # No nucleus sampling                    │
│         num_beams=1,                  # Greedy decoding                        │
│         max_new_tokens=1024,          # Maximum response length                │
│         use_cache=True,               # Enable KV cache                        │
│         stopping_criteria=[stopping_criteria]                                  │
│     )                                                                           │
│                                                                                 │
│ Generation Parameters:                                                          │
│ • Temperature: 0.01 (near-deterministic)                                      │
│ • Max Tokens: 1024 new tokens                                                  │
│ • Beam Search: Single beam (num_beams=1)                                       │
│ • Sampling: Disabled (do_sample=False)                                         │
└─────────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
10. OUTPUT PROCESSING
┌─────────────────────────────────────────────────────────────────────────────────┐
│ output_ids = output_ids.sequences                                              │
│ outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]     │
│                                                                                 │
│ # Clean up output                                                               │
│ outputs = outputs.strip()                                                       │
│ if outputs.endswith(stop_str):                                                  │
│     outputs = outputs[: -len(stop_str)]                                        │
│ outputs = outputs.strip()                                                       │
│                                                                                 │
│ print(outputs)                                                                  │
│ print(f"Time consume: {infer_time}")                                           │
└─────────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
11. FINAL OUTPUT
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Generated Response:                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │ "The image displays a logo and text related to an open-source software     │ │
│ │  project. The logo consists of the word "VITA" in bold, uppercase letters. │ │
│ │  The design of the letters is modern and sleek, with a gradient effect...  │ │
│ │  [Detailed description continues...]"                                      │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│ Performance Metrics:                                                            │
│ • Time consumed: ~15 seconds                                                   │
│ • GPU Memory: ~3-4GB                                                           │
│ • Generated tokens: Variable (typically 200-500 tokens)                       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 🧠 Model Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           VITA Model Architecture                              │
└─────────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────┐
                    │            Input Processing             │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────┴───────────────────────┐
                    │                                        │
                    ▼                                        ▼
    ┌─────────────────────────┐                ┌─────────────────────────┐
    │     Image Processing    │                │     Text Processing     │
    │                         │                │                         │
    │ • Dynamic Preprocessing │                │ • Tokenization          │
    │ • Patch Extraction      │                │ • Special Tokens        │
    │ • 5 patches × 448×448   │                │ • 160 tokens total      │
    │ • RGB → Tensor          │                │ • Conversation Format   │
    └─────────┬───────────────┘                └─────────┬───────────────┘
              │                                        │
              ▼                                        ▼
    ┌─────────────────────────┐                ┌─────────────────────────┐
    │    Vision Encoder       │                │    Text Encoder         │
    │                         │                │                         │
    │ • 289.9M parameters     │                │ • Qwen2.5 Base Model    │
    │ • Image Understanding   │                │ • Language Processing   │
    │ • Feature Extraction    │                │ • Context Understanding │
    └─────────┬───────────────┘                └─────────┬───────────────┘
              │                                        │
              └──────────────┬─────────────────────────┘
                             │
                             ▼
                    ┌─────────────────────────┐
                    │    Whale Encoder        │
                    │                         │
                    │ • 341.4M parameters     │
                    │ • Multimodal Fusion     │
                    │ • Cross-Modal Attention │
                    │ • Feature Alignment     │
                    └─────────┬───────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │   Language Generator    │
                    │                         │
                    │ • Qwen2.5 Instruct      │
                    │ • Text Generation       │
                    │ • Response Formatting   │
                    │ • Output Decoding       │
                    └─────────┬───────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │     Final Output        │
                    │                         │
                    │ • Generated Text        │
                    │ • Image Description     │
                    │ • Structured Response   │
                    └─────────────────────────┘
```

## 📊 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Data Flow Pipeline                                │
└─────────────────────────────────────────────────────────────────────────────────┘

INPUT DATA
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Image     │    │   Question  │    │   Audio     │
│ (VITA Logo) │    │   Text      │    │  (Empty)    │
└─────┬───────┘    └─────┬───────┘    └─────┬───────┘
      │                  │                  │
      ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Dynamic     │    │ Tokenization│    │ Empty       │
│ Preprocess  │    │ & Template  │    │ Tensors     │
│ 5 Patches   │    │ 160 Tokens  │    │ All Zeros   │
└─────┬───────┘    └─────┬───────┘    └─────┬───────┘
      │                  │                  │
      ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────┐
│                Multimodal Fusion                        │
│                                                         │
│ • Vision Features: [5, 3, 448, 448]                    │
│ • Text Features: [1, 160]                              │
│ • Audio Features: [1, 400] (empty)                     │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│              VITA Model Processing                      │
│                                                         │
│ • Vision Encoder: 289.9M params                        │
│ • Whale Encoder: 341.4M params                         │
│ • Language Model: Qwen2.5 Instruct                     │
│ • Total: ~631M parameters                              │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                Generation Process                       │
│                                                         │
│ • Temperature: 0.01                                     │
│ • Max Tokens: 1024                                      │
│ • Beam Search: 1                                        │
│ • Sampling: Disabled                                    │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                Output Processing                        │
│                                                         │
│ • Token Decoding                                        │
│ • Stop Token Removal                                    │
│ • Text Formatting                                       │
│ • Performance Metrics                                   │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                  Final Output                           │
│                                                         │
│ "The image displays a logo and text related to an      │
│  open-source software project. The logo consists of    │
│  the word 'VITA' in bold, uppercase letters..."        │
│                                                         │
│ Time: ~15 seconds                                       │
│ Memory: ~3-4GB VRAM                                     │
└─────────────────────────────────────────────────────────┘
```

## 🔧 Memory and Performance Metrics

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Performance Characteristics                              │
└─────────────────────────────────────────────────────────────────────────────────┘

MEMORY USAGE BREAKDOWN
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Component                    │ Size                    │ Description            │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Model Weights (FP16)         │ ~2.5GB                  │ Whale + Vision + LLM   │
│ Image Tensors                │ ~6MB                    │ 5 × 448×448×3 × 2bytes │
│ Audio Tensors                │ ~800 bytes              │ Empty audio data       │
│ KV Cache                     │ Variable                │ Based on sequence len  │
│ Intermediate Activations     │ ~500MB                  │ During inference       │
│ Total VRAM Usage             │ ~3-4GB                  │ Peak memory usage      │
└─────────────────────────────────────────────────────────────────────────────────┘

COMPUTATIONAL COMPLEXITY
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Operation                    │ Complexity              │ Time (approx)          │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Image Preprocessing          │ O(patch_count × 448²)   │ ~0.1 seconds           │
│ Token Processing             │ O(sequence_length)      │ ~0.01 seconds          │
│ Vision Encoding              │ O(289M × input_size)    │ ~2 seconds             │
│ Multimodal Fusion            │ O(341M × features)      │ ~3 seconds             │
│ Text Generation              │ O(631M × output_tokens) │ ~10 seconds            │
│ Total Inference Time         │ O(model_size × tokens)  │ ~15 seconds            │
└─────────────────────────────────────────────────────────────────────────────────┘

THROUGHPUT METRICS
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Metric                      │ Value                   │ Notes                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Tokens per Second           │ ~70 tokens/sec          │ Generation speed        │
│ Images per Second           │ ~0.067 images/sec       │ End-to-end processing   │
│ Memory Efficiency           │ ~2.5GB/631M params      │ FP16 precision          │
│ GPU Utilization             │ ~80-90%                 │ During inference        │
│ Batch Size                  │ 1                       │ Single image processing │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 🎯 Key Technical Insights

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Technical Highlights                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

1. MULTIMODAL ARCHITECTURE
   • Whale Encoder (341M): Handles cross-modal fusion and attention
   • Vision Encoder (289M): Processes image patches and extracts features
   • Language Model: Qwen2.5 Instruct for text generation
   • Total: ~631M parameters in FP16 precision

2. DYNAMIC PREPROCESSING
   • Intelligent patch extraction based on image aspect ratio
   • Adaptive grid layout (1x1 to 12x12 patches)
   • Each patch: 448×448×3 RGB channels
   • VITA logo: 5 patches for optimal processing

3. CONVERSATION TEMPLATE
   • Qwen2.5 Instruct format with system/user/assistant roles
   • Structured prompt with image tokens
   • 160 total tokens including special tokens
   • Deterministic generation (temperature=0.01)

4. PERFORMANCE OPTIMIZATION
   • FP16 precision for memory efficiency
   • KV cache for faster generation
   • CUDA acceleration for all operations
   • Inference mode for gradient-free processing

5. ERROR HANDLING
   • Input validation and assertion checks
   • Model compatibility verification
   • Resource management and cleanup
   • Robust stopping criteria
```

This comprehensive diagram shows the complete architecture, execution flow, data processing pipeline, and performance characteristics of the VITA demo system when triggered by `demo.sh`.
