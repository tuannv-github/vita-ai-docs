# VITA Training Script Documentation

This document provides a comprehensive explanation of the VITA training script located at `/home/tuannv/vlaa/3thrdparties/VITA/vita/train/train.py`.

## 📋 **Overview**

The `train.py` script is the main training entry point for the VITA multimodal model. It handles the complete training pipeline including model initialization, parameter freezing/unfreezing, data loading, and training execution.

## 🏗️ **Script Architecture**

### **Main Components**
```
train.py
├── Imports and Dependencies
├── Utility Functions
├── Data Classes (ModelArguments, TrainingArguments)
├── Parameter Management Functions
├── Model Saving Functions
└── Main Training Function
```

## 📦 **1. Imports and Dependencies**

### **Core Libraries**
```python
# Lines 1-11
import logging
import os
import pathlib
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import transformers
from transformers import set_seed
```

### **VITA-Specific Imports**
```python
# Lines 13-22
from vita import conversation as conversation_lib
from vita.model import *
from vita.train.vita_trainer import VITATrainer
from vita.util.data_utils_video_audio_neg_patch import make_supervised_data_module, DataArguments
#from vita.util.data_utils_video_audio_neg_patch_fo import make_supervised_data_module, DataArguments
#from vita.util.data_utils_video_audio_patch import make_supervised_data_module, DataArguments
#from vita.util.data_utils_video_audio_patch_sf import make_supervised_data_module, DataArguments
#from vita.util.data_utils_video_patch_audio import make_supervised_data_module, DataArguments
```

**Key Imports:**
- **`conversation_lib`**: Handles conversation templates and formatting
- **`vita.model`**: VITA model implementations (VITAQwen2ForCausalLM, etc.)
- **`VITATrainer`**: Custom trainer class extending HuggingFace Trainer
- **`make_supervised_data_module`**: Data loading and preprocessing utilities (currently using `data_utils_video_audio_neg_patch`)

## 🔧 **2. Utility Functions**

### **Random Seed Setting**
```python
# Lines 24-33
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

set_random_seed(42)  # Fixed seed for reproducibility
```

### **Distributed Training Utilities**
```python
# Lines 36-42
local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)
```

**Purpose**: Ensures only the main process (rank 0) prints messages in distributed training.

## 📊 **3. Data Classes**

### **ModelArguments**
```python
# Lines 44-64
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    model_type: Optional[str] = field(default=None)
    version: Optional[str] = field(default=None)
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_audio_mlp_adapter: bool = field(default=False)
    audio_prompt_finetune: bool = field(default=False)
    audio_prompt_num: Optional[int] = field(default=None)
    audio_state_predictor_tuning: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    audio_encoder: Optional[str] = field(default=None)
    freeze_audio_encoder: bool = field(default=True)
    freeze_audio_encoder_adapter: bool = field(default=True)
    unfreeze_vision_tower: bool = field(default=False)
    use_s2: bool = field(default=False)
    pretrain_audio_mlp_adapter: Optional[str] = field(default=None)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
```

**Key Parameters:**
- **`model_name_or_path`**: Path to the base language model
- **`model_type`**: Model architecture type (qwen2p5_instruct, mixtral-8x7b, etc.)
- **`vision_tower`**: Path to the vision encoder model
- **`audio_encoder`**: Path to the audio encoder model
- **`tune_mm_mlp_adapter`**: Whether to train the vision projector
- **`tune_audio_mlp_adapter`**: Whether to train the audio adapter
- **`freeze_*`**: Parameters controlling which components to freeze during training

### **TrainingArguments**
```python
# Lines 66-96
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(default=512)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16)
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
```

**Key Parameters:**
- **`optim`**: Optimizer type (default: adamw_torch)
- **`model_max_length`**: Maximum sequence length (default: 512)
- **`bits`**: Quantization bits (4, 8, or 16)
- **`lora_enable`**: Whether to use LoRA fine-tuning
- **`mm_projector_lr`**: Learning rate for multimodal projector

## 🔄 **4. Parameter Management Functions**

### **DeepSpeed Integration**
```python
# Lines 98-112
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param
```

**Purpose**: Handles parameter gathering in DeepSpeed ZeRO optimization.

### **LoRA State Management**
```python
# Lines 115-146
def get_peft_state_maybe_zero_3(named_params, bias):
    # Extracts LoRA parameters from model state
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    # ... more bias handling logic
    return to_return

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    # Extracts non-LoRA parameters
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    return to_return
```

### **Multimodal Adapter State**
```python
# Lines 149-154
def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {
        k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return
```

**Purpose**: Extracts multimodal adapter parameters (vision projector, audio adapter).

### **LoRA Module Discovery**
```python
# Lines 157-170
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)
```

**Purpose**: Finds all linear layers suitable for LoRA adaptation, excluding multimodal components.

## 💾 **5. Model Saving Functions**

### **Safe Model Saving**
```python
# Lines 173-210
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    
    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ["mm_projector"]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(["embed_tokens", "embed_in"])
        
        weight_to_save = get_mm_adapter_state_maybe_zero_3(
            trainer.model.named_parameters(), keys_to_match
        )
        # Save only the adapter weights
        torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
        return
    
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    
    # Standard model saving
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)
```

**Purpose**: Handles different saving strategies based on training configuration.

## 🚀 **6. Main Training Function**

### **Function Signature and Setup**
```python
# Lines 212-223
def train():
    global local_rank
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
```

### **Quantization Configuration**
```python
# Lines 224-244
bnb_model_from_pretrained_args = {}
if training_args.bits in [4, 8]:
    from transformers import BitsAndBytesConfig
    
    bnb_model_from_pretrained_args.update(
        dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            ),
        )
    )
```

**Key Features:**
- **4-bit/8-bit quantization** support
- **Skip multimodal projector** from quantization
- **Double quantization** for better compression

### **Tokenizer Initialization**
```python
# Lines 246-261
assert model_args.vision_tower is not None
if model_args.model_type in {"mixtral-8x7b", "nemo", "qwen2p5_instruct", "qwen2p5_fo_instruct"}:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

if tokenizer.unk_token is not None and tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token

if model_args.model_type == "llama3-8b":
    tokenizer.pad_token = tokenizer.eos_token
```

### **Model Loading**
```python
# Lines 262-299
if model_args.model_type == "mixtral-8x7b":
    torch_dtype = torch.float16 if training_args.fp16 else torch.bfloat16
    model = VITAMixtralForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
        **bnb_model_from_pretrained_args,
    )
elif model_args.model_type == "qwen2p5_instruct":
    torch_dtype = torch.float16 if training_args.fp16 else torch.bfloat16
    model = VITAQwen2ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
        **bnb_model_from_pretrained_args,
    )
# ... more model types
```

**Supported Model Types:**
- **`mixtral-8x7b`**: VITAMixtralForCausalLM
- **`nemo`**: VITAMistralForCausalLM  
- **`qwen2p5_instruct`**: VITAQwen2ForCausalLM
- **`qwen2p5_fo_instruct`**: VITAFOQwen2ForCausalLM
- **`llama3-8b`**: Special tokenizer handling (pad_token = eos_token)

### **Model Configuration**
```python
# Lines 301-302
model.config.use_cache = False

if model_args.freeze_backbone:
    model.model.requires_grad_(False)
```

### **Quantization Preparation**
```python
# Lines 306-316
if training_args.bits in [4, 8]:
    from peft import prepare_model_for_kbit_training
    
    model.config.torch_dtype = (
        torch.float32
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=training_args.gradient_checkpointing
    )
```

### **Gradient Checkpointing**
```python
# Lines 318-326
if training_args.gradient_checkpointing:
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
```

### **LoRA Configuration**
```python
# Lines 328-345
if training_args.lora_enable:
    from peft import LoraConfig, get_peft_model
    
    lora_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=find_all_linear_names(model),
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    if training_args.bits == 16:
        if training_args.bf16:
            model.to(torch.bfloat16)
        if training_args.fp16:
            model.to(torch.float16)
    rank0_print("Adding LoRA adapters...")
    model = get_peft_model(model, lora_config)
```

### **Conversation Template Setup**
```python
# Lines 347-350
if model_args.version in conversation_lib.conv_templates:
    conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
else:
    conversation_lib.default_conversation = conversation_lib.conv_templates["default"]
```

### **Vision Module Initialization**
```python
# Lines 352-363
model.get_model().initialize_vision_modules(model_args=model_args)

vision_tower = model.get_vision_tower()
vision_tower.to(
    dtype=torch.bfloat16 if training_args.bf16 else torch.float16, 
    device=training_args.device
)
```

### **Audio Module Initialization**
```python
# Lines 354-368
model.config.freeze_audio_encoder = model_args.freeze_audio_encoder
model.config.freeze_audio_encoder_adapter = model_args.freeze_audio_encoder_adapter
model.config.audio_prompt_finetune = model_args.audio_prompt_finetune
model.config.audio_prompt_num = model_args.audio_prompt_num
model.get_model().initialize_audio_modules(model_args=model_args)

audio_encoder = model.get_audio_encoder()
audio_encoder.to(
    dtype=torch.bfloat16 if training_args.bf16 else torch.float16, 
    device=training_args.device
)
```

### **Data Processor Setup**
```python
# Lines 370-375
data_args.image_processor = vision_tower.image_processor
data_args.audio_processor = audio_encoder.audio_processor

model.config.image_aspect_ratio = data_args.image_aspect_ratio
model.config.tokenizer_padding_side = tokenizer.padding_side
model.config.tokenizer_model_max_length = tokenizer.model_max_length
```

### **Parameter Freezing/Unfreezing Logic**

#### **Vision Projector Training**
```python
# Lines 377-383
model.config.tune_mm_mlp_adapter = (
    training_args.tune_mm_mlp_adapter
) = model_args.tune_mm_mlp_adapter
if model_args.tune_mm_mlp_adapter:
    model.requires_grad_(False)
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = True
```

#### **Audio Adapter Training**
```python
# Lines 385-391
model.config.tune_audio_mlp_adapter = (
    training_args.tune_audio_mlp_adapter
) = model_args.tune_audio_mlp_adapter
if model_args.tune_audio_mlp_adapter:
    model.requires_grad_(False)
    for p in model.model.audio_encoder.adpter.parameters():  # Note: "adpter" (typo in code)
        p.requires_grad = True
```

#### **Audio Prompt Fine-tuning**
```python
# Lines 393-406
model.config.audio_prompt_finetune = (
    training_args.audio_prompt_finetune
) = model_args.audio_prompt_finetune
model.config.audio_state_predictor_tuning = (
    training_args.audio_state_predictor_tuning
) = model_args.audio_state_predictor_tuning
if model_args.audio_prompt_finetune or model_args.audio_state_predictor_tuning:
    model.requires_grad_(False)
    if model_args.audio_prompt_finetune:
        for p in model.model.audio_encoder.prompt_embeddings.parameters():
            p.requires_grad = True        
    if model_args.audio_state_predictor_tuning:
        for p in model.predictor_head.parameters():
            p.requires_grad = True
```

#### **Vision Tower Unfreezing**
```python
# Lines 420-425
model.config.unfreeze_vision_tower = (
    training_args.unfreeze_vision_tower
) = model_args.unfreeze_vision_tower
if training_args.unfreeze_vision_tower:
    for p in model.get_model().vision_tower.parameters():
        p.requires_grad = True
```

### **Data Module and Trainer Setup**
```python
# Lines 441-442
data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
trainer = VITATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
```

**Data Loading Module:**
- **Currently using**: `data_utils_video_audio_neg_patch` (line 17)
- **Alternative modules**: Several commented options for different data processing strategies
- **Key function**: `make_supervised_data_module()` creates train/eval datasets

**Key Features of VITATrainer:**
- **Custom data sampling**: `get_modality_length_grouped_indices()` for efficient multimodal batching
- **Length-based grouping**: Groups samples by modality and length for better training efficiency
- **Multimodal batch handling**: Separates vision/audio samples from text-only samples

### **Training Execution**
```python
# Lines 444-447
if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()
trainer.save_state()
```

### **Model Saving**
```python
# Lines 452-463
if training_args.lora_enable:
    state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
    non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        model.config.save_pretrained(training_args.output_dir)
        model.save_pretrained(training_args.output_dir, state_dict=state_dict)
        torch.save(
            non_lora_state_dict,
            os.path.join(training_args.output_dir, "non_lora_trainables.bin"),
        )
else:
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
```

## 🎯 **7. Training Stages and Parameter Control**

### **Stage 1: Vision-Language Alignment (pretrain_mlp_qwen.sh)**
```bash
# Dataset: Pretrain_video0
--tune_mm_mlp_adapter True \
--freeze_audio_encoder True \
--freeze_audio_encoder_adapter True \
--unfreeze_vision_tower False  # (default, not explicitly set)
```

**Parameters:**
- **`tune_mm_mlp_adapter=True`**: Train vision projector (mm_projector)
- **`freeze_audio_encoder=True`**: Freeze audio encoder weights
- **`freeze_audio_encoder_adapter=True`**: Freeze audio adapter (adpter)
- **`unfreeze_vision_tower=False`**: Keep vision tower frozen (default)
- **Dataset**: `Pretrain_video0` (vision-language data)
- **Vision Tower**: SigLIP (siglip-so400m-patch14-384)

### **Stage 2: Audio-Language Alignment (pretrain_audio_mlp_qwen.sh)**
```bash
# Dataset: Pretrain_audio
--tune_mm_mlp_adapter False \
--tune_audio_mlp_adapter True \
--freeze_audio_encoder True \
--freeze_audio_encoder_adapter False \
--unfreeze_vision_tower False  # (default, not explicitly set)
```

**Parameters:**
- **`tune_mm_mlp_adapter=False`**: Freeze vision projector (mm_projector)
- **`tune_audio_mlp_adapter=True`**: Train audio adapter (adpter)
- **`freeze_audio_encoder=True`**: Freeze audio encoder weights
- **`freeze_audio_encoder_adapter=False`**: Train audio adapter (adpter)
- **`unfreeze_vision_tower=False`**: Keep vision tower frozen (default)
- **Dataset**: `Pretrain_audio` (audio-language data)
- **Vision Tower**: InternViT (InternViT-300M-448px)

### **Stage 3: Vision Tower Fine-tuning (finetune_qwen.sh)**
```bash
# Dataset: Pretrain_video0
--tune_mm_mlp_adapter False  # (default, not explicitly set)
--tune_audio_mlp_adapter False  # (default, not explicitly set)
--freeze_audio_encoder True \
--freeze_audio_encoder_adapter True \
--unfreeze_vision_tower True \
--pretrain_mm_mlp_adapter ${OUTPUT_DIR}/llava-s1-pretrain_mlp_video/mm_projector.bin
```

**Parameters:**
- **`tune_mm_mlp_adapter=False`**: Freeze vision projector (loads pretrained weights)
- **`tune_audio_mlp_adapter=False`**: Freeze audio adapter
- **`freeze_audio_encoder=True`**: Freeze audio encoder weights
- **`freeze_audio_encoder_adapter=True`**: Freeze audio adapter
- **`unfreeze_vision_tower=True`**: Train vision tower weights
- **`pretrain_mm_mlp_adapter`**: Load pretrained vision projector from Stage 1
- **Dataset**: `Pretrain_video0` (vision-language data)
- **Learning Rate**: Lower LR (2e-5) with separate mm_projector_lr (2e-6)

### **Stage 4: Task-Specific Fine-tuning (finetuneTask_qwen.sh)**
```bash
# Dataset: Pretrain_video0 (or task-specific data)
--tune_mm_mlp_adapter False  # (default, not explicitly set)
--tune_audio_mlp_adapter False  # (default, not explicitly set)
--freeze_audio_encoder True \
--freeze_audio_encoder_adapter True \
--unfreeze_vision_tower False  # (default, not explicitly set)
```

**Parameters:**
- **`tune_mm_mlp_adapter=False`**: Freeze vision projector
- **`tune_audio_mlp_adapter=False`**: Freeze audio adapter
- **`freeze_audio_encoder=True`**: Freeze audio encoder weights
- **`freeze_audio_encoder_adapter=True`**: Freeze audio adapter
- **`unfreeze_vision_tower=False`**: Keep vision tower frozen
- **Model Path**: Uses output from Stage 3 as starting point
- **Dataset**: `Pretrain_video0` (or task-specific datasets)
- **Model Max Length**: 33,300 (much longer sequences)

### **Training Stages Summary Table**

| Stage | Script | Dataset | Vision Projector | Audio Adapter | Vision Tower | Audio Encoder | Purpose |
|-------|--------|---------|------------------|---------------|--------------|---------------|---------|
| **1** | `pretrain_mlp_qwen.sh` | `Pretrain_video0` | ✅ Train | ❌ Freeze | ❌ Freeze | ❌ Freeze | Vision-Language Alignment |
| **2** | `pretrain_audio_mlp_qwen.sh` | `Pretrain_audio` | ❌ Freeze | ✅ Train | ❌ Freeze | ❌ Freeze | Audio-Language Alignment |
| **3** | `finetune_qwen.sh` | `Pretrain_video0` | 🔄 Load Pretrained | ❌ Freeze | ✅ Train | ❌ Freeze | Vision Tower Fine-tuning |
| **4** | `finetuneTask_qwen.sh` | `Pretrain_video0` | ❌ Freeze | ❌ Freeze | ❌ Freeze | ❌ Freeze | Task-Specific Fine-tuning |

**Legend:**
- ✅ **Train**: Component is trained/updated
- ❌ **Freeze**: Component is frozen (not updated)
- 🔄 **Load Pretrained**: Component loads weights from previous stage

## 🔧 **8. Key Features**

### **1. Multi-Model Support**
- **Qwen2.5-7B-Instruct**: Primary model
- **Mixtral-8x7B**: Alternative model
- **NeMo**: NVIDIA's model
- **Qwen2.5-FO-Instruct**: First-order model

### **2. Quantization Support**
- **4-bit quantization**: BitsAndBytesConfig
- **8-bit quantization**: BitsAndBytesConfig
- **16-bit precision**: FP16/BF16

### **3. LoRA Fine-tuning**
- **Automatic module discovery**: `find_all_linear_names()`
- **Configurable parameters**: r, alpha, dropout
- **Bias handling**: none, all, lora_only

### **4. DeepSpeed Integration**
- **ZeRO optimization**: Parameter partitioning
- **Gradient checkpointing**: Memory optimization
- **Distributed training**: Multi-GPU support

### **5. Progressive Training**
- **Stage-based parameter control**: Freeze/unfreeze components
- **Selective training**: Train only specific adapters
- **Checkpoint resuming**: Resume from previous checkpoints

## 📊 **9. Memory and Performance**

### **Memory Optimization**
- **Gradient checkpointing**: Reduces memory usage
- **Quantization**: 4-bit/8-bit for memory efficiency
- **DeepSpeed ZeRO**: Parameter partitioning
- **Selective training**: Only train necessary components

### **Performance Features**
- **Flash Attention 2**: Efficient attention computation
- **Mixed precision**: FP16/BF16 training
- **Distributed training**: Multi-GPU scaling
- **LoRA**: Efficient fine-tuning

## 🚀 **10. Usage Examples**

### **Basic Training (Stage 1 - Vision-Language Alignment)**
```bash
deepspeed --include localhost:0 vita/train/train.py \
    --deepspeed ./script/deepspeed/zero3.json \
    --model_name_or_path /path/to/Qwen2.5-7B-Instruct \
    --model_type qwen2p5_instruct \
    --version qwen2p5_instruct \
    --dataset_use Pretrain_video0 \
    --vision_tower /path/to/siglip-so400m-patch14-384 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --audio_encoder /path/to/audio-encoder_Mixtral-8x7B_New_dim3584 \
    --freeze_audio_encoder True \
    --freeze_audio_encoder_adapter True \
    --image_aspect_ratio square \
    --bf16 True \
    --output_dir ./output \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-4 \
    --model_max_length 6200 \
    --gradient_checkpointing True
```

### **LoRA Fine-tuning**
```bash
python train.py \
    --model_name_or_path /path/to/qwen2.5-7b \
    --model_type qwen2p5_instruct \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 16 \
    --output_dir ./lora_output
```

### **Quantized Training**
```bash
python train.py \
    --model_name_or_path /path/to/qwen2.5-7b \
    --model_type qwen2p5_instruct \
    --bits 4 \
    --double_quant True \
    --quant_type nf4 \
    --output_dir ./quantized_output
```

## 📝 **11. Summary**

The VITA training script (`train.py`) provides a comprehensive training framework that:

1. **Supports multiple model architectures** (Qwen2.5, Mixtral, NeMo)
2. **Implements progressive training** with stage-based parameter control
3. **Integrates advanced optimization** (LoRA, quantization, DeepSpeed)
4. **Handles multimodal components** (vision, audio, language)
5. **Provides flexible configuration** through command-line arguments
6. **Optimizes memory usage** through various techniques
7. **Supports distributed training** for scalability

This script is the core of VITA's training pipeline, enabling the progressive training strategy that builds multimodal capabilities step by step while maintaining efficiency and scalability.
