# VITA Model Generation Hooks Guide

## 📋 Overview

Generation hooks are more powerful than callbacks because they can intercept and modify the model's internal operations during generation. Unlike callbacks that work at the generation loop level, hooks operate at the PyTorch module level, allowing you to:

- Intercept forward passes
- Modify activations and gradients
- Monitor internal states
- Inject custom logic at any layer
- Debug model behavior

## 🔧 Types of Hooks

### **1. Forward Hooks**
Intercept the forward pass of modules during generation.

### **2. Backward Hooks**
Intercept gradients during training (less relevant for generation).

### **3. Pre/Post Hooks**
Execute before or after module operations.

## 🚀 Implementation Methods

### **Method 1: PyTorch Module Hooks**

```python
import torch
import torch.nn as nn
from typing import Dict, List, Any, Callable
import time

class GenerationHook:
    """Base class for generation hooks"""
    
    def __init__(self, name: str):
        self.name = name
        self.hook_handles = []
        self.data = {}
    
    def register_hook(self, module: nn.Module, hook_fn: Callable):
        """Register a hook on a module"""
        handle = module.register_forward_hook(hook_fn)
        self.hook_handles.append(handle)
        return handle
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

class ActivationMonitorHook(GenerationHook):
    """Hook to monitor activations during generation"""
    
    def __init__(self, target_modules: List[str] = None):
        super().__init__("activation_monitor")
        self.target_modules = target_modules or []
        self.activations = {}
        self.timings = {}
    
    def create_hook_fn(self, module_name: str):
        """Create a hook function for a specific module"""
        
        def hook_fn(module, input, output):
            # Record activation
            if isinstance(output, torch.Tensor):
                self.activations[module_name] = {
                    'shape': output.shape,
                    'dtype': output.dtype,
                    'device': output.device,
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item()
                }
            elif isinstance(output, tuple):
                self.activations[module_name] = {
                    'type': 'tuple',
                    'length': len(output),
                    'shapes': [t.shape if hasattr(t, 'shape') else str(type(t)) for t in output]
                }
            
            # Record timing
            self.timings[module_name] = time.time()
        
        return hook_fn
    
    def register_on_model(self, model):
        """Register hooks on target modules"""
        for name, module in model.named_modules():
            if not self.target_modules or any(target in name for target in self.target_modules):
                hook_fn = self.create_hook_fn(name)
                self.register_hook(module, hook_fn)
                print(f"🔗 Registered hook on: {name}")
    
    def get_summary(self):
        """Get summary of monitored activations"""
        summary = {}
        for name, data in self.activations.items():
            if 'shape' in data:
                summary[name] = {
                    'shape': data['shape'],
                    'mean': f"{data['mean']:.4f}",
                    'std': f"{data['std']:.4f}",
                    'range': f"[{data['min']:.4f}, {data['max']:.4f}]"
                }
        return summary

class AttentionVisualizationHook(GenerationHook):
    """Hook to capture attention weights for visualization"""
    
    def __init__(self):
        super().__init__("attention_visualization")
        self.attention_weights = {}
        self.layer_names = []
    
    def create_attention_hook(self, layer_name: str):
        """Create hook for attention layers"""
        
        def hook_fn(module, input, output):
            # Extract attention weights if available
            if hasattr(module, 'attention_weights'):
                self.attention_weights[layer_name] = module.attention_weights.detach().cpu()
            elif isinstance(output, tuple) and len(output) > 1:
                # Try to extract attention from output tuple
                if hasattr(output[1], 'shape') and len(output[1].shape) == 4:
                    self.attention_weights[layer_name] = output[1].detach().cpu()
        
        return hook_fn
    
    def register_attention_layers(self, model):
        """Register hooks on attention layers"""
        for name, module in model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                hook_fn = self.create_attention_hook(name)
                self.register_hook(module, hook_fn)
                self.layer_names.append(name)
                print(f"👁️ Registered attention hook on: {name}")

class MemoryMonitorHook(GenerationHook):
    """Hook to monitor memory usage during generation"""
    
    def __init__(self):
        super().__init__("memory_monitor")
        self.memory_usage = []
        self.peak_memory = 0
    
    def create_memory_hook(self, module_name: str):
        """Create memory monitoring hook"""
        
        def hook_fn(module, input, output):
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
                
                self.memory_usage.append({
                    'module': module_name,
                    'current_gb': current_memory,
                    'peak_gb': peak_memory,
                    'timestamp': time.time()
                })
                
                self.peak_memory = max(self.peak_memory, peak_memory)
        
        return hook_fn
    
    def register_memory_monitoring(self, model):
        """Register memory monitoring on all modules"""
        for name, module in model.named_modules():
            hook_fn = self.create_memory_hook(name)
            self.register_hook(module, hook_fn)

class CustomLogicHook(GenerationHook):
    """Hook to inject custom logic during generation"""
    
    def __init__(self, custom_function: Callable):
        super().__init__("custom_logic")
        self.custom_function = custom_function
        self.results = []
    
    def create_custom_hook(self, module_name: str):
        """Create custom logic hook"""
        
        def hook_fn(module, input, output):
            # Apply custom function
            result = self.custom_function(module, input, output, module_name)
            self.results.append({
                'module': module_name,
                'result': result,
                'timestamp': time.time()
            })
        
        return hook_fn
    
    def register_custom_logic(self, model, target_modules: List[str]):
        """Register custom logic on target modules"""
        for name, module in model.named_modules():
            if any(target in name for target in target_modules):
                hook_fn = self.create_custom_hook(name)
                self.register_hook(module, hook_fn)

# Example custom function
def analyze_attention_patterns(module, input, output, module_name):
    """Custom function to analyze attention patterns"""
    if 'attention' in module_name.lower():
        # Extract attention weights and analyze
        if isinstance(output, tuple) and len(output) > 1:
            attention = output[1]
            if hasattr(attention, 'shape') and len(attention.shape) == 4:
                # Analyze attention patterns
                attention_entropy = -torch.sum(attention * torch.log(attention + 1e-8), dim=-1)
                return {
                    'entropy_mean': attention_entropy.mean().item(),
                    'entropy_std': attention_entropy.std().item(),
                    'max_attention': attention.max().item()
                }
    return None
```

### **Method 2: VITA-Specific Generation Hooks**

```python
class VITAGenerationHooks:
    """Specialized hooks for VITA model generation"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = {}
        self.generation_data = {}
    
    def setup_vision_encoder_hooks(self):
        """Setup hooks for vision encoder"""
        
        def vision_hook(module, input, output, name):
            if 'vision' in name.lower() or 'image' in name.lower():
                self.generation_data[f'{name}_vision'] = {
                    'input_shape': [inp.shape if hasattr(inp, 'shape') else str(type(inp)) for inp in input],
                    'output_shape': output.shape if hasattr(output, 'shape') else str(type(output)),
                    'timestamp': time.time()
                }
        
        # Register on vision-related modules
        for name, module in self.model.named_modules():
            if any(keyword in name.lower() for keyword in ['vision', 'image', 'patch']):
                hook_fn = lambda m, i, o, n=name: vision_hook(m, i, o, n)
                handle = module.register_forward_hook(hook_fn)
                self.hooks[f'{name}_vision'] = handle
    
    def setup_whale_encoder_hooks(self):
        """Setup hooks for whale encoder (multimodal fusion)"""
        
        def whale_hook(module, input, output, name):
            if 'whale' in name.lower() or 'fusion' in name.lower():
                self.generation_data[f'{name}_whale'] = {
                    'input_shapes': [inp.shape if hasattr(inp, 'shape') else str(type(inp)) for inp in input],
                    'output_shape': output.shape if hasattr(output, 'shape') else str(type(output)),
                    'timestamp': time.time()
                }
        
        # Register on whale/fusion modules
        for name, module in self.model.named_modules():
            if any(keyword in name.lower() for keyword in ['whale', 'fusion', 'multimodal']):
                hook_fn = lambda m, i, o, n=name: whale_hook(m, i, o, n)
                handle = module.register_forward_hook(hook_fn)
                self.hooks[f'{name}_whale'] = handle
    
    def setup_language_model_hooks(self):
        """Setup hooks for language model components"""
        
        def lm_hook(module, input, output, name):
            if any(keyword in name.lower() for keyword in ['lm_head', 'embed', 'transformer']):
                self.generation_data[f'{name}_lm'] = {
                    'input_shapes': [inp.shape if hasattr(inp, 'shape') else str(type(inp)) for inp in input],
                    'output_shape': output.shape if hasattr(output, 'shape') else str(type(output)),
                    'timestamp': time.time()
                }
        
        # Register on language model modules
        for name, module in self.model.named_modules():
            if any(keyword in name.lower() for keyword in ['lm_head', 'embed', 'transformer', 'layer']):
                hook_fn = lambda m, i, o, n=name: lm_hook(m, i, o, n)
                handle = module.register_forward_hook(hook_fn)
                self.hooks[f'{name}_lm'] = handle
    
    def setup_token_generation_hooks(self):
        """Setup hooks to monitor token generation process"""
        
        def token_hook(module, input, output, name):
            if 'lm_head' in name.lower():
                # This is where logits are generated
                logits = output
                if hasattr(logits, 'shape'):
                    # Get top-k predictions
                    top_k = 5
                    top_values, top_indices = torch.topk(logits, top_k, dim=-1)
                    
                    self.generation_data[f'{name}_tokens'] = {
                        'logits_shape': logits.shape,
                        'top_k_values': top_values.detach().cpu().numpy().tolist(),
                        'top_k_indices': top_indices.detach().cpu().numpy().tolist(),
                        'top_k_tokens': [self.tokenizer.decode([idx]) for idx in top_indices[0, -1, :top_k]],
                        'timestamp': time.time()
                    }
        
        # Register on language head
        for name, module in self.model.named_modules():
            if 'lm_head' in name.lower():
                hook_fn = lambda m, i, o, n=name: token_hook(m, i, o, n)
                handle = module.register_forward_hook(hook_fn)
                self.hooks[f'{name}_tokens'] = handle
    
    def remove_all_hooks(self):
        """Remove all registered hooks"""
        for name, handle in self.hooks.items():
            handle.remove()
        self.hooks.clear()
        print(f"🗑️ Removed {len(self.hooks)} hooks")
    
    def get_generation_summary(self):
        """Get summary of generation data"""
        summary = {
            'total_hooks': len(self.hooks),
            'modules_monitored': len(self.generation_data),
            'data_keys': list(self.generation_data.keys())
        }
        return summary
```

### **Method 3: Enhanced Demo with Hooks**

```python
def enhanced_demo_with_hooks():
    """Enhanced demo with comprehensive hook monitoring"""
    
    # ... existing setup code ...
    
    # Create VITA-specific hooks
    vita_hooks = VITAGenerationHooks(model, tokenizer)
    
    # Setup all hook types
    vita_hooks.setup_vision_encoder_hooks()
    vita_hooks.setup_whale_encoder_hooks()
    vita_hooks.setup_language_model_hooks()
    vita_hooks.setup_token_generation_hooks()
    
    # Create additional monitoring hooks
    activation_hook = ActivationMonitorHook(['vision', 'whale', 'lm_head'])
    memory_hook = MemoryMonitorHook()
    attention_hook = AttentionVisualizationHook()
    
    try:
        print("🚀 Starting generation with hooks...")
        
        # Register additional hooks
        activation_hook.register_on_model(model)
        memory_hook.register_memory_monitoring(model)
        attention_hook.register_attention_layers(model)
        
        start_time = time.time()
        
        # Generate with hooks active
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                audios=audios,
                do_sample=False,
                temperature=0.01,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )
        
        infer_time = time.time() - start_time
        
        # Collect hook data
        print("\n📊 Hook Data Summary:")
        print(f"VITA Hooks: {vita_hooks.get_generation_summary()}")
        print(f"Activation Hook: {len(activation_hook.activations)} modules monitored")
        print(f"Memory Peak: {memory_hook.peak_memory:.2f} GB")
        print(f"Attention Layers: {len(attention_hook.attention_weights)} captured")
        
        # Show detailed activation summary
        print("\n🔍 Activation Summary:")
        for name, data in activation_hook.get_summary().items():
            print(f"  {name}: {data['shape']} - mean: {data['mean']}, std: {data['std']}")
        
        # Show token generation details
        print("\n🎯 Token Generation Details:")
        for key, data in vita_hooks.generation_data.items():
            if 'tokens' in key:
                print(f"  {key}:")
                print(f"    Top tokens: {data['top_k_tokens']}")
                print(f"    Logits shape: {data['logits_shape']}")
        
        return output_ids
        
    finally:
        # Clean up hooks
        vita_hooks.remove_all_hooks()
        activation_hook.remove_hooks()
        memory_hook.remove_hooks()
        attention_hook.remove_hooks()
        print("🧹 All hooks cleaned up")

# Usage
if __name__ == "__main__":
    enhanced_demo_with_hooks()
```

### **Method 4: Debugging and Analysis Hooks**

```python
class DebuggingHook(GenerationHook):
    """Hook for debugging generation issues"""
    
    def __init__(self, debug_level: str = "basic"):
        super().__init__("debugging")
        self.debug_level = debug_level
        self.debug_data = {}
        self.nan_count = 0
        self.inf_count = 0
    
    def create_debug_hook(self, module_name: str):
        """Create debugging hook"""
        
        def hook_fn(module, input, output):
            debug_info = {
                'module_name': module_name,
                'timestamp': time.time()
            }
            
            # Check for NaN and Inf values
            if isinstance(output, torch.Tensor):
                if torch.isnan(output).any():
                    self.nan_count += 1
                    debug_info['has_nan'] = True
                    debug_info['nan_locations'] = torch.isnan(output).sum().item()
                
                if torch.isinf(output).any():
                    self.inf_count += 1
                    debug_info['has_inf'] = True
                    debug_info['inf_locations'] = torch.isinf(output).sum().item()
                
                debug_info['output_stats'] = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item()
                }
            
            # Store debug info
            if module_name not in self.debug_data:
                self.debug_data[module_name] = []
            self.debug_data[module_name].append(debug_info)
        
        return hook_fn
    
    def register_debugging(self, model, target_modules: List[str] = None):
        """Register debugging hooks"""
        for name, module in model.named_modules():
            if target_modules is None or any(target in name for target in target_modules):
                hook_fn = self.create_debug_hook(name)
                self.register_hook(module, hook_fn)
    
    def get_debug_summary(self):
        """Get debugging summary"""
        return {
            'total_modules': len(self.debug_data),
            'nan_count': self.nan_count,
            'inf_count': self.inf_count,
            'problematic_modules': [
                name for name, data in self.debug_data.items()
                if any(entry.get('has_nan', False) or entry.get('has_inf', False) for entry in data)
            ]
        }

# Usage for debugging
def debug_generation():
    """Debug generation with comprehensive monitoring"""
    
    # ... setup code ...
    
    debug_hook = DebuggingHook("detailed")
    debug_hook.register_debugging(model, ['vision', 'whale', 'lm_head'])
    
    try:
        output_ids = model.generate(...)
        
        # Check for issues
        summary = debug_hook.get_debug_summary()
        if summary['nan_count'] > 0 or summary['inf_count'] > 0:
            print(f"⚠️ Found {summary['nan_count']} NaN and {summary['inf_count']} Inf values")
            print(f"Problematic modules: {summary['problematic_modules']}")
        else:
            print("✅ No NaN or Inf values detected")
            
    finally:
        debug_hook.remove_hooks()
```

## 🎯 Practical Examples

### **Example 1: Memory Profiling During Generation**

```python
def profile_memory_usage():
    """Profile memory usage during generation"""
    
    memory_hook = MemoryMonitorHook()
    memory_hook.register_memory_monitoring(model)
    
    try:
        output_ids = model.generate(...)
        
        # Analyze memory usage
        memory_data = memory_hook.memory_usage
        peak_usage = max(entry['peak_gb'] for entry in memory_data)
        
        print(f"📊 Memory Analysis:")
        print(f"  Peak memory usage: {peak_usage:.2f} GB")
        print(f"  Memory samples: {len(memory_data)}")
        
        # Find memory-intensive modules
        module_memory = {}
        for entry in memory_data:
            module = entry['module']
            if module not in module_memory:
                module_memory[module] = []
            module_memory[module].append(entry['current_gb'])
        
        # Sort by average memory usage
        avg_memory = {mod: sum(mem)/len(mem) for mod, mem in module_memory.items()}
        top_modules = sorted(avg_memory.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print(f"  Top memory-consuming modules:")
        for module, avg_mem in top_modules:
            print(f"    {module}: {avg_mem:.2f} GB")
            
    finally:
        memory_hook.remove_hooks()
```

### **Example 2: Attention Pattern Analysis**

```python
def analyze_attention_patterns():
    """Analyze attention patterns during generation"""
    
    attention_hook = AttentionVisualizationHook()
    attention_hook.register_attention_layers(model)
    
    try:
        output_ids = model.generate(...)
        
        # Analyze attention weights
        for layer_name, attention in attention_hook.attention_weights.items():
            print(f"👁️ Attention Analysis - {layer_name}:")
            print(f"  Shape: {attention.shape}")
            print(f"  Mean attention: {attention.mean().item():.4f}")
            print(f"  Attention entropy: {(-attention * torch.log(attention + 1e-8)).sum(dim=-1).mean().item():.4f}")
            
            # Find most attended positions
            if len(attention.shape) == 4:  # [batch, heads, seq, seq]
                attention_sum = attention.sum(dim=1)  # Sum over heads
                max_attention = attention_sum.max(dim=-1)[0]
                print(f"  Max attention per position: {max_attention.mean().item():.4f}")
                
    finally:
        attention_hook.remove_hooks()
```

## 🔧 Integration with Existing Demo

To add hooks to the existing demo:

```bash
# Create enhanced demo script
#!/bin/bash

python /workspace/3thrdparties/VITA/enhanced_video_audio_demo_with_hooks.py \
--model_path ~/models/VITA-1.5 \
--image_path /workspace/3thrdparties/VITA/asset/vita_newlog.jpg \
--model_type qwen2p5_instruct \
--conv_mode qwen2p5_instruct \
--question "Describe this images." \
--enable_hooks \
--hook_types "activation,memory,attention,debug"
```

## 📊 Benefits of Using Hooks

1. **Deep Inspection**: Access to internal model states and activations
2. **Performance Monitoring**: Real-time memory and computation tracking
3. **Debugging**: Identify issues like NaN/Inf values or attention problems
4. **Visualization**: Capture attention weights and activation patterns
5. **Custom Logic**: Inject any custom processing at any layer
6. **Non-intrusive**: Don't modify the model architecture

## ⚠️ Important Considerations

1. **Performance Impact**: Hooks add computational overhead
2. **Memory Usage**: Storing hook data increases memory consumption
3. **Hook Management**: Always clean up hooks to prevent memory leaks
4. **Thread Safety**: Hooks are not thread-safe by default
5. **Model Compatibility**: Ensure hooks work with VITA's multimodal architecture

This comprehensive hook system provides deep insights into the VITA model's generation process, enabling advanced debugging, monitoring, and analysis capabilities.
