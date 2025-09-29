# VITA Model.generate Callbacks and Streaming Guide

## 📋 Overview

The VITA model's `generate()` method doesn't natively support callbacks, but there are several ways to implement callback-like functionality for monitoring, logging, and streaming text generation. This guide shows you how to add custom behavior during the generation process.

## 🔧 Current Implementation

### **Standard Generation (No Callbacks)**
```python
# Current implementation in video_audio_demo.py
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

## 🚀 Callback Implementation Methods

### **Method 1: TextStreamer for Real-time Output**

The most straightforward way to add callback functionality is using Hugging Face's `TextStreamer`:

```python
from transformers import TextStreamer, TextIteratorStreamer
import threading
import queue

def generate_with_streaming(model, tokenizer, input_ids, images, audios, **kwargs):
    """Generate text with real-time streaming output"""
    
    # Create a text streamer
    streamer = TextStreamer(
        tokenizer, 
        skip_prompt=True,  # Skip the input prompt
        skip_special_tokens=True
    )
    
    # Add streamer to generation parameters
    generation_kwargs = {
        **kwargs,
        'streamer': streamer,
        'pad_token_id': tokenizer.eos_token_id
    }
    
    print("Generating response...")
    output_ids = model.generate(
        input_ids,
        images=images,
        audios=audios,
        **generation_kwargs
    )
    
    return output_ids

# Usage in video_audio_demo.py
def main_with_streaming():
    # ... existing setup code ...
    
    # Generate with streaming
    output_ids = generate_with_streaming(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        images=image_tensor,
        audios=audios,
        do_sample=False,
        temperature=0.01,
        max_new_tokens=1024,
        use_cache=True,
        stopping_criteria=[stopping_criteria]
    )
    
    # Process final output
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return outputs
```

### **Method 2: Custom Generation Loop with Callbacks**

For more control, implement a custom generation loop:

```python
import torch
from typing import Callable, List, Optional

class GenerationCallback:
    """Base class for generation callbacks"""
    
    def on_generation_start(self, input_ids: torch.Tensor, **kwargs):
        """Called at the start of generation"""
        pass
    
    def on_token_generated(self, token_id: int, token_text: str, step: int, **kwargs):
        """Called after each token is generated"""
        pass
    
    def on_generation_end(self, output_ids: torch.Tensor, **kwargs):
        """Called at the end of generation"""
        pass

class LoggingCallback(GenerationCallback):
    """Callback for logging generation progress"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.step_count = 0
        self.start_time = None
    
    def on_generation_start(self, input_ids: torch.Tensor, **kwargs):
        self.start_time = time.time()
        print(f"🚀 Starting generation with {input_ids.shape[1]} input tokens")
    
    def on_token_generated(self, token_id: int, token_text: str, step: int, **kwargs):
        self.step_count += 1
        if self.step_count % 10 == 0:  # Log every 10 tokens
            elapsed = time.time() - self.start_time
            tokens_per_sec = self.step_count / elapsed
            print(f"📝 Step {step}: Generated '{token_text}' (tokens/sec: {tokens_per_sec:.2f})")
    
    def on_generation_end(self, output_ids: torch.Tensor, **kwargs):
        total_time = time.time() - self.start_time
        total_tokens = output_ids.shape[1]
        print(f"✅ Generation complete: {total_tokens} tokens in {total_time:.2f}s")

class ProgressCallback(GenerationCallback):
    """Callback for showing progress bar"""
    
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.current_tokens = 0
    
    def on_generation_start(self, input_ids: torch.Tensor, **kwargs):
        print(f"Progress: [{' ' * 50}] 0%")
    
    def on_token_generated(self, token_id: int, token_text: str, step: int, **kwargs):
        self.current_tokens += 1
        progress = min(self.current_tokens / self.max_tokens, 1.0)
        bar_length = int(progress * 50)
        bar = '█' * bar_length + ' ' * (50 - bar_length)
        print(f"\rProgress: [{bar}] {progress*100:.1f}%", end='', flush=True)
    
    def on_generation_end(self, output_ids: torch.Tensor, **kwargs):
        print()  # New line after progress bar

def generate_with_callbacks(
    model, 
    tokenizer, 
    input_ids, 
    images, 
    audios, 
    callbacks: List[GenerationCallback],
    max_new_tokens: int = 1024,
    **kwargs
):
    """Custom generation with callback support"""
    
    # Initialize generation state
    generated_ids = input_ids.clone()
    past_key_values = None
    
    # Call start callbacks
    for callback in callbacks:
        callback.on_generation_start(input_ids, **kwargs)
    
    try:
        for step in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                outputs = model(
                    input_ids=generated_ids,
                    images=images if step == 0 else None,  # Only pass images on first step
                    audios=audios if step == 0 else None,  # Only pass audios on first step
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
            
            # Get next token
            logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Update generated sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            past_key_values = outputs.past_key_values
            
            # Decode token for callbacks
            token_text = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            
            # Call token callbacks
            for callback in callbacks:
                callback.on_token_generated(
                    token_id=next_token_id.item(),
                    token_text=token_text,
                    step=step,
                    **kwargs
                )
            
            # Check for stopping criteria
            if next_token_id.item() == tokenizer.eos_token_id:
                break
    
    finally:
        # Call end callbacks
        for callback in callbacks:
            callback.on_generation_end(generated_ids, **kwargs)
    
    return generated_ids

# Usage example
def main_with_callbacks():
    # ... existing setup code ...
    
    # Create callbacks
    callbacks = [
        LoggingCallback(tokenizer),
        ProgressCallback(max_new_tokens=1024)
    ]
    
    # Generate with callbacks
    output_ids = generate_with_callbacks(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        images=image_tensor,
        audios=audios,
        callbacks=callbacks,
        max_new_tokens=1024
    )
    
    return output_ids
```

### **Method 3: Iterator Streamer for Async Processing**

For asynchronous processing or web applications:

```python
from transformers import TextIteratorStreamer
import threading
import queue

def generate_with_iterator_streamer(model, tokenizer, input_ids, images, audios, **kwargs):
    """Generate text with iterator streamer for async processing"""
    
    # Create iterator streamer
    streamer = TextIteratorStreamer(
        tokenizer, 
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=30.0  # Timeout in seconds
    )
    
    # Generation parameters
    generation_kwargs = {
        **kwargs,
        'streamer': streamer,
        'pad_token_id': tokenizer.eos_token_id
    }
    
    # Start generation in a separate thread
    generation_thread = threading.Thread(
        target=model.generate,
        args=(input_ids,),
        kwargs={
            'images': images,
            'audios': audios,
            **generation_kwargs
        }
    )
    generation_thread.start()
    
    # Process streamed tokens
    generated_text = ""
    try:
        for new_text in streamer:
            generated_text += new_text
            print(f"Generated: {new_text}", end='', flush=True)
            # Here you can add custom processing for each token
            yield new_text
    except queue.Empty:
        print("Generation timeout")
    
    generation_thread.join()
    return generated_text

# Usage for async processing
def process_streamed_generation():
    for token in generate_with_iterator_streamer(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        images=image_tensor,
        audios=audios,
        max_new_tokens=1024
    ):
        # Process each token as it's generated
        # Could send to web client, save to file, etc.
        pass
```

### **Method 4: Custom Model Wrapper with Hooks**

Create a wrapper that adds callback functionality:

```python
class VITAModelWithCallbacks:
    """Wrapper for VITA model with callback support"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.callbacks = []
    
    def add_callback(self, callback: GenerationCallback):
        """Add a callback to the model"""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: GenerationCallback):
        """Remove a callback from the model"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def generate(self, input_ids, images=None, audios=None, **kwargs):
        """Generate with callback support"""
        
        # Call start callbacks
        for callback in self.callbacks:
            callback.on_generation_start(input_ids, images=images, audios=audios, **kwargs)
        
        # Use the custom generation function
        output_ids = generate_with_callbacks(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=input_ids,
            images=images,
            audios=audios,
            callbacks=self.callbacks,
            **kwargs
        )
        
        return output_ids

# Usage
def main_with_wrapper():
    # ... existing setup code ...
    
    # Create wrapped model
    wrapped_model = VITAModelWithCallbacks(model, tokenizer)
    
    # Add callbacks
    wrapped_model.add_callback(LoggingCallback(tokenizer))
    wrapped_model.add_callback(ProgressCallback(max_new_tokens=1024))
    
    # Generate with callbacks
    output_ids = wrapped_model.generate(
        input_ids=input_ids,
        images=image_tensor,
        audios=audios,
        max_new_tokens=1024
    )
    
    return output_ids
```

## 🎯 Practical Examples

### **Example 1: Enhanced Demo Script with Callbacks**

```python
# Enhanced video_audio_demo.py with callbacks
def enhanced_demo_with_callbacks():
    # ... existing setup code ...
    
    # Create callbacks
    callbacks = [
        LoggingCallback(tokenizer),
        ProgressCallback(max_new_tokens=1024)
    ]
    
    print("🚀 Starting VITA generation with callbacks...")
    
    start_time = time.time()
    output_ids = generate_with_callbacks(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        images=image_tensor,
        audios=audios,
        callbacks=callbacks,
        do_sample=False,
        temperature=0.01,
        max_new_tokens=1024,
        use_cache=True,
        stopping_criteria=[stopping_criteria]
    )
    infer_time = time.time() - start_time
    
    # Process final output
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    print(f"\n✅ Final output: {outputs}")
    print(f"⏱️ Total time: {infer_time:.2f} seconds")
    
    return outputs
```

### **Example 2: Web Application with Streaming**

```python
# For web applications (Flask/FastAPI)
from flask import Flask, jsonify, stream_template
import json

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_streaming():
    """Generate text with streaming response"""
    
    def generate():
        for token in generate_with_iterator_streamer(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            images=image_tensor,
            audios=audios,
            max_new_tokens=1024
        ):
            yield f"data: {json.dumps({'token': token})}\n\n"
    
    return app.response_class(
        generate(),
        mimetype='text/plain'
    )
```

### **Example 3: Performance Monitoring Callback**

```python
class PerformanceCallback(GenerationCallback):
    """Callback for monitoring generation performance"""
    
    def __init__(self):
        self.token_times = []
        self.start_time = None
    
    def on_generation_start(self, input_ids: torch.Tensor, **kwargs):
        self.start_time = time.time()
        print(f"📊 Performance monitoring started")
    
    def on_token_generated(self, token_id: int, token_text: str, step: int, **kwargs):
        current_time = time.time()
        if step > 0:
            token_time = current_time - self.last_time
            self.token_times.append(token_time)
        self.last_time = current_time
    
    def on_generation_end(self, output_ids: torch.Tensor, **kwargs):
        total_time = time.time() - self.start_time
        avg_token_time = sum(self.token_times) / len(self.token_times) if self.token_times else 0
        
        print(f"📈 Performance Stats:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Total tokens: {len(self.token_times)}")
        print(f"   Avg time per token: {avg_token_time*1000:.2f}ms")
        print(f"   Tokens per second: {len(self.token_times)/total_time:.2f}")
```

## 🔧 Integration with Existing Demo

To add callbacks to the existing `demo.sh` script:

1. **Create a new enhanced demo script**:
```bash
# Create enhanced_demo.sh
#!/bin/bash

python /workspace/3thrdparties/VITA/enhanced_video_audio_demo.py \
--model_path ~/models/VITA-1.5 \
--image_path /workspace/3thrdparties/VITA/asset/vita_newlog.jpg \
--model_type qwen2p5_instruct \
--conv_mode qwen2p5_instruct \
--question "Describe this images." \
--enable_callbacks \
--streaming
```

2. **Modify the Python script** to accept callback parameters and use the enhanced generation methods shown above.

## 📊 Benefits of Using Callbacks

1. **Real-time Monitoring**: See generation progress in real-time
2. **Performance Analysis**: Monitor tokens per second, memory usage
3. **User Experience**: Provide progress feedback for long generations
4. **Debugging**: Log intermediate states and token generation
5. **Streaming**: Enable real-time text streaming for web applications
6. **Custom Processing**: Add custom logic for each generated token

## ⚠️ Important Notes

1. **Performance Impact**: Callbacks add overhead to generation
2. **Memory Usage**: Storing intermediate states increases memory usage
3. **Threading**: Iterator streamer requires careful thread management
4. **Error Handling**: Implement proper error handling in callbacks
5. **Compatibility**: Ensure callbacks work with VITA's multimodal inputs

This guide provides multiple approaches to add callback functionality to VITA's generation process, from simple streaming to complex custom generation loops with full callback support.
