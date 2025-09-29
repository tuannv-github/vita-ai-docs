# VITA-1.5 Web Demo Server Documentation

This document provides comprehensive documentation for the VITA-1.5 web demo server implementation, including architecture, data flow, and detailed code analysis.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Data Flow Diagram](#data-flow-diagram)
- [Core Components](#core-components)
  - [LLM Generation and Output Collection](#llm-generation-and-output-collection-details)
  - [Understanding Logits and Conversion to Text/Audio](#understanding-logits-and-conversion-to-textaudio)
  - [Tokenization Process](#tokenization-process)
  - [Conversation History Management](#conversation-history-management)
  - [Voice Activity Detection (VAD) System](#voice-activity-detection-vad-system)
- [Client-Side Audio and Image Streaming](#client-side-audio-and-image-streaming)
- [Data Transmission Timing](#data-transmission-timing)
- [WebSocket Events](#websocket-events)
- [Multiprocessing Architecture](#multiprocessing-architecture)
- [Code Analysis](#code-analysis)
- [Configuration](#configuration)
  - [Command Line Arguments](#command-line-arguments)
  - [Flask App and Socket Configuration](#flask-app-and-socket-configuration)
  - [Global Constants](#global-constants)
- [Deployment](#deployment)

## 🌟 Overview

The VITA-1.5 web demo server is a real-time multimodal AI interaction system that supports:

### 🚀 Quick Start - Flask App & Socket Configuration

**Default Server Settings:**
- **IP**: `127.0.0.1` (localhost)
- **Port**: `8081`
- **Protocol**: HTTPS with self-signed SSL
- **Access URL**: `https://127.0.0.1:8081`

**Quick Launch:**
```bash
python web_demo/server.py --model_path /path/to/vita-1.5
```

**Network Access:**
```bash
python web_demo/server.py --model_path /path/to/vita-1.5 --ip 0.0.0.0 --port 8081
```

**Key Features:**
- **Real-time audio processing** with Voice Activity Detection (VAD)
- **Video frame capture** and processing
- **Multimodal inference** (text, image, audio, video)
- **Text-to-Speech (TTS)** generation
- **WebSocket-based communication** for low-latency interaction
- **Multi-user support** with session management

## 🏗 Architecture

### System Components

```mermaid
graph TB
    subgraph "Client Layer"
        A[Web Browser]
        B[WebSocket Client]
    end
    
    subgraph "Web Server Layer"
        C["Flask App<br/>File: web_demo/server.py:71-75"]
        D["SocketIO Server<br/>File: web_demo/server.py:73"]
        E["User Session Manager<br/>File: web_demo/server.py:75,800-820"]
    end
    
    subgraph "Processing Layer"
        F["Audio Processing<br/>File: web_demo/server.py:850-895"]
        G["Video Processing<br/>File: web_demo/server.py:897-920"]
        H["VAD System<br/>File: web_demo/server.py:720-786"]
    end
    
    subgraph "Model Workers"
        I["LLM Worker 1<br/>File: web_demo/server.py:155-430"]
        J["TTS Worker<br/>File: web_demo/server.py:431-641"]
        K["Global History<br/>File: web_demo/server.py:984,642-718"]
    end
    
    subgraph "Queue System"
        L["Request Queue<br/>File: web_demo/server.py:965,1050"]
        M["TTS Input Queue<br/>File: web_demo/server.py:966,1051"]
        N["TTS Output Queue<br/>File: web_demo/server.py:967,1052"]
    end
    
    A --> C
    B --> D
    C --> E
    D --> F
    D --> G
    F --> H
    H --> L
    L --> I
    I --> M
    M --> J
    J --> N
    N --> D
    I --> K
```

## 📊 Data Flow Diagram

```mermaid
sequenceDiagram
    participant Client as Web Client
    participant Server as "Flask Server<br/>File: web_demo/server.py:792-820"
    participant VAD as "VAD System<br/>File: web_demo/server.py:720-786"
    participant LLM as "LLM Worker<br/>File: web_demo/server.py:155-430"
    participant TTS as "TTS Worker<br/>File: web_demo/server.py:431-641"
    participant Queue as "Message Queues<br/>File: web_demo/server.py:965-967"
    
    Note over Client,Queue: Real-time Audio Processing Flow
    
    Client->>Server: WebSocket Connect<br/>File: web_demo/server.py:792-808
    Server->>Client: Connection Established
    
    Client->>Server: Audio Stream (PCM)<br/>File: web_demo/server.py:850-895
    Server->>VAD: Process Audio Chunk<br/>File: web_demo/server.py:720-786
    VAD->>VAD: Voice Activity Detection
    
    alt Voice Activity Detected
        VAD->>Server: Start Recording<br/>File: web_demo/server.py:822-834
        Server->>Client: Recording Started
    end
    
    alt End of Speech Detected
        VAD->>Server: Audio Complete<br/>File: web_demo/server.py:745-786
        Server->>Queue: Put Request<br/>File: web_demo/server.py:785
        Queue->>LLM: Process Multimodal Input<br/>File: web_demo/server.py:205-320
        LLM->>LLM: Generate Response<br/>File: web_demo/server.py:380-387
        LLM->>Queue: Put TTS Request<br/>File: web_demo/server.py:415-416
        Queue->>TTS: Generate Speech<br/>File: web_demo/server.py:621-641
        TTS->>Queue: Put Audio Output<br/>File: web_demo/server.py:640
        Queue->>Server: Audio Data<br/>File: web_demo/server.py:862-878
        Server->>Client: Stream Audio
    end
    
    Note over Client,Queue: Video Processing Flow
    
    Client->>Server: Video Frame<br/>File: web_demo/server.py:897-920
    Server->>Server: Store Frame Buffer<br/>File: web_demo/server.py:910-915
    Server->>Server: Create Video on Speech End<br/>File: web_demo/server.py:763-765
    
    Note over Client,Queue: Session Management
    
    Server->>Server: User Timeout Check<br/>File: web_demo/server.py:78-87
    alt Timeout
        Server->>Client: Disconnect<br/>File: web_demo/server.py:810-820
        Server->>Server: Cleanup Resources<br/>File: web_demo/server.py:929-956
    end
```

## 🔧 Core Components

### 1. Flask Application Setup

```python
# File: web_demo/server.py:71-75
# Initialize Flask application with custom template and static folders
app = Flask(__name__, 
    template_folder='./vita_html/web/resources', 
    static_folder='./vita_html/web/static'
)
socketio = SocketIO(app)  # WebSocket support for real-time communication
connected_users = {}  # Track active user sessions
```

### 2. User Session Management

```python
# File: web_demo/server.py:78-87
def disconnect_user(sid):
    """
    Disconnect user due to timeout and cleanup resources.
    
    Args:
        sid (str): Session ID of user to disconnect
    """
    if sid in connected_users:
        print(f"Disconnecting user {sid} due to time out")
        socketio.emit('out_time', to=sid)  # Notify client of timeout
        connected_users[sid][0].cancel()  # Cancel timeout timer
        connected_users[sid][1].interrupt()  # Interrupt audio processing
        connected_users[sid][1].stop_pcm = True  # Stop PCM processing
        connected_users[sid][1].release()  # Release audio resources
        time.sleep(3)  # Wait for cleanup
        del connected_users[sid]  # Remove from active users
```

### 3. Multimodal Input Processing

```python
# File: web_demo/server.py:205-320
def _process_inputs(inputs):
    """
    Process multimodal inputs (image, audio, video) for model inference.
    
    Args:
        inputs (dict): Input data containing prompt and multimodal data
        
    Returns:
        dict: Processed inputs ready for model inference
    """
    
    def _process_image(image_path):
        """Process image input - supports file path or numpy array"""
        if isinstance(image_path, str):
            assert os.path.exists(image_path), f"Image file {image_path} does not exist."
            return Image.open(image_path).convert("RGB").transpose(Image.FLIP_LEFT_RIGHT)
        else:
            assert isinstance(image_path, np.ndarray), "Image must be either a file path or a numpy array."
            return Image.fromarray(image_path).convert("RGB").transpose(Image.FLIP_LEFT_RIGHT)

    def _process_audio(audio_path):
        """Process audio input using feature extractor"""
        assert os.path.exists(audio_path), f"Audio file {audio_path} does not exist."
        audio, sr = torchaudio.load(audio_path)
        audio_features = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")["input_features"]
        audio_features = audio_features.squeeze(0)
        return audio_features
    
    def _process_video(video_path, max_frames=4, min_frames=4, s=None, e=None):
        """Process video input by extracting frames"""
        # Handle time range parameters
        if s is None or e is None:
            start_time, end_time = None, None
        else:
            start_time = int(s)
            end_time = int(e)
            start_time = max(start_time, 0)
            end_time = max(end_time, 0)
            if start_time > end_time:
                start_time, end_time = end_time, start_time
            elif start_time == end_time:
                end_time = start_time + 1

        # Load video using decord for efficient decoding
        if os.path.exists(video_path):
            vreader = VideoReader(video_path, ctx=cpu(0))
        else:
            raise FileNotFoundError(f"Video file {video_path} does not exist.")

        # Calculate frame positions
        fps = vreader.get_avg_fps()
        f_start = 0 if start_time is None else int(start_time * fps)
        f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
        num_frames = f_end - f_start + 1
        
        if num_frames > 0:
            # Sample frames based on max/min frame requirements
            all_pos = list(range(f_start, f_end + 1))
            if len(all_pos) > max_frames:
                sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
            elif len(all_pos) < min_frames:
                sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=min_frames, dtype=int)]
            else:
                sample_pos = all_pos

            # Extract and return frames
            patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
            return patch_images
        else:
            print("video path: {} error.".format(video_path))

    # Process multimodal data based on input type
    if "multi_modal_data" in inputs:
        # Process images
        if "image" in inputs["multi_modal_data"]:
            image_inputs = inputs["multi_modal_data"]["image"]
            if not isinstance(image_inputs, list):
                image_inputs = [image_inputs]
            inputs["multi_modal_data"]["image"] = [_process_image(f) for f in image_inputs]
            
            # Validate image token count matches input count
            if "prompt" in inputs:
                assert inputs["prompt"].count(IMAGE_TOKEN) == len(image_inputs), \
                    f"Number of image token {IMAGE_TOKEN} in prompt must match the number of image inputs."

        # Process audio
        if "audio" in inputs["multi_modal_data"]:
            audio_inputs = inputs["multi_modal_data"]["audio"]
            if not isinstance(audio_inputs, list):
                audio_inputs = [audio_inputs]
            inputs["multi_modal_data"]["audio"] = [_process_audio(f) for f in audio_inputs]
            
            # Validate audio token count matches input count
            if "prompt" in inputs:
                assert inputs["prompt"].count(AUDIO_TOKEN) == len(inputs["multi_modal_data"]["audio"]), \
                    f"Number of audio token {AUDIO_TOKEN} in prompt must match the number of audio inputs."

        # Process video (converts to image frames)
        if "video" in inputs["multi_modal_data"]:
            video_inputs = inputs["multi_modal_data"]["video"]
            if not isinstance(video_inputs, list):
                video_inputs = [video_inputs]

            assert "prompt" in inputs, "Prompt must be provided when video inputs are provided."
            assert "image" not in inputs["multi_modal_data"], "Image inputs are not supported when video inputs are provided."
            assert inputs["prompt"].count(VIDEO_TOKEN) == 1, "Currently only one video token is supported in prompt."
            assert inputs["prompt"].count(VIDEO_TOKEN) == len(inputs["multi_modal_data"]["video"]), \
                f"Number of video token {VIDEO_TOKEN} in prompt must match the number of video inputs."
            
            # Convert video to image frames
            video_frames_inputs = []
            for video_input in video_inputs:
                video_frames_inputs.extend(_process_video(video_input, max_frames=4, min_frames=4))
            
            # Replace video token with image tokens
            inputs["prompt"] = inputs["prompt"].replace(VIDEO_TOKEN, IMAGE_TOKEN * len(video_frames_inputs))
            if "image" not in inputs["multi_modal_data"]:
                inputs["multi_modal_data"]["image"] = []
            inputs["multi_modal_data"]["image"].extend(video_frames_inputs)
            inputs["multi_modal_data"].pop("video", None)

    return inputs
```

### 4. LLM Worker Process

```python
# File: web_demo/server.py:155-430
def load_model(
    llm_id,
    engine_args,
    cuda_devices,
    inputs_queue,
    outputs_queue,
    tts_outputs_queue,
    stop_event,
    other_stop_event,
    worker_ready,
    wait_workers_ready,
    start_event,
    other_start_event,
    start_event_lock,
    global_history,
    global_history_limit=0,
):
    """
    LLM worker process that handles multimodal inference.
    
    Args:
        llm_id (int): Worker ID for identification
        engine_args (str): Model path for loading
        cuda_devices (str): CUDA device specification
        inputs_queue (Queue): Input request queue
        outputs_queue (Queue): Output queue for TTS
        tts_outputs_queue (Queue): TTS output queue
        stop_event (Event): Stop signal for this worker
        other_stop_event (Event): Stop signal for other workers
        worker_ready (Event): Worker ready signal
        wait_workers_ready (list): List of workers to wait for
        start_event (Event): Start signal
        other_start_event (Event): Other worker start signal
        start_event_lock (Lock): Synchronization lock
        global_history (list): Global conversation history
        global_history_limit (int): History limit
    """
    print(f"Starting Model Worker {llm_id} with CUDA devices: {cuda_devices}")
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    
    # Import CUDA-dependent packages
    import torch
    import torchaudio
    from vllm import LLM, SamplingParams
    from transformers import AutoFeatureExtractor, AutoTokenizer
    from decord import VideoReader, cpu
    from vita.model.language_model.vita_qwen2 import VITAQwen2Config, VITAQwen2ForCausalLM
    
    # Wait for other workers to initialize
    if len(wait_workers_ready) > 1:
        wait_workers_ready[1].wait()
    
    # Initialize vLLM engine
    llm = LLM(
        model=engine_args,
        dtype="float16",
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,  # Reduced for single GPU
        disable_custom_all_reduce=True,
        limit_mm_per_prompt={'image':256,'audio':50},  # Limit multimodal tokens
    )

    # Load tokenizer and feature extractor
    tokenizer = AutoTokenizer.from_pretrained(engine_args, trust_remote_code=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(engine_args, subfolder="feature_extractor", trust_remote_code=True)

    # Set sampling parameters
    sampling_params = SamplingParams(temperature=0.001, max_tokens=512, best_of=1, skip_special_tokens=False)

    # Main processing loop
    while True:
        # Wait for all workers to be ready
        if not all([worker.is_set() for worker in wait_workers_ready]):
            time.sleep(0.1)
            continue

        if not inputs_queue.empty():
            print(f"[DEBUG] Worker {llm_id}: Queue not empty, processing request...")
            
            # Get request from queue
            inputs = inputs_queue.get()
            print(f"[DEBUG] Worker {llm_id}: Got request: {inputs.get('prompt', 'No prompt')[:100]}...")
            
            # Process multimodal inputs
            inputs = _process_inputs(inputs)
            current_inputs = inputs.copy()
            
            # Merge with conversation history
            inputs = merge_current_and_history(
                global_history[-global_history_limit:],
                inputs,
                skip_history_vision=True,
                move_image_token_to_start=True
            )
        
            # Process prompt tokens
            if "prompt" in inputs:
                inputs["prompt_token_ids"] = tokenizer_image_audio_token(
                    inputs["prompt"], tokenizer, 
                    image_token_index=IMAGE_TOKEN_INDEX, 
                    audio_token_index=AUDIO_TOKEN_INDEX
                )
            else:
                assert "prompt_token_ids" in inputs, "Either 'prompt' or 'prompt_token_ids' must be provided."

            print(f"Process {cuda_devices} is processing inputs: {inputs}")
            inputs.pop("prompt", None)

            # Generate response using vLLM
            print(f"[DEBUG] Worker {llm_id}: Starting LLM generation...")
            llm_start_time = time.time()
            output = llm.generate(inputs, sampling_params=sampling_params)
            llm_end_time = time.time()
            print(f"[DEBUG] Worker {llm_id}: LLM generation completed")
            print(f"{Colors.GREEN}LLM process time: {llm_end_time - llm_start_time}{Colors.RESET}")

            # Process output
            llm_output = output[0].outputs[0].text
            print(f"LLM output: {llm_output}")
            llm_output = '$$FIRST_SENTENCE_MARK$$' + llm_output  # Mark first sentence

            # Stream results to TTS
            async def collect_results(llm_output):
                results = []
                is_first_time_to_work = True
                history_generated_text = ''
                
                for newly_generated_text in llm_output:
                    is_negative = False  # Simplified negative detection

                    if not is_negative:
                        history_generated_text += newly_generated_text
                        
                        if is_first_time_to_work:
                            print(f"Process {cuda_devices} is starting work")
                            stop_event.clear()
                            clear_queue(outputs_queue)
                            clear_queue(tts_outputs_queue)
                            is_first_time_to_work = False

                        if not stop_event.is_set():
                            results.append(newly_generated_text)
                            history_generated_text = history_generated_text.replace('☞ ', '').replace('☞', '')
                            
                            # Send to TTS on punctuation
                            if newly_generated_text in [",", "，", ".", "。", "?", "\n", "？", "!", "！", "、"]:
                                outputs_queue.put({"id": llm_id, "response": history_generated_text})
                                history_generated_text = ''
                        else:
                            print(f"Process {cuda_devices} is interrupted.")
                            break
                    else:
                        print(f"Process {cuda_devices} is generating negative text.")
                        break
                
                # Update global history
                current_inputs["response"] = "".join(results)
                if not current_inputs["response"] == "":
                    global_history.append(current_inputs)
                return results

            # Run async collection
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(collect_results(llm_output))
```

#### LLM Generation and Output Collection Details

The VITA server uses **vLLM** (a high-performance LLM inference engine) for model serving. Here's a detailed breakdown of how `llm.generate` works and how model output is collected:

##### 1. **LLM Initialization**
```python
# File: web_demo/server.py:189-198
llm = LLM(
    model=engine_args,  # Path to the VITA model
    dtype="float16",
    tensor_parallel_size=1,
    trust_remote_code=True,
    gpu_memory_utilization=0.85,
    disable_custom_all_reduce=True,
    limit_mm_per_prompt={'image':256,'audio':50},  # Limits for multimodal inputs
)
```

##### 2. **Sampling Parameters Configuration**
```python
# File: web_demo/server.py:203
sampling_params = SamplingParams(
    temperature=0.001,  # Very low temperature for deterministic output
    max_tokens=512,     # Maximum tokens to generate
    best_of=1,          # Number of sequences to generate
    skip_special_tokens=False  # Keep special tokens in output
)
```

##### 3. **Input Processing Pipeline**
Before calling `llm.generate`, inputs go through extensive preprocessing:

1. **Multimodal Data Processing**: Images, audio, and video are processed and converted to appropriate formats
2. **Tokenization**: The prompt is tokenized with special tokens for images (`<image>`) and audio (`<audio>`)
3. **History Merging**: Previous conversation history is merged with the current request
4. **Prompt Construction**: System prompts and conversation format are applied

##### 4. **LLM Generation Call**
```python
# File: web_demo/server.py:382-384
output = llm.generate(inputs, sampling_params=sampling_params)
```

**Key Points:**
- `inputs` contains the processed multimodal data and tokenized prompt
- The `prompt` field is removed before generation (only `prompt_token_ids` and `multi_modal_data` remain)
- vLLM handles the actual inference using the VITA model

##### 5. **Output Collection and Processing**

###### **Understanding `output[0].outputs[0].text`**

The vLLM `generate()` method returns a complex nested structure. Here's the breakdown:

```python
# File: web_demo/server.py:382-389
output = llm.generate(inputs, sampling_params=sampling_params)
llm_output = output[0].outputs[0].text
```

**Data Structure Explanation:**
```python
# vLLM output structure:
output = [
    RequestOutput(  # output[0] - First (and only) request
        request_id="req_123",
        prompt="<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
        prompt_token_ids=[1, 2, 3, ...],
        outputs=[  # outputs - List of generated sequences
            CompletionOutput(  # outputs[0] - First completion
                index=0,
                text="Hello! How can I help you today?",  # .text - The actual generated text
                token_ids=[4, 5, 6, ...],
                cumulative_logprob=0.0,
                logprobs=None,
                finish_reason="stop"
            )
        ],
        finished=True
    )
]
```

**Toy Example:**
```python
# Example vLLM response structure
output = [
    RequestOutput(
        request_id="req_001",
        prompt="<|im_start|>user\nWhat is AI?<|im_end|>\n<|im_start|>assistant\n",
        outputs=[
            CompletionOutput(
                index=0,
                text="AI, or Artificial Intelligence, is a branch of computer science that focuses on creating intelligent machines capable of performing tasks that typically require human intelligence.",
                token_ids=[151, 44, 12, 8, 25, ...],
                finish_reason="stop"
            )
        ]
    )
]

# Extracting the text:
generated_text = output[0].outputs[0].text
# Result: "AI, or Artificial Intelligence, is a branch of computer science..."
```

###### **Immediate Output Extraction**
```python
# File: web_demo/server.py:389-392
llm_output = output[0].outputs[0].text
print(f"LLM ouput: {llm_output}")
# First sentence mark
llm_output = '$$FIRST_SENTENCE_MARK$$' + llm_output
```

The output is extracted from the vLLM response structure and marked for first sentence processing.

###### **Streaming Results Processing**
The server includes a `stream_results` function for streaming generation:

```python
# File: web_demo/server.py:327-334
async def stream_results(results_generator) -> AsyncGenerator[bytes, None]:
    previous_text = ""
    async for request_output in results_generator:
        text = request_output.outputs[0].text
        newly_generated_text = text[len(previous_text):]
        previous_text = text
        yield newly_generated_text
```

###### **Async Mechanism and Character-by-Character Processing**

The server uses an async mechanism to process the generated text character by character. Here's how it works:

**Async Setup:**
```python
# File: web_demo/server.py:488-490
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
results = loop.run_until_complete(collect_results(llm_output))
```

**Why Async?**
- **Non-blocking Processing**: Allows other operations to continue while processing text
- **Real-time Streaming**: Enables immediate TTS processing as text is generated
- **Interruption Handling**: Can be stopped mid-processing if needed

**Toy Example - Async Processing Flow:**
```python
# Simulated character-by-character processing
llm_output = "Hello, how are you today?"

async def collect_results_example(text):
    results = []
    history_text = ''
    
    for char in text:  # Process each character
        print(f"Processing character: '{char}'")
        
        # Add to history
        history_text += char
        
        # Check for punctuation (trigger TTS)
        if char in [",", ".", "?", "!"]:
            print(f"Punctuation found! Sending to TTS: '{history_text}'")
            # In real code: outputs_queue.put({"id": llm_id, "response": history_text})
            history_text = ''  # Reset for next segment
        
        results.append(char)
    
    return results

# Run the async function
import asyncio
results = asyncio.run(collect_results_example("Hello, how are you today?"))
```

**Character-by-Character Processing:**
```python
# File: web_demo/server.py:394-427
async def collect_results(llm_output):
    results = []
    is_first_time_to_work = True
    history_generated_text = ''
    
    for newly_generated_text in llm_output:
        # is_negative = judge_negative(newly_generated_text)
        is_negative = False

        if not is_negative:
            history_generated_text += newly_generated_text
            if is_first_time_to_work:
                print(f"Process {cuda_devices} is starting work")
                stop_event.clear()
                clear_queue(outputs_queue)
                clear_queue(tts_outputs_queue)
                is_first_time_to_work = False

            if not stop_event.is_set():
                results.append(newly_generated_text)
                history_generated_text = history_generated_text.replace('☞ ', '').replace('☞', '')                            
                if newly_generated_text in [",", "，", ".", "。", "?", "\n", "？", "!", "！", "、"]:
                    outputs_queue.put({"id": llm_id, "response": history_generated_text})
                    history_generated_text = ''
            else:
                print(f"Process {cuda_devices} is interrupted.")
                break
        else:
            print(f"Process {cuda_devices} is generating negative text.")
            break
    
    current_inputs["response"] = "".join(results)
    if not current_inputs["response"] == "":
        global_history.append(current_inputs)
    return results
```

**Toy Example - Complete Processing Flow:**
```python
# Simulated complete processing with toy data
def simulate_llm_processing():
    # 1. LLM generates text
    llm_output = "Hello, I'm Vita! How can I help you today?"
    
    # 2. Add first sentence marker
    llm_output = '$$FIRST_SENTENCE_MARK$$' + llm_output
    
    # 3. Process character by character
    results = []
    history_text = ''
    tts_segments = []
    
    for char in llm_output:
        history_text += char
        
        # Remove special tokens
        if char in ['☞', '☟']:
            continue
            
        results.append(char)
        
        # Check for punctuation (trigger TTS)
        if char in [",", ".", "!", "?"]:
            tts_segments.append(history_text.strip())
            print(f"TTS Segment: '{history_text.strip()}'")
            history_text = ''
    
    print(f"Final result: {''.join(results)}")
    print(f"TTS Segments: {tts_segments}")
    
    return results, tts_segments

# Run simulation
results, tts_segments = simulate_llm_processing()
```

**Key Async Benefits:**
1. **Real-time Processing**: Text is processed as it's generated
2. **TTS Integration**: Punctuation triggers immediate TTS processing
3. **Interruption Support**: Can be stopped mid-processing
4. **Queue Management**: Efficient communication with TTS worker
5. **History Management**: Maintains conversation context

###### **Streaming vs Non-Streaming Generation**

The VITA server supports both streaming and non-streaming generation modes:

**Non-Streaming Mode (Current Implementation):**
```python
# File: web_demo/server.py:382-384
output = llm.generate(inputs, sampling_params=sampling_params)
llm_output = output[0].outputs[0].text  # Complete text at once
```

**Streaming Mode (Available but not used):**
```python
# File: web_demo/server.py:327-334
async def stream_results(results_generator) -> AsyncGenerator[bytes, None]:
    previous_text = ""
    async for request_output in results_generator:
        text = request_output.outputs[0].text
        newly_generated_text = text[len(previous_text):]
        previous_text = text
        yield newly_generated_text
```

**Toy Example - Streaming vs Non-Streaming:**
```python
# Non-streaming: Get complete response at once
def non_streaming_example():
    output = llm.generate(inputs, sampling_params=sampling_params)
    complete_text = output[0].outputs[0].text
    print(f"Complete response: {complete_text}")
    # Process entire text at once
    return complete_text

# Streaming: Get text incrementally
async def streaming_example():
    results_generator = llm.generate(inputs, sampling_params=sampling_params, stream=True)
    previous_text = ""
    
    async for request_output in results_generator:
        current_text = request_output.outputs[0].text
        new_text = current_text[len(previous_text):]
        previous_text = current_text
        
        print(f"New text chunk: '{new_text}'")
        # Process each chunk immediately
        
        # Trigger TTS on punctuation
        if new_text in [",", ".", "!", "?"]:
            print(f"TTS trigger: '{current_text}'")
    
    return previous_text

# Comparison
print("Non-streaming: Wait for complete response, then process")
print("Streaming: Process text as it's generated, better for real-time TTS")
```

**Why Non-Streaming is Used:**
- **Simplicity**: Easier to implement and debug
- **Consistency**: Predictable output format
- **Error Handling**: Better error recovery
- **TTS Integration**: Character-by-character processing provides similar real-time benefits

###### **Understanding Logits and Conversion to Text/Audio**

**What are Logits?**
Logits are raw, unnormalized probability scores that the model outputs for each possible token in the vocabulary. They represent the model's confidence for each possible next token.

**Logits to Text Conversion Process:**

```python
# File: web_demo/server.py:82 (from vita_qwen2.py)
hidden_states = outputs[0]  # Hidden states from transformer layers
logits = self.lm_head(hidden_states)  # Linear projection to vocab size
```

**Step-by-Step Conversion:**

1. **Hidden States → Logits:**
```python
# File: vita/model/language_model/vita_qwen2.py:82-83
hidden_states = outputs[0]  # Shape: [batch_size, seq_len, hidden_size]
logits = self.lm_head(hidden_states)  # Shape: [batch_size, seq_len, vocab_size]
```

2. **Logits → Probabilities:**
```python
# Apply softmax to convert logits to probabilities
import torch.nn.functional as F
probabilities = F.softmax(logits, dim=-1)  # Shape: [batch_size, seq_len, vocab_size]
```

3. **Probabilities → Token Selection:**
```python
# Greedy decoding (select highest probability token)
next_token_id = torch.argmax(probabilities, dim=-1)  # Shape: [batch_size, seq_len]

# Or sampling with temperature
temperature = 0.001  # Low temperature for deterministic output
scaled_logits = logits / temperature
probabilities = F.softmax(scaled_logits, dim=-1)
next_token_id = torch.multinomial(probabilities, num_samples=1)
```

4. **Token IDs → Text:**
```python
# Convert token IDs back to text using tokenizer
generated_text = tokenizer.decode(next_token_id, skip_special_tokens=True)
```

**Toy Example - Complete Logits to Text Flow:**
```python
def simulate_logits_to_text():
    # 1. Model outputs logits for vocabulary
    vocab_size = 50000  # Example vocabulary size
    seq_len = 1  # Generating one token at a time
    batch_size = 1
    
    # Simulate logits (raw scores for each token)
    logits = torch.randn(batch_size, seq_len, vocab_size)
    print(f"Logits shape: {logits.shape}")
    print(f"Sample logits: {logits[0, 0, :5]}")  # First 5 logits
    
    # 2. Convert to probabilities
    probabilities = F.softmax(logits, dim=-1)
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Sum of probabilities: {probabilities.sum(dim=-1)}")  # Should be 1.0
    
    # 3. Select next token (greedy)
    next_token_id = torch.argmax(probabilities, dim=-1)
    print(f"Selected token ID: {next_token_id}")
    
    # 4. Convert to text (simplified)
    # In real implementation: tokenizer.decode(next_token_id)
    print(f"Generated token: {next_token_id.item()}")
    
    return next_token_id

# Run simulation
next_token = simulate_logits_to_text()
```

**Logits to Audio Conversion (TTS Process):**

The VITA system uses a two-stage process for audio generation:

1. **Text → Embeddings:**
```python
# File: web_demo/server.py:620
embeddings = llm_embedding(torch.tensor(tokenizer.encode(tts_input_text)).to(device))
```

2. **Embeddings → Audio Tokens:**
```python
# File: vita/model/vita_tts/decoder/llm2tts.py:142-143
for next_token_id in self.infer(hidden, top_k, prefix, penalty_window_size, penalty):
    token = torch.cat([token, next_token_id], dim=-1)
```

3. **Audio Tokens → Audio Waveform:**
```python
# File: vita/model/vita_tts/decoder/llm2tts.py:145-147
syn = self.codec_model.vqvae(token.unsqueeze(-1), 
                             torch.tensor(self.codec_model.vqvae.h.global_tokens, 
                             device=token.device).unsqueeze(0).unsqueeze(0))
```

**Toy Example - Audio Generation Flow:**
```python
def simulate_text_to_audio():
    # 1. Text input
    text = "Hello, how are you?"
    print(f"Input text: {text}")
    
    # 2. Text → Token IDs
    token_ids = tokenizer.encode(text)  # [101, 7592, 117, 1291, 1128, 136, 102]
    print(f"Token IDs: {token_ids}")
    
    # 3. Token IDs → Embeddings
    embeddings = model.get_input_embeddings()(torch.tensor(token_ids))
    print(f"Embeddings shape: {embeddings.shape}")  # [seq_len, embedding_dim]
    
    # 4. Embeddings → Audio Tokens (simplified)
    # In real TTS: LLM generates audio token logits
    audio_token_logits = tts_model(embeddings)  # [seq_len, audio_vocab_size]
    audio_tokens = torch.argmax(audio_token_logits, dim=-1)
    print(f"Audio tokens: {audio_tokens}")
    
    # 5. Audio Tokens → Audio Waveform
    audio_waveform = codec_model.decode(audio_tokens)
    print(f"Audio waveform shape: {audio_waveform.shape}")  # [samples]
    
    return audio_waveform

# Run simulation
audio = simulate_text_to_audio()
```

**Key Differences:**

| Process | Input | Output | Model Component |
|---------|-------|--------|-----------------|
| **Text Generation** | Hidden States | Text Logits | `lm_head` (Linear layer) |
| **Audio Generation** | Text Embeddings | Audio Token Logits | TTS LLM + Codec |
| **Text Decoding** | Text Logits | Token IDs | `argmax` or sampling |
| **Audio Decoding** | Audio Tokens | Waveform | VQ-VAE Codec |

**Sampling Parameters in VITA:**
```python
# File: web_demo/server.py:203
sampling_params = SamplingParams(
    temperature=0.001,  # Very low = deterministic
    max_tokens=512,     # Maximum generation length
    best_of=1,          # Number of sequences to generate
    skip_special_tokens=False  # Keep special tokens
)
```

**Why Low Temperature (0.001)?**
- **Deterministic Output**: Low temperature makes the model more confident in its predictions
- **Consistent Responses**: Reduces randomness in generated text
- **Better for TTS**: More predictable text leads to better audio generation
- **Real-time Performance**: Faster processing with greedy decoding

##### 6. **Output Flow Architecture**

The collected output flows through multiple stages:

1. **LLM Generation** → Raw text output
2. **Character Processing** → Individual characters with punctuation detection
3. **TTS Queue** → Text segments sent to Text-to-Speech worker
4. **Audio Generation** → TTS worker converts text to audio
5. **WebSocket Streaming** → Audio data sent to client in real-time

##### 7. **Key Features**

- **Real-time Streaming**: Text is processed character-by-character and sent to TTS as soon as punctuation is encountered
- **Interruption Handling**: The generation can be stopped via `stop_event`
- **Multimodal Support**: Handles images, audio, and video inputs through special token processing
- **History Management**: Maintains conversation context across multiple turns
- **Queue-based Architecture**: Uses multiprocessing queues for communication between LLM and TTS workers

##### 8. **Performance Monitoring**

The server tracks generation timing:
```python
# File: web_demo/server.py:381-387
llm_start_time = time.time()
output = llm.generate(inputs, sampling_params=sampling_params)
llm_end_time = time.time()
print(f"[DEBUG] Worker {llm_id}: LLM generation completed")
print(f"{Colors.GREEN}LLM process time: {llm_end_time - llm_start_time}{Colors.RESET}")
```

This architecture enables real-time multimodal conversation with the VITA model, where text generation, TTS conversion, and audio streaming happen concurrently for a responsive user experience.

#### Tokenization Process

The VITA server implements a sophisticated tokenization system for handling multimodal inputs. Here's how it works:

##### 1. **Multimodal Token Processing**
```python
# File: web_demo/server.py:127-153
def tokenizer_image_audio_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, audio_token_index=AUDIO_TOKEN_INDEX, return_tensors=None):
    """
    Tokenize prompt with special tokens for images and audio.
    
    Args:
        prompt (str): Input prompt containing <image> and <audio> tokens
        tokenizer: HuggingFace tokenizer
        image_token_index (int): Token index for images (51000)
        audio_token_index (int): Token index for audio (51001)
        return_tensors (str): Tensor format ('pt' for PyTorch)
        
    Returns:
        list or torch.Tensor: Tokenized input IDs
    """
    prompt_chunks = []
    for chunk in re.split(r'(<audio>|<image>)', prompt):
        if chunk == '<audio>':
            prompt_chunks.append([audio_token_index])
        elif chunk == '<image>':
            prompt_chunks.append([image_token_index])
        else:
            prompt_chunks.append(tokenizer(chunk).input_ids)
    
    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in prompt_chunks:
        if x != [image_token_index] and x != [audio_token_index]:
            input_ids.extend(x[offset:])
        else:
            input_ids.extend(x[:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.LongTensor(input_ids)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids
```

##### 2. **Token Constants**
```python
# File: web_demo/server.py:61-65
IMAGE_TOKEN_INDEX = 51000  # Special token index for images
AUDIO_TOKEN_INDEX = 51001  # Special token index for audio
IMAGE_TOKEN = "<image>"    # Image token string
AUDIO_TOKEN = "<audio>"    # Audio token string
VIDEO_TOKEN = "<video>"    # Video token string
```

##### 3. **Token Processing Flow**

1. **Prompt Parsing**: The prompt is split using regex to identify `<image>` and `<audio>` tokens
2. **Token Replacement**: Special tokens are replaced with their corresponding token indices
3. **BOS Token Handling**: Beginning-of-sequence tokens are properly handled
4. **Tensor Conversion**: Final token IDs can be converted to PyTorch tensors

##### 4. **Multimodal Input Validation**
```python
# File: web_demo/server.py:274-295
# Validate image token count matches input count
if "prompt" in inputs:
    assert inputs["prompt"].count(IMAGE_TOKEN) == len(image_inputs), \
        f"Number of image token {IMAGE_TOKEN} in prompt must match the number of image inputs."

# Validate audio token count matches input count  
if "prompt" in inputs:
    assert inputs["prompt"].count(AUDIO_TOKEN) == len(inputs["multi_modal_data"]["audio"]), \
        f"Number of audio token {AUDIO_TOKEN} in prompt must match the number of audio inputs."
```

##### 5. **Video Token Processing**
```python
# File: web_demo/server.py:310-318
# Convert video to image frames and replace video token with image tokens
video_frames_inputs = []
for video_input in video_inputs:
    video_frames_inputs.extend(_process_video(video_input, max_frames=4, min_frames=4))

# Replace video token with multiple image tokens
inputs["prompt"] = inputs["prompt"].replace(VIDEO_TOKEN, IMAGE_TOKEN * len(video_frames_inputs))
if "image" not in inputs["multi_modal_data"]:
    inputs["multi_modal_data"]["image"] = []
inputs["multi_modal_data"]["image"].extend(video_frames_inputs)
```

##### 6. **Token Processing in LLM Worker**
```python
# File: web_demo/server.py:421-432
# Process prompt tokens
if "prompt" in inputs:
    inputs["prompt_token_ids"] = tokenizer_image_audio_token(
        inputs["prompt"], tokenizer, 
        image_token_index=IMAGE_TOKEN_INDEX, 
        audio_token_index=AUDIO_TOKEN_INDEX
    )
else:
    assert "prompt_token_ids" in inputs, "Either 'prompt' or 'prompt_token_ids' must be provided."

print(f"Process {cuda_devices} is processing inputs: {inputs}")
inputs.pop("prompt", None)  # Remove prompt string, keep only token IDs
```

This tokenization system ensures that multimodal inputs (images, audio, video) are properly integrated into the language model's input sequence, enabling the VITA model to process and understand multiple modalities simultaneously.

#### Conversation History Management

The VITA server implements a sophisticated conversation history system that maintains context across multiple turns:

##### 1. **History Merging Function**
```python
# File: web_demo/server.py:642-718
def merge_current_and_history(
    global_history,
    current_request,
    skip_history_vision=False,
    move_image_token_to_start=False
):
    """
    Merge conversation history with current request.
    
    Args:
        global_history (list): Previous conversation turns
        current_request (dict): Current user request
        skip_history_vision (bool): Skip vision data in history
        move_image_token_to_start (bool): Move image tokens to prompt start
        
    Returns:
        dict: Merged request with conversation context
    """
    
    system_prompts = {
        "video": "<|im_start|>system\nYou are an AI robot and your name is Vita...",
        "image": "<|im_start|>system\nYou are an AI robot and your name is Vita...",
        "audio": "<|im_start|>system\nYou are an AI robot and your name is Vita..."
    }
    
    def select_system_prompt(current_request):
        if "multi_modal_data" in current_request:
            if "video" in current_request["multi_modal_data"]:
                return system_prompts["video"]
            elif "image" in current_request["multi_modal_data"]:
                return system_prompts["video"]  # Use video prompt for images
            elif "audio" in current_request["multi_modal_data"]:
                return system_prompts["audio"]
        return system_prompts["audio"]
    
    system_prompt = select_system_prompt(current_request)
    user_prefix = "<|im_start|>user\n"
    bot_prefix = "<|im_start|>assistant\n"
    eos = "<|im_end|>\n"
    
    # Handle first conversation turn
    if len(global_history) == 0:
        current_request["prompt"] = (system_prompt + user_prefix + current_request["prompt"] + eos + bot_prefix)
        return current_request
    
    # Build conversation context
    current_prompt = system_prompt
    current_multi_modal_data = {"image": [], "audio": [], "video": []}
    
    # Add history to current prompt
    for history in global_history:
        assert "prompt" in history, "Prompt must be provided in history."
        assert "response" in history, "Response must be provided in history."
        
        if skip_history_vision:
            history_prompt = history["prompt"].replace(IMAGE_TOKEN, "").replace(VIDEO_TOKEN, "")
        else:
            history_prompt = history["prompt"]
            
        history_prompt = user_prefix + history_prompt + eos + bot_prefix + history["response"] + eos
        
        # Collect multimodal data from history
        for modality in ["image", "audio", "video"]:
            if skip_history_vision and modality in ["image", "video"]:
                continue
            if "multi_modal_data" in history and modality in history["multi_modal_data"]:
                current_multi_modal_data[modality].extend(history["multi_modal_data"][modality])
        current_prompt += history_prompt
    
    # Add current request
    current_prompt += user_prefix + current_request["prompt"] + eos + bot_prefix
    for modality in ["image", "audio", "video"]:
        if "multi_modal_data" in current_request and modality in current_request["multi_modal_data"]:
            current_multi_modal_data[modality].extend(current_request["multi_modal_data"][modality])
    
    # Clean up empty modalities
    for modality in ["image", "audio", "video"]:
        if current_multi_modal_data[modality] == []:
            current_multi_modal_data.pop(modality, None)
    
    # Move image tokens to start if requested
    if move_image_token_to_start:
        num_image_tokens = current_prompt.count(IMAGE_TOKEN)
        current_prompt = current_prompt.replace(IMAGE_TOKEN, "")
        current_prompt = current_prompt.replace(system_prompt, "")
        current_prompt = system_prompt + user_prefix + IMAGE_TOKEN * num_image_tokens + current_prompt.replace(user_prefix,'')
    
    current_request["prompt"] = current_prompt.replace('☞ ','☞').replace('☟ ','☟')
    current_request["multi_modal_data"] = current_multi_modal_data
    
    return current_request
```

##### 2. **History Configuration**
```python
# File: web_demo/server.py:984-985
global_history = manager.list()  # Shared conversation history
global_history_limit = 1         # Number of previous turns to keep
```

##### 3. **History Update Process**
```python
# File: web_demo/server.py:424-427
# Update global history after generation
current_inputs["response"] = "".join(results)
if not current_inputs["response"] == "":
    global_history.append(current_inputs)
```

##### 4. **History Reset Function**
```python
# File: web_demo/server.py:922-927
@socketio.on('reset_state')
def handle_reset_state():
    """Reset conversation history."""
    global_history = current_app.config['GLOBAL_HISTORY']
    while len(global_history) > 0:
        global_history.pop()
    print("Resetting the state")
```

##### 5. **Key Features**

- **Context Preservation**: Maintains conversation context across multiple turns
- **Multimodal History**: Preserves both text and multimodal data from previous interactions
- **Configurable Limits**: Adjustable history length to balance context and performance
- **Vision Skipping**: Option to skip vision data in history to reduce memory usage
- **Token Management**: Proper handling of special tokens in conversation context
- **System Prompts**: Dynamic system prompt selection based on input modality

This history management system enables the VITA model to maintain coherent, context-aware conversations while efficiently managing memory and processing resources.

#### Voice Activity Detection (VAD) System

The VITA server implements a sophisticated Voice Activity Detection system for real-time audio processing:

##### 1. **PCM Audio Processing**
```python
# File: web_demo/server.py:720-786
def send_pcm(sid, request_inputs_queue):
    """
    Sends PCM audio data to the dialogue system for processing.
    """
    chunk_size = connected_users[sid][1].wakeup_and_vad.get_chunk_size()
    
    print(f"Sid: {sid} Start listening")
    while True:
        if connected_users[sid][1].stop_pcm:
            print(f"Sid: {sid} Stop pcm")
            connected_users[sid][1].stop_generate = True 
            connected_users[sid][1].stop_tts = True
            break
            
        time.sleep(0.01)
        e = connected_users[sid][1].pcm_fifo_queue.get(chunk_size)
        if e is None:
            continue

        res = connected_users[sid][1].wakeup_and_vad.predict(e)

        if res is not None:
            if 'start' in res:
                print(f"Sid: {sid} Vad start")
            elif 'cache_dialog' in res:
                print(f"Sid: {sid} Vad end")
                
                # Save audio file
                directory = './chat_history'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                audio_duration = len(res["cache_dialog"]) / target_sample_rate

                if audio_duration < 1:
                    print("The duration of the audio is less than 1s, skipping...")
                    continue

                current_time = datetime.datetime.now()
                timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                audio_filename = f"{directory}/test_dialog_{timestamp}.wav"
                torchaudio.save(audio_filename, res["cache_dialog"].unsqueeze(0), target_sample_rate)

                # Handle video frames if available
                video_filename = None
                if len(connected_users[sid][1].collected_images) > 0:
                    video_filename = f"{directory}/test_video_{timestamp}.mp4"
                    save_video(connected_users[sid][1].collected_images, video_filename)

                # Create request for LLM processing
                print("Start to generate response")
                if video_filename:
                    current_request = {
                        "prompt": "<video><audio>",
                        "multi_modal_data": {
                            "video": [video_filename],
                            "audio": [audio_filename],
                        },
                    }
                else:
                    current_request = {
                        "prompt": "<audio>",
                        "multi_modal_data": {
                            "audio": [audio_filename],
                        },
                    }
                
                print(f"Start to put request into queue {current_request}")
                request_inputs_queue.put(current_request)
```

##### 2. **Audio Processing Configuration**
```python
# File: web_demo/server.py:56-59
decoder_topk = 2                    # TTS decoder top-k sampling
codec_padding_size = 10             # TTS codec padding size
target_sample_rate = 16000          # Target audio sample rate
```

##### 3. **VAD Processing Flow**

1. **Audio Capture**: Real-time PCM audio data from client
2. **Chunk Processing**: Audio processed in configurable chunk sizes
3. **VAD Detection**: Voice activity detection using wakeup_and_vad system
4. **Audio Caching**: Complete audio segments cached when speech ends
5. **File Generation**: Audio saved as WAV files with timestamps
6. **Request Creation**: Multimodal requests created for LLM processing

##### 4. **Audio Quality Control**
```python
# File: web_demo/server.py:751-755
audio_duration = len(res["cache_dialog"]) / target_sample_rate

if audio_duration < 1:
    print("The duration of the audio is less than 1s, skipping...")
    continue
```

##### 5. **Video Frame Integration**
```python
# File: web_demo/server.py:763-765
video_filename = None
if len(connected_users[sid][1].collected_images) > 0:
    video_filename = f"{directory}/test_video_{timestamp}.mp4"
    save_video(connected_users[sid][1].collected_images, video_filename)
```

##### 6. **Key Features**

- **Real-time Processing**: Continuous audio processing with minimal latency
- **Voice Activity Detection**: Automatic detection of speech start/end
- **Audio Quality Control**: Minimum duration requirements for processing
- **Multimodal Integration**: Seamless integration of audio and video data
- **File Management**: Automatic timestamped file generation
- **Queue-based Processing**: Efficient request queuing for LLM processing

This VAD system enables natural, hands-free interaction with the VITA model by automatically detecting when users start and stop speaking, creating a seamless conversational experience.

### 5. TTS Worker Process

```python
# File: web_demo/server.py:431-641
def tts_worker(
    model_path,
    inputs_queue,
    outputs_queue,
    worker_ready,
    wait_workers_ready,
):
    """
    Text-to-Speech worker process that converts text to audio.
    
    Args:
        model_path (str): Path to model checkpoint
        inputs_queue (Queue): Input queue from LLM worker
        outputs_queue (Queue): Output queue for audio data
        worker_ready (Event): Worker ready signal
        wait_workers_ready (list): List of workers to wait for
    """
    print("Starting TTS Worker")
    
    # Import CUDA-dependent packages
    import torch
    import torchaudio
    from vita.model.vita_tts.decoder.llm2tts import llm2TTS
    from vita.model.language_model.vita_qwen2 import VITAQwen2Config, VITAQwen2ForCausalLM
    from transformers import AutoTokenizer
    
    def remove_special_tokens(input_str):
        """Remove special tokens from input text"""
        special_tokens = ['☞', '☟', '☜', '<unk>', '<|im_end|>']
        for token in special_tokens:
            input_str = input_str.replace(token, '')
        return input_str

    def replace_equation(sentence):
        """Replace mathematical symbols with Chinese equivalents for TTS"""
        special_notations = {
            "sin": " sine ",
            "cos": " cosine ",
            "tan": " tangent ",
            "cot": " cotangent ",
            "sec": " secant ",
            "csc": " cosecant ",
            "log": " logarithm ",
            "exp": "e^",
            "sqrt": "根号 ",
            "abs": "绝对值 ",
        }
        
        special_operators = {
            "+": "加",
            "-": "减",
            "*": "乘",
            "/": "除",
            "=": "等于",
            '!=': '不等于',
            '>': '大于',
            '<': '小于',
            '>=': '大于等于',
            '<=': '小于等于',
        }

        greek_letters = {
            "α": "alpha ", "β": "beta ", "γ": "gamma ", "δ": "delta ",
            "ε": "epsilon ", "ζ": "zeta ", "η": "eta ", "θ": "theta ",
            "ι": "iota ", "κ": "kappa ", "λ": "lambda ", "μ": "mu ",
            "ν": "nu ", "ξ": "xi ", "ο": "omicron ", "π": "派 ",
            "ρ": "rho ", "σ": "sigma ", "τ": "tau ", "υ": "upsilon ",
            "φ": "phi ", "χ": "chi ", "ψ": "psi ", "ω": "omega "
        }

        # Apply replacements
        sentence = sentence.replace('**', ' ')
        sentence = re.sub(r'(?<![\d)])-(\d+)', r'负\1', sentence)

        for key in special_notations:
            sentence = sentence.replace(key, special_notations[key]) 
        for key in special_operators:
            sentence = sentence.replace(key, special_operators[key])
        for key in greek_letters:
            sentence = sentence.replace(key, greek_letters[key])

        sentence = re.sub(r'\(?(\d+)\)?\((\d+)\)', r'\1乘\2', sentence)
        sentence = re.sub(r'\(?(\w+)\)?\^\(?(\w+)\)?', r'\1的\2次方', sentence)
        
        return sentence

    # Initialize TTS model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm_embedding = load_model_embemding(model_path).to(device)
    tts = llm2TTS(os.path.join(model_path, 'vita_tts_ckpt/'))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    worker_ready.set()
    if not isinstance(wait_workers_ready, list):
        wait_workers_ready = [wait_workers_ready]

    past_llm_id = 0

    # Main TTS processing loop
    while True:
        # Wait for all workers to be ready
        if not all([worker.is_set() for worker in wait_workers_ready]):
            time.sleep(0.1)
            continue

        tts_input_text = ""
        while not inputs_queue.empty():
            print(f"[DEBUG] TTS Worker: Queue not empty, processing TTS request...")
            time.sleep(0.03)

            stop_at_punc_or_len = False
            response = inputs_queue.get()
            llm_id, newly_generated_text = response["id"], response["response"]
            print(f"[DEBUG] TTS Worker: Got TTS request for LLM ID {llm_id}, text: {newly_generated_text[:50]}...")

            # Process text character by character
            for character in newly_generated_text:
                if past_llm_id != 0 and past_llm_id != llm_id:
                    tts_input_text = ""
                    outputs_queue.put({"id": llm_id, "response": ("|PAUSE|", None, 0.2)})
                
                tts_input_text += character
                past_llm_id = llm_id
                
                # Stop at punctuation or length limit
                if character in [",", "，", ".", "。", "?", "\n", "？", "!", "！", "、"] and len(tts_input_text) >= 5:
                    stop_at_punc_or_len = True
                    break

            if stop_at_punc_or_len:
                break

        if tts_input_text.strip() == "":
            continue

        # Configure TTS parameters based on first sentence
        if '$$FIRST_SENTENCE_MARK$$' in tts_input_text.strip():
            codec_chunk_size = 20
            seg_threshold = 0.1
            tts_input_text = tts_input_text.replace('$$FIRST_SENTENCE_MARK$$', '').replace('，', '。').replace(',', '。')
            IS_FIRST_SENTENCE = True
        else:
            codec_chunk_size = 40
            seg_threshold = 0.015
            IS_FIRST_SENTENCE = False
            
        # Clean and process text
        tts_input_text = remove_special_tokens(tts_input_text)
        tts_input_text = replace_equation(tts_input_text)
        tts_input_text = tts_input_text.lower()

        if tts_input_text.strip() == "":
            continue
        
        # Generate embeddings and run TTS
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embeddings = llm_embedding(torch.tensor(tokenizer.encode(tts_input_text)).to(device))
        
        for seg in tts.run(embeddings.reshape(-1, 896).unsqueeze(0), decoder_topk,
                            None, 
                            codec_chunk_size=codec_chunk_size,
                            codec_padding_size=codec_padding_size,
                            seg_threshold=seg_threshold):

            # Process first sentence differently
            if IS_FIRST_SENTENCE:
                try:
                    split_idx = torch.nonzero(seg.abs() > 0.03, as_tuple=True)[-1][0]
                    seg = seg[:, :, split_idx:]
                except:
                    print('Do not need to split')
                    pass

            # Convert to audio data
            seg = torch.cat([seg], -1).float().cpu()
            audio_data = (seg.squeeze().numpy() * 32768.0).astype(np.int16)
            audio_duration = seg.shape[-1]/24000
            
            # Send audio to output queue
            if past_llm_id == 0 or past_llm_id == llm_id:
                outputs_queue.put({"id": llm_id, "response": (tts_input_text, audio_data, audio_duration)})
```

## 🔌 WebSocket Events

## 📡 Client-Side Audio and Image Streaming

### Audio Streaming Architecture

#### Client-Side Audio Capture
```javascript
// File: demo.html:245-271
// Audio Worklet Node for real-time processing
audioWorklet = new AudioWorkletNode(audioContext, 'audio-processor', {
    processorOptions: {},
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [1],
    bufferSize: 256  // Small buffer for low latency
});

// Real-time audio data streaming
audioWorklet.port.onmessage = (e) => {
    if (!isRecording) return;
    
    const { audio, inputData } = e.data;
    socket.emit('audio', JSON.stringify({ 
        sample_rate: audioContext.sampleRate,  // 16000 Hz
        audio: audio  // Int16Array converted to Uint8Array
    }));
};
```

#### Audio Processing Worklet
```javascript
// File: audio-processor.js:13-30
process(inputs, outputs, parameters) {
    const input = inputs[0];
    const inputChannel = input[0];
    
    if (this.isRecording && inputChannel) {
        // Convert Float32 to Int16 for transmission
        const int16Array = new Int16Array(inputChannel.length);
        for (let i = 0; i < inputChannel.length; i++) {
            int16Array[i] = inputChannel[i] * 0x7FFF;
        }
        
        // Send audio data to main thread
        this.port.postMessage({
            audio: Array.from(new Uint8Array(int16Array.buffer)),
            inputData: Array.from(inputChannel)
        });
    }
}
```

### Image/Video Streaming Architecture

#### Client-Side Video Capture
```javascript
// File: demo.html:493-497
function sendVideoFrame() {
    // Capture frame from video element
    hiddenCtx.drawImage(videoElement, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
    
    // Convert to base64 JPEG (70% quality)
    const imageData = hiddenCanvas.toDataURL('image/jpeg', 0.7);
    
    // Send to server via WebSocket
    socket.emit('video_frame', imageData);
}

// Send frames every 500ms
videoInterval = setInterval(sendVideoFrame, 500);
```

### Data Flow Summary

```mermaid
graph TB
    subgraph "Client Side"
        A1[🎤 Microphone]
        A2[📹 Camera]
        A3[AudioWorkletNode<br/>256 samples]
        A4[Canvas Element]
        A5[WebSocket Client]
    end
    
    subgraph "Data Processing"
        B1[Float32 → Int16<br/>Conversion]
        B2[JPEG Compression<br/>70% Quality]
        B3[Base64 Encoding]
        B4[JSON Serialization]
    end
    
    subgraph "Server Side"
        C1[WebSocket Handler]
        C2[PCM FIFO Queue]
        C3[VAD Processing]
        C4[Frame Collection]
        C5[LLM Processing]
        C6[TTS Generation]
    end
    
    A1 --> A3
    A2 --> A4
    A3 --> B1
    A4 --> B2
    B1 --> B4
    B2 --> B3
    B3 --> B4
    B4 --> A5
    A5 --> C1
    C1 --> C2
    C1 --> C4
    C2 --> C3
    C3 --> C5
    C4 --> C5
    C5 --> C6
    C6 --> A5
```

#### Audio Streaming Flow
1. **Microphone Capture** → AudioWorkletNode (256 sample buffer)
2. **Real-time Processing** → Float32 to Int16 conversion
3. **WebSocket Transmission** → JSON with sample_rate + audio data
4. **Server Processing** → PCM queue → VAD → LLM processing
5. **TTS Response** → Audio bytes back to client

#### Video Streaming Flow
1. **Camera Capture** → Video element → Canvas
2. **Frame Encoding** → JPEG compression (70% quality)
3. **Base64 Encoding** → Data URL format
4. **WebSocket Transmission** → Raw base64 string
5. **Server Processing** → Decode → RGB conversion → Frame collection

### Technical Specifications

#### Audio Specifications
- **Sample Rate**: 16,000 Hz (fixed)
- **Buffer Size**: 256 samples (low latency)
- **Format**: Int16 → Uint8Array for transmission
- **Processing**: Real-time via AudioWorklet

#### Video Specifications
- **Frame Rate**: 2 FPS (500ms intervals)
- **Format**: JPEG with 70% quality
- **Encoding**: Base64 data URL
- **Processing**: RGB conversion on server

## ⏰ Data Transmission Timing

This section details when and how frequently data is transmitted between client and server in the VITA web demo.

### Audio Data Transmission

#### Client → Server (Real-time Audio Streaming)

**Transmission Frequency:**
- **Buffer Size**: 256 samples
- **Sample Rate**: 16,000 Hz
- **Transmission Rate**: Every ~16ms (256/16000 = 0.016 seconds)
- **Continuous**: While `isRecording = true`

```javascript
// File: demo.html:263-266
// Audio data is transmitted continuously while recording
socket.emit('audio', JSON.stringify({ 
    sample_rate: audioContext.sampleRate,  // 16000 Hz
    audio: audio  // Int16Array converted to Uint8Array
}));
```

**Audio Streaming Timeline:**
```
Time: 0ms    16ms   32ms   48ms   64ms   80ms   96ms   112ms  128ms
      |      |      |      |      |      |      |      |      |
      ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼
   [Audio] [Audio] [Audio] [Audio] [Audio] [Audio] [Audio] [Audio] [Audio]
   Buffer  Buffer  Buffer  Buffer  Buffer  Buffer  Buffer  Buffer  Buffer
   (256)   (256)   (256)   (256)   (256)   (256)   (256)   (256)   (256)
```

#### Server → Client (TTS Audio Response)

**Transmission Triggers:**
- **When**: TTS_OUTPUT_QUEUE is not empty
- **Condition**: `if not current_app.config['TTS_OUTPUT_QUEUE'].empty()`
- **Frequency**: As soon as TTS generation completes
- **Format**: Raw audio bytes (binary data)

```python
# File: server.py:876
# TTS audio is sent back when available in the output queue
emit('audio', audio.tobytes())
```

### Video Data Transmission

#### Client → Server (Periodic Frame Streaming)

**Transmission Frequency:**
- **Interval**: 500ms (0.5 seconds)
- **Frame Rate**: 2 FPS
- **Format**: Base64 encoded JPEG (70% quality)
- **Condition**: While video is active

```javascript
// File: demo.html:463
// Video frames are sent every 500ms
videoInterval = setInterval(sendVideoFrame, 500);
```

**Video Streaming Timeline:**
```
Time: 0ms   500ms  1000ms 1500ms 2000ms 2500ms 3000ms 3500ms 4000ms
      |      |      |      |      |      |      |      |      |
      ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼
   [Frame] [Frame] [Frame] [Frame] [Frame] [Frame] [Frame] [Frame] [Frame]
   (JPEG)  (JPEG)  (JPEG)  (JPEG)  (JPEG)  (JPEG)  (JPEG)  (JPEG)  (JPEG)
```

### Control Event Transmission

#### Recording Control Events

```javascript
// File: demo.html:277, 324
socket.emit('recording-started');  // When recording begins
socket.emit('recording-stopped');  // When recording ends
```

#### Server Response Events

```python
# File: server.py:830, 844, 873, 883
socketio.emit('stop_tts', to=sid)  // Stop TTS playback
```

**Transmission Triggers:**
- **Recording Started**: When user clicks start recording
- **Recording Stopped**: When user clicks stop recording
- **TTS Stop**: When new recording starts or TTS timeout occurs

### Transmission Conditions

#### Audio Transmission Conditions

**Client Side:**
- `isRecording = true`
- AudioWorkletNode is active
- Microphone permission granted
- Audio context is running

**Server Side:**
- User session exists (`sid in connected_users`)
- TTS output queue has data
- No TTS timeout or interruption

#### Video Transmission Conditions

**Client Side:**
- Video stream is active
- Camera permission granted
- `videoInterval` is set
- Video element is playing

**Server Side:**
- User session exists
- Video frame data is valid
- No processing errors

### Performance Characteristics

#### Audio Performance
- **Latency**: ~16ms per buffer (very low)
- **Bandwidth**: ~32KB/s (256 samples × 2 bytes × 62.5 buffers/sec)
- **CPU Usage**: Low (hardware-accelerated AudioWorklet)

#### Video Performance
- **Latency**: 500ms (acceptable for video)
- **Bandwidth**: ~50-200KB per frame (depending on content)
- **CPU Usage**: Moderate (JPEG compression)

### Transmission States

#### Active States
- **Recording**: Audio streaming every 16ms
- **Video Active**: Frame streaming every 500ms
- **TTS Playing**: Audio response streaming

#### Inactive States
- **Idle**: No data transmission
- **Paused**: Transmission stopped
- **Error**: Transmission halted with error handling

This architecture ensures **real-time audio interaction** with minimal latency while **efficiently managing video bandwidth** through periodic frame transmission.

## WebSocket Events
- **`audio`**: Real-time audio data streaming
- **`video_frame`**: Periodic image frame transmission
- **`recording-started/stopped`**: Audio session control
- **`reset_state`**: Clear conversation history

#### Performance Optimizations
- **Audio**: Small buffer size (256) for low latency
- **Video**: JPEG compression and 2 FPS to reduce bandwidth
- **Queue Management**: PCM FIFO queue for audio processing
- **Frame Management**: Automatic cleanup of old frames

## 🔌 WebSocket Events

### Connection Management

```python
# File: web_demo/server.py:792-820
@socketio.on('connect')
def handle_connect():
    """
    Handle new WebSocket connections with user limit and session setup.
    """
    if len(connected_users) >= args.max_users:
        print('Too many users connected, disconnecting new user')
        emit('too_many_users')  # Notify client of user limit
        return

    sid = request.sid  # Get session ID
    connected_users[sid] = []  # Initialize user session
    
    # Set up timeout timer
    connected_users[sid].append(Timer(args.timeout, disconnect_user, [sid]))
    connected_users[sid].append(GlobalParams())  # Initialize audio processing params
    connected_users[sid][0].start()  # Start timeout timer
    
    # Start PCM processing thread
    request_queue = current_app.config['REQUEST_QUEUE']
    pcm_thread = threading.Thread(target=send_pcm, args=(sid, request_queue,))
    pcm_thread.start()
    print(f'User {sid} connected')

@socketio.on('disconnect')
def handle_disconnect():
    """
    Handle WebSocket disconnections and cleanup resources.
    """
    sid = request.sid
    if sid in connected_users:
        connected_users[sid][0].cancel()  # Cancel timeout timer
        connected_users[sid][1].interrupt()  # Interrupt audio processing
        connected_users[sid][1].stop_pcm = True  # Stop PCM processing
        connected_users[sid][1].release()  # Release audio resources
        time.sleep(3)  # Wait for cleanup
        del connected_users[sid]  # Remove from active users
    print(f'User {sid} disconnected')
```

### Audio Processing Events

```python
# File: web_demo/server.py:850-895
@socketio.on('audio')
def handle_audio(data):
    """
    Handle real-time audio data from client.
    
    Args:
        data (str): JSON string containing audio data and sample rate
    """
    global last_tts_model_id
    sid = request.sid
    
    if sid in connected_users:
        try:
            # Handle TTS output if available
            if not current_app.config['TTS_OUTPUT_QUEUE'].empty():
                connected_users[sid][0].cancel()  # Reset timeout
                connected_users[sid][0] = Timer(args.timeout, disconnect_user, [sid])
                connected_users[sid][0].start()

                tts_output_queue = current_app.config['TTS_OUTPUT_QUEUE']
                try:
                    output_data = tts_output_queue.get_nowait()
                    print("output_data", output_data)

                    if output_data is not None:
                        llm_id = output_data["id"]
                        _, audio, length = output_data["response"]

                        print(f"llm_id: {llm_id}, last_tts_model_id: {last_tts_model_id}")
                        if last_tts_model_id != llm_id:
                            print(f"Received output from other process {llm_id}, last output tts model is {last_tts_model_id}, skipping...")
                            socketio.emit('stop_tts', to=sid)
                        else:
                            print(f"Sid: {sid} Send TTS data")
                            emit('audio', audio.tobytes())  # Send audio to client

                        last_tts_model_id = llm_id
                except Empty:
                    pass
        
            # Handle TTS timeout
            if connected_users[sid][1].tts_over_time > 0:
                socketio.emit('stop_tts', to=sid)
                connected_users[sid][1].tts_over_time = 0
            
            # Process incoming audio data
            data = json.loads(data)
            audio_data = np.frombuffer(bytes(data['audio']), dtype=np.int16)
            sample_rate = data['sample_rate']
            
            # Add to PCM processing queue
            connected_users[sid][1].pcm_fifo_queue.put(
                torch.tensor(audio_data, dtype=torch.float32) / 32768.0
            )

        except Exception as e:
            print(f"Error processing audio: {e}")
    else:
        disconnect()  # Disconnect if user not in active list
```

### Video Frame Processing

```python
# File: web_demo/server.py:897-920
@socketio.on('video_frame')
def handle_video_frame(data):
    """
    Handle video frames from client for visual context.
    
    Args:
        data (str): Base64 encoded image data
    """
    import cv2
    
    sid = request.sid
    if sid in connected_users:
        try:
            # Decode base64 image data
            image_data = base64.b64decode(data.split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            
            # Manage frame buffer with time-based clearing
            current_time = time.time()
            if current_time - connected_users[sid][1].last_image_time > 1:
                connected_users[sid][1].collected_images.clear()
                print("Clearing the collected images")
            
            # Add frame to collection
            connected_users[sid][1].collected_images.append(frame)
            connected_users[sid][1].last_image_time = current_time
            
        except Exception as e:
            print(f"Error processing video frame: {e}")
    else:
        disconnect()
```

## ⚙️ Multiprocessing Architecture

### Process Initialization

```python
# File: web_demo/server.py:958-1078
if __name__ == "__main__":
    print("Start VITA server")
    
    # 1. Initialize multiprocessing resources
    multiprocessing.set_start_method('spawn', force=True)

    manager = multiprocessing.Manager()
    request_inputs_queue = manager.Queue()  # Queue for LLM requests
    tts_inputs_queue = manager.Queue()      # Queue for TTS requests
    tts_output_queue = manager.Queue()      # Queue for TTS outputs

    # Event synchronization
    worker_1_stop_event = manager.Event() 
    worker_2_stop_event = manager.Event() 
    worker_1_start_event = manager.Event() 
    worker_2_start_event = manager.Event()
    worker_1_start_event.set()

    worker_1_2_start_event_lock = manager.Lock()

    # Worker ready events
    llm_worker_1_ready = manager.Event()
    llm_worker_2_ready = manager.Event()
    tts_worker_ready = manager.Event()
    gradio_worker_ready = manager.Event()

    # Global conversation history
    global_history = manager.list()
    global_history_limit = 1

    # 2. Start worker processes

    # TTS Worker Process
    tts_worker_process = multiprocessing.Process(
        target=tts_worker,
        kwargs={
            "model_path": args.model_path,
            "inputs_queue": tts_inputs_queue,
            "outputs_queue": tts_output_queue,
            "worker_ready": tts_worker_ready,
            "wait_workers_ready": [],  # TTS worker doesn't need to wait
        }
    )

    # LLM Worker Process
    model_1_process = multiprocessing.Process(
        target=load_model,
        kwargs={
            "llm_id": 1,
            "engine_args": args.model_path, 
            "cuda_devices": "0",
            "inputs_queue": request_inputs_queue,
            "outputs_queue": tts_inputs_queue,
            "tts_outputs_queue": tts_output_queue,
            "start_event": worker_1_start_event,
            "other_start_event": worker_2_start_event,
            "start_event_lock": worker_1_2_start_event_lock,
            "stop_event": worker_1_stop_event,
            "other_stop_event": worker_2_stop_event,
            "worker_ready": llm_worker_1_ready,
            "wait_workers_ready": [],  # Model_1 worker doesn't need to wait
            "global_history": global_history,
            "global_history_limit": global_history_limit,
        }
    )

    # 3. Start processes
    model_1_process.start()
    tts_worker_process.start()

    # 4. Add multiprocessing resources to Flask app context
    app.config['REQUEST_QUEUE'] = request_inputs_queue
    app.config['TTS_QUEUE'] = tts_inputs_queue
    app.config['TTS_OUTPUT_QUEUE'] = tts_output_queue
    app.config['WORKER_1_STOP'] = worker_1_stop_event
    app.config['WORKER_2_STOP'] = worker_2_stop_event
    app.config['WORKER_1_START'] = worker_1_start_event
    app.config['WORKER_2_START'] = worker_2_start_event
    app.config['START_LOCK'] = worker_1_2_start_event_lock
    app.config['WORKER_1_READY'] = llm_worker_1_ready
    app.config['WORKER_2_READY'] = llm_worker_2_ready
    app.config['TTS_READY'] = tts_worker_ready
    app.config['GLOBAL_HISTORY'] = global_history
    app.config['MODEL_1_PROCESS'] = model_1_process
    app.config['TTS_WORKER_PROCESS'] = tts_worker_process

    # 5. Start Flask application with SSL
    cert_file = "web_demo/vita_html/web/resources/cert.pem"
    key_file = "web_demo/vita_html/web/resources/key.pem"
    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        generate_self_signed_cert(cert_file, key_file)
    
    socketio.run(app, host=args.ip, port=args.port, debug=False, 
                ssl_context=(cert_file, key_file), allow_unsafe_werkzeug=True)

    # 6. Wait for processes to complete
    model_1_process.join()
    tts_worker_process.join()
```

## 🔧 Configuration

### Command Line Arguments

```python
# File: web_demo/server.py:28-37
def get_args():
    """
    Parse command line arguments for server configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='VITA')
    parser.add_argument('--model_path', help='model_path to load', default='../VITA_ckpt')
    parser.add_argument('--ip', help='ip of server', default='127.0.0.1')
    parser.add_argument('--port', help='port of server', default=8081)
    parser.add_argument('--max_users', type=int, default=2)  # Maximum concurrent users
    parser.add_argument('--timeout', type=int, default=600)  # User timeout in seconds
    args = parser.parse_args()
    print(args)
    return args
```

### Flask App and Socket Configuration

#### Flask App Initialization
```python
# File: web_demo/server.py:72-73
app = Flask(__name__, template_folder='./vita_html/web/resources', static_folder='./vita_html/web/static')
socketio = SocketIO(app)
```

#### Server Launch Configuration
```python
# File: web_demo/server.py:1073
socketio.run(app, host=args.ip, port=args.port, debug=False, ssl_context=(cert_file, key_file), allow_unsafe_werkzeug=True)
```

#### Default Network Settings
- **Default IP**: `127.0.0.1` (localhost only)
- **Default Port**: `8081`
- **Protocol**: HTTPS with self-signed SSL certificates
- **WebSocket**: Enabled for real-time communication

#### Usage Examples

**Local Development (Default)**
```bash
python web_demo/server.py --model_path /path/to/vita-1.5
# Server accessible at: https://127.0.0.1:8081
```

**Network Access (All Interfaces)**
```bash
python web_demo/server.py --model_path /path/to/vita-1.5 --ip 0.0.0.0 --port 8081
# Server accessible from any network interface: https://[YOUR_IP]:8081
```

**Custom Port**
```bash
python web_demo/server.py --model_path /path/to/vita-1.5 --ip 127.0.0.1 --port 9000
# Server accessible at: https://127.0.0.1:9000
```

**Production Deployment**
```bash
python web_demo/server.py --model_path /path/to/vita-1.5 --ip 0.0.0.0 --port 443 --max_users 10 --timeout 1800
# Production server with increased capacity
```

#### Network Configuration Options

| Parameter | Default | Description | Example |
|-----------|---------|-------------|---------|
| `--ip` | `127.0.0.1` | Server IP address | `0.0.0.0` for all interfaces |
| `--port` | `8081` | Server port number | `443` for HTTPS, `80` for HTTP |
| `--max_users` | `2` | Maximum concurrent users | `10` for production |
| `--timeout` | `600` | Session timeout (seconds) | `1800` for longer sessions |

#### SSL/TLS Configuration
- **Automatic SSL**: Self-signed certificates generated automatically
- **Certificate Location**: `web_demo/vita_html/web/resources/`
- **Files**: `cert.pem` (certificate), `key.pem` (private key)
- **Protocol**: HTTPS only (no HTTP fallback)

#### WebSocket Configuration
- **Transport**: WebSocket over HTTPS (WSS)
- **Real-time Events**: Audio streaming, video frames, TTS output
- **Connection Management**: Automatic reconnection, session tracking
- **Message Types**: Binary (audio/video), JSON (control messages)

### Global Constants

```python
# File: web_demo/server.py:56-65
decoder_topk = 2                    # TTS decoder top-k sampling
codec_padding_size = 10             # TTS codec padding size
target_sample_rate = 16000          # Target audio sample rate
last_tts_model_id = 0               # Last TTS model ID for synchronization

# Token indices for multimodal processing
IMAGE_TOKEN_INDEX = 51000           # Image token index
AUDIO_TOKEN_INDEX = 51001           # Audio token index
IMAGE_TOKEN = "<image>"             # Image token string
AUDIO_TOKEN = "<audio>"             # Audio token string
VIDEO_TOKEN = "<video>"             # Video token string
```

## 🚀 Deployment

### SSL Certificate Generation

```python
# File: web_demo/server.py:1069-1073
# Generate self-signed SSL certificates if not present
cert_file = "web_demo/vita_html/web/resources/cert.pem"
key_file = "web_demo/vita_html/web/resources/key.pem"
if not os.path.exists(cert_file) or not os.path.exists(key_file):
    generate_self_signed_cert(cert_file, key_file)
```

### Resource Cleanup

```python
# File: web_demo/server.py:929-956
def cleanup_resources():
    """
    Clean up multiprocessing resources on exit.
    """
    print("正在清理资源...")
    with app.app_context():
        # Stop worker processes
        if 'WORKER_1_STOP' in current_app.config:
            current_app.config['WORKER_1_STOP'].set()
        if 'WORKER_2_STOP' in current_app.config:
            current_app.config['WORKER_2_STOP'].set()
        
        # Clear queues
        if 'REQUEST_QUEUE' in current_app.config:
            clear_queue(current_app.config['REQUEST_QUEUE'])
        if 'TTS_QUEUE' in current_app.config:
            clear_queue(current_app.config['TTS_QUEUE'])
        if 'TTS_OUTPUT_QUEUE' in current_app.config:
            clear_queue(current_app.config['TTS_OUTPUT_QUEUE'])
        
        # Terminate processes
        if 'MODEL_1_PROCESS' in current_app.config:
            current_app.config['MODEL_1_PROCESS'].terminate()
        if 'MODEL_2_PROCESS' in current_app.config:
            current_app.config['MODEL_2_PROCESS'].terminate() 
        if 'TTS_WORKER_PROCESS' in current_app.config:
            current_app.config['TTS_WORKER_PROCESS'].terminate()

# Register cleanup function
atexit.register(cleanup_resources)
```

## 📊 Performance Considerations

### Memory Management
- **GPU Memory**: Limited to 85% utilization to prevent OOM
- **Queue Management**: Automatic queue clearing to prevent memory buildup
- **Process Isolation**: Separate processes for LLM and TTS to isolate memory usage

### Latency Optimization
- **Streaming TTS**: Real-time audio generation and streaming
- **VAD Processing**: Efficient voice activity detection
- **Multiprocessing**: Parallel processing of LLM and TTS

### Scalability
- **User Limits**: Configurable maximum concurrent users
- **Timeout Management**: Automatic user disconnection on timeout
- **Resource Cleanup**: Proper cleanup of resources on disconnect

---

**Note**: This documentation provides a comprehensive overview of the VITA-1.5 web demo server implementation, including real-time multimodal processing, multiprocessing architecture, and WebSocket-based communication.

**Last Updated**: January 2025  
**Server Version**: 1.0  
**Model Version**: VITA-1.5

## 📝 Recent Updates

This documentation has been significantly enhanced with detailed explanations of:

- **LLM Generation Process**: Comprehensive breakdown of how `llm.generate` works with vLLM
- **Model Output Collection**: Detailed explanation of character-by-character processing and streaming
- **Tokenization System**: Complete coverage of multimodal token processing
- **Conversation History Management**: Sophisticated context preservation system
- **Voice Activity Detection**: Real-time audio processing and VAD implementation

The documentation now provides complete technical details for developers working with the VITA web demo server implementation.
