# embee.cpp Architecture Overview

## System Architecture

embee.cpp is designed with a modular architecture that consists of several key components that work together to provide efficient inference for transformer-based language models. The design prioritizes:

1. **Modularity**: Components can be swapped, extended, or replaced
2. **Efficiency**: Optimized for CPU-only environments
3. **Flexibility**: Support for various model architectures
4. **Simplicity**: Clean APIs and readable code

The high-level architecture consists of the following components:

```
+----------------+     +----------------+     +----------------+
| Model Loader   |---->| Inference      |---->| Application    |
| & Tokenizer    |     | Engine         |     | Interface      |
+----------------+     +----------------+     +----------------+
      ^                      ^                      ^
      |                      |                      |
+----------------+     +----------------+     +----------------+
| File Format    |     | Operators &    |     | API Layer      |
| & Quantizer    |     | Kernels        |     | (C++/Python)   |
+----------------+     +----------------+     +----------------+
```

## Core Components

### 1. Model Representation

**Model** (`include/embee/model.h`)
- Stores model configuration and weights
- Provides access to tensors by name
- Manages the tokenizer
- Loads models from various formats

**ModelConfig** (`include/embee/types.h`)
- Contains architecture parameters
- Stores metadata about the model
- Defines quantization settings

**Tensor** (`include/embee/types.h`)
- Represents n-dimensional arrays
- Stores tensor metadata and data
- Supports various data types and layouts

### 2. Tokenization

**Tokenizer** (`include/embee/tokenizer.h`)
- Abstract interface for all tokenizers
- Encodes text to token IDs
- Decodes token IDs back to text

**BPETokenizer** and **SentencePieceTokenizer**
- Concrete tokenizer implementations
- Load tokenizer data from model files
- Provide efficient tokenization

### 3. Inference Engine

**Engine** (`include/embee/engine.h`)
- Main inference engine
- Manages model execution
- Handles generation with caching

**Transformer** (Internal)
- Implements the transformer architecture
- Manages attention and feed-forward layers
- Optimizes execution flow

**Attention** and **FeedForward** (Internal)
- Implement core transformer operations
- Optimized for various CPU architectures
- Support different model variants

### 4. Quantization

**Quantizer** (`include/embee/quantizer.h`)
- Handles weight quantization
- Supports various quantization methods
- Provides dequantization during inference

**QuantizationParams** (`include/embee/types.h`)
- Defines quantization parameters
- Stores scales, zero-points, etc.
- Configures quantization behavior

### 5. Utilities

**Memory** (`include/embee/memory.h`)
- Memory management utilities
- Aligned allocation
- Memory pools for efficient reuse

**Platform** (`include/embee/platform.h`)
- Platform-specific optimizations
- CPU feature detection
- Threading utilities

## Data Flow

The typical data flow for inference is:

1. **Loading**:
   - Load model weights and configuration from AMB/GGUF/ONNX file
   - Load tokenizer data

2. **Tokenization**:
   - Convert input text to token IDs
   - Handle special tokens (BOS, EOS, etc.)

3. **Inference**:
   - Process input tokens through transformer layers
   - Use KV cache for efficient autoregressive generation
   - Apply optimized attention and feed-forward operations

4. **Generation**:
   - Sample from output logits using temperature/top-p
   - Generate new tokens one by one
   - Stream results as they're generated

5. **Output**:
   - Convert generated token IDs back to text
   - Return or stream the final output

## Memory Management

embee.cpp uses several strategies to minimize memory usage:

1. **Weight Sharing**: Share tensors between identical layers
2. **Quantization**: Store weights in lower precision (INT4, INT5)
3. **Memory Mapping**: Load model weights on-demand
4. **KV Cache Management**: Efficient storage for attention cache
5. **Tensor Reuse**: Reuse activation buffers during inference

## Performance Optimizations

The engine incorporates multiple optimizations:

1. **SIMD Instructions**: SSE, AVX, AVX2, AVX-512, NEON
2. **Block-wise Computation**: Process data in cache-friendly blocks
3. **Multi-threading**: Parallelize computation where beneficial
4. **Kernel Fusion**: Combine operations to reduce memory traffic
5. **Quantized Compute**: Perform calculations in lower precision
6. **Optimized Matrix Multiplication**: Fast GEMM implementations

## Extension Points

embee.cpp provides several extension points:

1. **Model Architectures**: Add support for new architectures
2. **Tokenizers**: Implement custom tokenization methods
3. **Operators**: Add specialized layer implementations
4. **Quantization Methods**: Implement custom quantization schemes
5. **File Formats**: Support additional model formats

## Supported Model Architectures

The engine currently targets these architectures:

1. **LLaMA/LLaMA-2**: Meta's open-source LLM
2. **Mistral/Mixtral**: Mixture-of-experts models
3. **Phi/Phi-2/Phi-3**: Microsoft's efficient models
4. **Gemma**: Google's lightweight models
5. **Falcon**: Technology Innovation Institute's models

## File Formats

The engine supports multiple file formats:

1. **AMB**: Custom format optimized for embee.cpp
2. **GGUF**: Compatible with llama.cpp ecosystem
3. **ONNX**: Standard interchange format for ML models