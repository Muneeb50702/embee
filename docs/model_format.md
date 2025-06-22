# embee.cpp Model Format Specification (.amb)

## Overview

The AMB (Adaptive Model Binary) format is a custom binary format designed for embee.cpp to efficiently store transformer model weights and configuration in a highly optimized way. The format prioritizes:

1. Fast loading speed
2. Minimal memory usage
3. Support for various quantization methods
4. Extensibility for future model architectures

## File Format Structure

An AMB file consists of a header followed by a series of sections:

```
+----------------+
| AMB Header     |
+----------------+
| Metadata       |
+----------------+
| Config         |
+----------------+
| Tokenizer      |
+----------------+
| Weights        |
+----------------+
```

### AMB Header (28 bytes)

| Offset | Size | Description                                    |
|--------|------|------------------------------------------------|
| 0      | 5    | Magic identifier: "AMBEE"                      |
| 5      | 1    | File format version (current: 1)               |
| 6      | 2    | Flags (reserved for future use)                |
| 8      | 4    | Metadata section size                          |
| 12     | 4    | Config section size                            |
| 16     | 4    | Tokenizer section size                         |
| 20     | 8    | Weights section size                           |

### Metadata Section

Contains general information about the model in JSON format:

```json
{
  "name": "phi-3-mini-4bit",
  "family": "Phi",
  "creator": "Microsoft",
  "description": "Quantized version of Microsoft Phi-3 Mini",
  "license": "MIT",
  "created": "2024-06-22",
  "version": "1.0",
  "tags": ["conversational", "instruction-following", "coding"]
}
```

### Config Section

Contains model architecture and configuration parameters in JSON format:

```json
{
  "architecture": "phi",
  "n_vocab": 32000,
  "n_embd": 2048,
  "n_layers": 24,
  "n_heads": 16,
  "n_kv_heads": 16,
  "max_seq_len": 2048,
  "is_rope": true,
  "activation_fn": "silu",
  "rope_freq_base": 10000.0,
  "rope_scaling": 1.0,
  "quant": {
    "type": "int4_block",
    "block_size": 128,
    "scale_type": "fp16"
  }
}
```

### Tokenizer Section

Contains the tokenizer data, which includes:

1. Tokenizer type identifier (1 byte): 
   - 0: BPE
   - 1: SentencePiece
   - 2: WordPiece
   - 3: Custom

2. Special tokens (2 bytes each):
   - BOS token ID
   - EOS token ID
   - PAD token ID
   - UNK token ID
   - MASK token ID

3. Vocabulary data (format depends on tokenizer type)
   - For BPE: merges list + vocabulary mapping
   - For SentencePiece: serialized SentencePiece model

### Weights Section

Contains all model weights in a binary format optimized for fast loading. Each weight tensor is stored as:

| Field        | Size (bytes)      | Description                                       |
|-------------|--------------------|---------------------------------------------------|
| Name length | 2                  | Length of the tensor name                         |
| Name        | Name length        | Name of the tensor (e.g., "transformer.wte")      |
| Shape dims  | 1                  | Number of dimensions                              |
| Shape       | 4 * dims           | Size of each dimension (uint32)                   |
| Data type   | 1                  | Type of data (FP32, FP16, INT8, INT4, etc.)       |
| Data size   | 8                  | Size of the data in bytes                         |
| Data        | Data size          | The actual tensor data                            |
| Alignment   | 0-7                | Padding to 8-byte alignment                       |

For quantized weights, additional metadata is stored:

| Field        | Size (bytes)    | Description                                        |
|-------------|------------------|----------------------------------------------------|
| Quant type  | 1                | Quantization type                                  |
| Block size  | 2                | Size of quantization blocks                        |
| Scale type  | 1                | Type of scales (FP32 or FP16)                      |
| Scales      | Varies           | Quantization scales                                |

## Data Types

| Value | Type    | Description                           |
|-------|---------|---------------------------------------|
| 0     | FP32    | 32-bit floating point                 |
| 1     | FP16    | 16-bit floating point                 |
| 2     | BF16    | 16-bit brain floating point           |
| 3     | INT8    | 8-bit integer                         |
| 4     | INT4    | 4-bit integer                         |
| 5     | INT5    | 5-bit integer                         |
| 6     | INT4BLOCK | 4-bit block-wise quantization       |
| 7     | INT5BLOCK | 5-bit block-wise quantization       |
| 8     | ADAPTIVE | Adaptive precision quantization      |

## Quantization Formats

### INT4/INT5 Block Quantization

In block quantization:

1. Weights are divided into blocks (typically 32 or 128 elements)
2. Each block is quantized independently
3. Each block has its own scale factor

Storage format for each block:
```
[scale: fp16][quantized weights: int4/int5]
```

### Adaptive Precision Quantization

Adaptive quantization assigns different precision to different weights based on their importance:

1. The most sensitive weights use higher precision (FP16)
2. Less sensitive weights use lower precision (INT4/INT5)
3. A mask indicates which precision is used for each block

Storage format:
```
[precision mask][scales][quantized weights]
```

## File Validation

To validate an AMB file:
1. Check the magic number "AMBEE"
2. Verify version compatibility
3. Ensure section sizes match the file size
4. Verify checksums (if present in flags)

## Example Python Tool for Creating AMB Files

```python
import numpy as np
import json
import struct

def write_amb_file(output_path, metadata, config, tokenizer_data, weights):
    """
    Write an AMB file.
    
    Args:
        output_path: Path to write the AMB file
        metadata: Dict containing model metadata
        config: Dict containing model configuration
        tokenizer_data: Dict containing tokenizer data
        weights: Dict mapping tensor names to numpy arrays
    """
    # Serialize sections
    metadata_bytes = json.dumps(metadata).encode('utf-8')
    config_bytes = json.dumps(config).encode('utf-8')
    
    # Serialize tokenizer (simplified)
    tokenizer_bytes = b''
    # ... serialize tokenizer data
    
    # Open file for writing
    with open(output_path, 'wb') as f:
        # Write header placeholder
        f.write(b'AMBEE\x01\x00\x00')  # Magic + version + flags
        
        # Reserve space for section sizes
        section_sizes_pos = f.tell()
        f.write(b'\x00' * 20)  # 5 * 4 bytes for section sizes
        
        # Write metadata section
        metadata_pos = f.tell()
        f.write(metadata_bytes)
        
        # Write config section
        config_pos = f.tell()
        f.write(config_bytes)
        
        # Write tokenizer section
        tokenizer_pos = f.tell()
        f.write(tokenizer_bytes)
        
        # Write weights section
        weights_pos = f.tell()
        for name, tensor in weights.items():
            # Write tensor name
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<H', len(name_bytes)))
            f.write(name_bytes)
            
            # Write tensor shape
            f.write(struct.pack('<B', len(tensor.shape)))
            for dim in tensor.shape:
                f.write(struct.pack('<I', dim))
            
            # Write data type
            dtype_code = 0  # FP32
            f.write(struct.pack('<B', dtype_code))
            
            # Write data size
            data_size = tensor.nbytes
            f.write(struct.pack('<Q', data_size))
            
            # Write tensor data
            tensor.tofile(f)
            
            # Align to 8 bytes
            align = (8 - f.tell() % 8) % 8
            f.write(b'\x00' * align)
        
        # Calculate section sizes
        metadata_size = config_pos - metadata_pos
        config_size = tokenizer_pos - config_pos
        tokenizer_size = weights_pos - tokenizer_pos
        weights_size = f.tell() - weights_pos
        
        # Update header with section sizes
        f.seek(section_sizes_pos)
        f.write(struct.pack('<IIIQ', metadata_size, config_size, 
                            tokenizer_size, weights_size))
```

## Future Extensions

The AMB format is designed to be extensible. Future versions may add:

1. Sparse weight storage for pruned models
2. Multiple tokenizer support
3. Support for additional model architectures
4. Model merging metadata
5. Fine-tuning history
6. Weight hashing for integrity verification

## Tools

Tools for working with AMB files are available in the `scripts/` directory:
- `convert_to_amb.py`: Convert models from other formats
- `inspect_amb.py`: Examine the contents of an AMB file
- `quantize_amb.py`: Quantize a full-precision AMB model