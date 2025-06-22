# embee.cpp

<div align="center">
  <img src="docs/logo/embee_logo.png" alt="embee.cpp Logo" width="250">
  <h3>A modular, lightweight C++ inference engine for LLMs</h3>
  <p>Run transformer-based models anywhere - from high-end servers to Raspberry Pi</p>
</div>

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/std/the-standard)
[![Build Status](https://img.shields.io/github/workflow/status/Muneeb50702/embee.cpp/CI)](https://github.com/Muneeb50702/embee.cpp/actions)

## 🐝 About embee.cpp

**embee.cpp** is an ultra-lightweight inference engine for transformer-based Large Language Models (LLMs), designed to be a more flexible and general-purpose alternative to llama.cpp. Our focus is on creating a universal runtime that can run any transformer architecture with minimal memory footprint and maximum efficiency on CPU-only systems.

### Key Features

- **Universal Model Support**: Run any transformer-based LLM model (Mistral, Phi-3, Gemma, LLaMA, etc.)
- **Efficient Quantization**: Advanced 4-bit and 5-bit static and adaptive quantization methods
- **Custom .amb Format**: Optimized format for fast loading and minimal memory usage
- **CPU-Optimized**: Designed to run efficiently on x86, ARM, Raspberry Pi, and mobile chips
- **Modular Architecture**: Easily extensible with plug-ins
- **Developer-Friendly**: Clean API, well-documented code, comprehensive tests

## 🚀 Getting Started

### Prerequisites

- C++17 compatible compiler (GCC 9+, Clang 10+, MSVC 2019+)
- CMake 3.14+
- (Optional) Python 3.8+ for model conversion scripts

### Build Instructions

```bash
git clone https://github.com/Muneeb50702/embee.git
cd embee.cpp
mkdir build && cd build
cmake ..
make -j
```

### Quick Usage Example

```cpp
#include "embee/engine.h"
#include "embee/tokenizer.h"
#include "embee/model.h"

int main() {
    // Load a quantized model
    embee::Model model("models/phi3-mini-4b.amb");
    
    // Create inference engine
    embee::Engine engine(model);
    
    // Run inference
    std::string prompt = "What is the meaning of life?";
    std::string response = engine.generate(prompt, 512);
    
    std::cout << response << std::endl;
    return 0;
}
```

## 📚 Documentation

For comprehensive documentation, including architecture details, API reference and advanced usage scenarios, please visit our [Documentation](docs/README.md).

- [Architecture Overview](docs/architecture.md)
- [Model Format Specification](docs/model_format.md)
- [Quantization Guide](docs/quantization.md)
- [API Reference](docs/api_reference.md)
- [Performance Benchmarks](docs/benchmarks.md)

## 🛠️ Components

The embee.cpp project consists of the following main components:

- **Core Inference Engine**: High-performance C++ implementation of transformer inference
- **Tokenizer**: Fast BPE/SentencePiece tokenization implementation
- **Model Loader**: For loading .amb, GGUF, and ONNX model formats
- **Quantizer**: Tools for model compression and optimization
- **REST API** (optional): For deploying models as services
- **GUI** (optional): Simple Qt-based interface for model interaction
- **Benchmarking Tools**: For performance testing and optimization

## 🤝 Contributing

We welcome contributions of all kinds! See our [Contributing Guide](CONTRIBUTING.md) for more information.

## 📊 Performance Benchmarks

| Model       | Size | Quantization | Tokens/sec (CPU) | Memory Usage |
|-------------|------|--------------|------------------|--------------|
| Phi-3-mini  | 4B   | 4-bit        | ~20              | ~2.5 GB      |
| Mistral-7B  | 7B   | 4-bit        | ~12              | ~4.5 GB      |
| Gemma       | 2B   | 5-bit        | ~35              | ~1.3 GB      |

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ✨ Acknowledgements

- Inspired by the incredible work of [llama.cpp](https://github.com/ggerganov/llama.cpp)
- Thanks to the open source LLM community for their research and models