/**
 * @file types.h
 * @brief Common types and constants used throughout the embee.cpp library
 */

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

namespace embee {

/**
 * Supported model architectures
 */
enum class ModelArchitecture {
    LLAMA,      // LLaMA, LLaMA-2, etc.
    MISTRAL,    // Mistral, Mixtral
    GEMMA,      // Google Gemma
    PHI,        // Microsoft Phi-2, Phi-3
    FALCON,     // Falcon
    GPT2,       // OpenAI GPT-2
    MPT,        // MosaicML MPT
    CUSTOM      // Custom architecture
};

/**
 * Quantization types supported by embee
 */
enum class QuantizationType {
    NONE,       // No quantization (FP32/FP16)
    INT8,       // 8-bit integer quantization
    INT4,       // 4-bit integer quantization
    INT5,       // 5-bit integer quantization
    INT4_BLOCK, // 4-bit block-wise quantization
    INT5_BLOCK, // 5-bit block-wise quantization
    ADAPTIVE    // Adaptive quantization
};

/**
 * Activation functions
 */
enum class ActivationFunction {
    GELU,
    SILU,
    RELU,
    SWIGLU
};

/**
 * Data type for model weights and computations
 */
enum class DataType {
    FP32,   // 32-bit floating point
    FP16,   // 16-bit floating point
    BF16,   // 16-bit brain floating point
    INT8,   // 8-bit integer
    INT4,   // 4-bit integer
    INT5    // 5-bit integer
};

/**
 * Token type (for tokenizer)
 */
using TokenId = int32_t;
using TokenVector = std::vector<TokenId>;

/**
 * Storage for tensor data with associated metadata
 */
class Tensor {
public:
    // Dimensions
    std::vector<size_t> shape;
    
    // Data type
    DataType data_type;
    
    // Raw tensor data
    std::vector<uint8_t> data;
    
    // Tensor name (for debugging and model analysis)
    std::string name;
    
    // Methods for accessing data will be added
};

} // namespace embee