/**
 * @file model.h
 * @brief Model representation and loading for embee.cpp
 */

#pragma once

#include "types.h"
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>

namespace embee {

/**
 * @struct ModelConfig
 * @brief Configuration parameters for a transformer model
 */
struct ModelConfig {
    // Basic model parameters
    size_t n_vocab;           // Vocabulary size
    size_t n_embd;            // Embedding dimension
    size_t n_layers;          // Number of layers
    size_t n_heads;           // Number of attention heads
    size_t n_kv_heads;        // Number of KV heads (for GQA/MQA)
    size_t max_seq_len;       // Maximum sequence length
    bool is_rope;             // Uses rotary position embeddings
    
    // Architecture-specific parameters
    ModelArchitecture architecture;
    ActivationFunction activation_function;
    float rope_freq_base;     // Base frequency for RoPE (usually 10000.0)
    float rope_scaling;       // Scaling factor for RoPE (for extended context)
    
    // Quantization parameters
    QuantizationType quant_type;
    
    // Optional model metadata
    std::string model_name;
    std::string model_family;
    std::string model_creator;
};

/**
 * @class Model
 * @brief Represents a transformer model with weights and configuration
 */
class Model {
public:
    /**
     * Load a model from a file
     * @param path Path to the model file
     */
    explicit Model(const std::string& path);
    
    /**
     * Get the model configuration
     * @return The model configuration
     */
    const ModelConfig& config() const { return config_; }
    
    /**
     * Get a tensor by name
     * @param name Name of the tensor to retrieve
     * @return Reference to the tensor
     * @throws std::out_of_range if tensor not found
     */
    const Tensor& get_tensor(const std::string& name) const;
    
    /**
     * Check if a tensor exists
     * @param name Name of the tensor to check
     * @return true if tensor exists, false otherwise
     */
    bool has_tensor(const std::string& name) const;
    
    /**
     * Get the tokenizer for this model
     * @return Pointer to the tokenizer
     */
    std::shared_ptr<Tokenizer> tokenizer() const { return tokenizer_; }
    
private:
    ModelConfig config_;
    std::unordered_map<std::string, Tensor> weights_;
    std::shared_ptr<Tokenizer> tokenizer_;
    
    // Model loading helpers
    void load_amb_model(const std::string& path);
    void load_gguf_model(const std::string& path);
    void load_onnx_model(const std::string& path);
    
    // File format detection
    static std::string detect_format(const std::string& path);
};

} // namespace embee