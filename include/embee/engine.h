/**
 * @file engine.h
 * @brief Inference engine for transformer models
 */

#pragma once

#include "model.h"
#include "types.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace embee {

/**
 * @struct GenerationConfig
 * @brief Configuration for text generation
 */
struct GenerationConfig {
    size_t max_length = 512;              // Maximum number of tokens to generate
    float temperature = 0.8f;             // Sampling temperature (1.0 = no change, 0.0 = greedy)
    float top_p = 0.9f;                   // Nucleus sampling probability threshold
    float repetition_penalty = 1.1f;      // Penalty for repeating tokens
    size_t batch_size = 1;               // Batch size for processing
    bool use_cache = true;               // Whether to use KV cache
};

/**
 * @class Engine
 * @brief Main inference engine for transformer models
 */
class Engine {
public:
    /**
     * Callback type for receiving generated tokens
     * @param token_id The ID of the generated token
     * @param text The text of the generated token
     * @return true to continue generation, false to stop
     */
    using TokenCallback = std::function<bool(TokenId token_id, const std::string& text)>;
    
    /**
     * Create an inference engine for a model
     * @param model The model to use for inference
     */
    explicit Engine(const Model& model);
    
    /**
     * Generate text from a prompt
     * @param prompt The input prompt
     * @param config Generation configuration
     * @return Generated text
     */
    std::string generate(const std::string& prompt, const GenerationConfig& config = {});
    
    /**
     * Generate text with streaming callback
     * @param prompt The input prompt
     * @param callback Callback function called for each generated token
     * @param config Generation configuration
     */
    void generate_with_callback(const std::string& prompt, TokenCallback callback, 
                               const GenerationConfig& config = {});
    
    /**
     * Get the raw logits for a prompt (last token)
     * @param prompt The input prompt
     * @return Vector of logits for the vocabulary
     */
    std::vector<float> get_logits(const std::string& prompt);
    
private:
    // Forward declaration of implementation
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace embee