/**
 * @file tokenizer.h
 * @brief Tokenizer interface for embee.cpp
 */

#pragma once

#include "types.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <optional>

namespace embee {

/**
 * @class Tokenizer
 * @brief Abstract base class for all tokenizers (BPE, SentencePiece, etc.)
 */
class Tokenizer {
public:
    virtual ~Tokenizer() = default;
    
    /**
     * Encode text into tokens
     * @param text Input text to tokenize
     * @return Vector of token IDs
     */
    virtual TokenVector encode(const std::string& text) const = 0;
    
    /**
     * Decode tokens back to text
     * @param tokens Vector of token IDs
     * @return Decoded text
     */
    virtual std::string decode(const TokenVector& tokens) const = 0;
    
    /**
     * Get the size of the vocabulary
     * @return Number of tokens in the vocabulary
     */
    virtual size_t vocab_size() const = 0;
    
    /**
     * Get the ID of the token used for beginning of sequence
     * @return BOS token ID or nullopt if not available
     */
    virtual std::optional<TokenId> bos_token() const = 0;
    
    /**
     * Get the ID of the token used for end of sequence
     * @return EOS token ID or nullopt if not available
     */
    virtual std::optional<TokenId> eos_token() const = 0;
    
    /**
     * Get the ID of the token used for padding
     * @return PAD token ID or nullopt if not available
     */
    virtual std::optional<TokenId> pad_token() const = 0;
    
    /**
     * Create a tokenizer from a file
     * @param path Path to the tokenizer file
     * @return Unique pointer to a Tokenizer instance
     */
    static std::unique_ptr<Tokenizer> load(const std::string& path);
};

/**
 * @class BPETokenizer
 * @brief Byte Pair Encoding tokenizer implementation
 */
class BPETokenizer : public Tokenizer {
public:
    BPETokenizer(const std::string& vocab_path, const std::string& merges_path);
    
    TokenVector encode(const std::string& text) const override;
    std::string decode(const TokenVector& tokens) const override;
    size_t vocab_size() const override;
    std::optional<TokenId> bos_token() const override;
    std::optional<TokenId> eos_token() const override;
    std::optional<TokenId> pad_token() const override;
    
private:
    std::unordered_map<std::string, TokenId> token_to_id_;
    std::unordered_map<TokenId, std::string> id_to_token_;
    std::vector<std::pair<std::string, std::string>> merges_;
    
    std::optional<TokenId> bos_token_id_;
    std::optional<TokenId> eos_token_id_;
    std::optional<TokenId> pad_token_id_;
};

/**
 * @class SentencePieceTokenizer
 * @brief SentencePiece tokenizer implementation
 */
class SentencePieceTokenizer : public Tokenizer {
public:
    SentencePieceTokenizer(const std::string& model_path);
    
    TokenVector encode(const std::string& text) const override;
    std::string decode(const TokenVector& tokens) const override;
    size_t vocab_size() const override;
    std::optional<TokenId> bos_token() const override;
    std::optional<TokenId> eos_token() const override;
    std::optional<TokenId> pad_token() const override;
    
private:
    // Implementation details
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace embee