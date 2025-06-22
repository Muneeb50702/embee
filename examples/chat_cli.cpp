/**
 * @file chat_cli.cpp
 * @brief Simple CLI chatbot example using embee.cpp
 */

#include "embee/engine.h"
#include "embee/model.h"
#include "embee/tokenizer.h"

#include <iostream>
#include <string>
#include <sstream>
#include <chrono>
#include <thread>

// ANSI color codes for prettier output
namespace Color {
    const std::string Reset = "\033[0m";
    const std::string Bold = "\033[1m";
    const std::string Red = "\033[31m";
    const std::string Green = "\033[32m";
    const std::string Yellow = "\033[33m";
    const std::string Blue = "\033[34m";
    const std::string Magenta = "\033[35m";
    const std::string Cyan = "\033[36m";
}

int main(int argc, char** argv) {
    // Parse command line arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path> [temperature] [top_p]" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    float temperature = argc > 2 ? std::stof(argv[2]) : 0.7f;
    float top_p = argc > 3 ? std::stof(argv[3]) : 0.9f;
    
    std::cout << Color::Bold << Color::Cyan 
              << "Loading model from: " << model_path << Color::Reset << std::endl;
    
    try {
        // Load the model
        embee::Model model(model_path);
        
        const auto& config = model.config();
        std::cout << "Model: " << config.model_name << " (" 
                  << config.n_layers << " layers, " 
                  << config.n_heads << " heads, "
                  << config.n_embd << " embedding size)" << std::endl;
        
        // Create the inference engine
        embee::Engine engine(model);
        
        // Set up generation config
        embee::GenerationConfig gen_config;
        gen_config.temperature = temperature;
        gen_config.top_p = top_p;
        gen_config.max_length = 1024;
        
        // Chat loop
        std::string user_input;
        
        // System prompt
        std::string conversation = "You are an AI assistant. You are helpful, harmless, and honest.\n\n";
        
        std::cout << Color::Bold << Color::Green 
                  << "Chat with the model. Type 'exit' to quit." << Color::Reset << std::endl;
        
        while (true) {
            // Get user input
            std::cout << Color::Bold << Color::Blue << "\nUser: " << Color::Reset;
            std::getline(std::cin, user_input);
            
            if (user_input == "exit") {
                break;
            }
            
            // Update conversation
            conversation += "User: " + user_input + "\n\nAssistant: ";
            
            // Generate response with streaming
            std::cout << Color::Bold << Color::Yellow << "Assistant: " << Color::Reset;
            std::cout.flush();
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Stream tokens
            engine.generate_with_callback(conversation, 
                [&conversation](embee::TokenId token_id, const std::string& text) {
                    std::cout << text << std::flush;
                    conversation += text;
                    return true;
                }, 
                gen_config
            );
            
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end_time - start_time;
            
            // Add newline after response and update conversation
            conversation += "\n\n";
            
            // Print stats
            std::cout << std::endl;
            std::cout << Color::Magenta << "[Generated in " 
                      << elapsed.count() << " seconds]" << Color::Reset << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << Color::Red << "Error: " << e.what() << Color::Reset << std::endl;
        return 1;
    }
    
    return 0;
}