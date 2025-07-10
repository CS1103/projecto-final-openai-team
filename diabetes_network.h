//
// Diabetes-specific neural network configurations
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DIABETES_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DIABETES_NETWORK_H

#include "neural_network.h"
#include <memory>
#include <vector>
#include <iostream>

namespace utec::diabetes {

using namespace utec::nn;

// Configuration structure for diabetes prediction models
struct DiabetesNetworkConfig {
    std::vector<size_t> layer_sizes;
    float learning_rate;
    size_t epochs;
    size_t batch_size;
    bool use_adam;
    std::string name;
    
    void print_config() const {
        std::cout << "\n🏗️ CONFIGURACIÓN DE RED: " << name << std::endl;
        std::cout << "📐 Arquitectura: ";
        for (size_t i = 0; i < layer_sizes.size(); ++i) {
            std::cout << layer_sizes[i];
            if (i < layer_sizes.size() - 1) std::cout << " → ";
        }
        std::cout << std::endl;
        std::cout << "⚡ Learning rate: " << learning_rate << std::endl;
        std::cout << "🔄 Épocas: " << epochs << std::endl;
        std::cout << "📦 Batch size: " << batch_size << std::endl;
        std::cout << "🧠 Optimizador: " << (use_adam ? "Adam" : "SGD") << std::endl;
    }
};

// Predefined network configurations for diabetes prediction

// Simple network for quick testing
DiabetesNetworkConfig get_simple_config() {
    return {
        .layer_sizes = {13, 32, 1},  // Asumiendo ~13 características después del preprocessing
        .learning_rate = 0.01f,
        .epochs = 500,
        .batch_size = 32,
        .use_adam = false,
        .name = "Simple Diabetes Classifier"
    };
}

// Standard configuration - balanced performance and training time
DiabetesNetworkConfig get_standard_config() {
    return {
        .layer_sizes = {13, 64, 32, 16, 1},
        .learning_rate = 0.001f,
        .epochs = 1500,
        .batch_size = 64,
        .use_adam = true,
        .name = "Standard Diabetes Classifier"
    };
}

// Complex configuration for maximum accuracy
DiabetesNetworkConfig get_complex_config() {
    return {
        .layer_sizes = {13, 128, 64, 32, 16, 8, 1},
        .learning_rate = 0.0005f,
        .epochs = 3000,
        .batch_size = 128,
        .use_adam = true,
        .name = "Complex Diabetes Classifier"
    };
}

// Optimized configuration based on diabetes domain knowledge
DiabetesNetworkConfig get_optimized_config() {
    return {
        .layer_sizes = {13, 64, 32, 16, 1},
        .learning_rate = 0.001f,
        .epochs = 2000,
        .batch_size = 128,
        .use_adam = true,
        .name = "Optimized Diabetes Classifier"
    };
}

// Fast training configuration - 500 epochs
DiabetesNetworkConfig get_fast_500_config() {
    return {
        .layer_sizes = {13, 32, 1},
        .learning_rate = 0.01f,
        .epochs = 500,
        .batch_size = 64,
        .use_adam = false,
        .name = "Fast 500 Epochs Classifier"
    };
}

// Medium training configuration - 1000 epochs
DiabetesNetworkConfig get_medium_1000_config() {
    return {
        .layer_sizes = {13, 32, 1},
        .learning_rate = 0.01f,
        .epochs = 1000,
        .batch_size = 64,
        .use_adam = false,
        .name = "Medium 1000 Epochs Classifier"
    };
}

// Factory functions to create diabetes classifiers

template<typename T>
std::unique_ptr<NeuralNetwork<T>> create_diabetes_classifier(const DiabetesNetworkConfig& config) {
    std::cout << "🏭 Creando red neuronal para predicción de diabetes..." << std::endl;
    config.print_config();
    
    // Validate configuration
    if (config.layer_sizes.size() < 2) {
        throw std::runtime_error("❌ La configuración debe tener al menos 2 capas (entrada y salida)");
    }
    
    if (config.layer_sizes.back() != 1) {
        throw std::runtime_error("❌ La capa de salida debe tener exactamente 1 neurona para clasificación binaria");
    }
    
    // Create the network using the base factory function
    auto network = build_binary_classifier<T>(
        config.layer_sizes,
        static_cast<T>(config.learning_rate),
        config.use_adam
    );
    
    std::cout << "✅ Red neuronal creada exitosamente con " << config.layer_sizes.size() << " capas" << std::endl;
    
    return network;
}

// Convenience factory functions

template<typename T = float>
std::unique_ptr<NeuralNetwork<T>> create_simple_diabetes_classifier() {
    return create_diabetes_classifier<T>(get_simple_config());
}

template<typename T = float>
std::unique_ptr<NeuralNetwork<T>> create_standard_diabetes_classifier() {
    return create_diabetes_classifier<T>(get_standard_config());
}

template<typename T = float>
std::unique_ptr<NeuralNetwork<T>> create_complex_diabetes_classifier() {
    return create_diabetes_classifier<T>(get_complex_config());
}

template<typename T = float>
std::unique_ptr<NeuralNetwork<T>> create_optimized_diabetes_classifier() {
    return create_diabetes_classifier<T>(get_optimized_config());
}

template<typename T = float>
std::unique_ptr<NeuralNetwork<T>> create_fast_500_diabetes_classifier() {
    return create_diabetes_classifier<T>(get_fast_500_config());
}

template<typename T = float>
std::unique_ptr<NeuralNetwork<T>> create_medium_1000_diabetes_classifier() {
    return create_diabetes_classifier<T>(get_medium_1000_config());
}

// Default factory - returns the optimized configuration
template<typename T = float>
std::unique_ptr<NeuralNetwork<T>> create_diabetes_classifier() {
    return create_optimized_diabetes_classifier<T>();
}

// Auto-adjust configuration for specific dataset
DiabetesNetworkConfig adjust_config_for_features(DiabetesNetworkConfig config, size_t num_features) {
    if (config.layer_sizes[0] != num_features) {
        std::cout << "🔧 Ajustando capa de entrada de " << config.layer_sizes[0] 
                  << " a " << num_features << " características" << std::endl;
        config.layer_sizes[0] = num_features;
    }
    return config;
}

} // namespace utec::diabetes

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_DIABETES_NETWORK_H 