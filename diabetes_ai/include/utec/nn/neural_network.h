//
// Main neural network implementation
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H

#include "nn_interfaces.h"
#include "nn_dense.h"
#include "nn_activation.h"
#include "nn_loss.h"
#include "nn_optimizer.h"
#include <memory>
#include <vector>
#include <random>

namespace utec::nn {

template<typename T>
class NeuralNetwork {
private:
    std::vector<std::unique_ptr<ILayer<T>>> layers_;
    std::unique_ptr<ILoss<T>> loss_function_;
    std::unique_ptr<IOptimizer<T>> optimizer_;

public:
    NeuralNetwork() = default;

    // Move constructor and assignment
    NeuralNetwork(NeuralNetwork&& other) noexcept = default;
    NeuralNetwork& operator=(NeuralNetwork&& other) noexcept = default;

    // Copy constructor and assignment (using clone)
    NeuralNetwork(const NeuralNetwork& other) {
        for (const auto& layer : other.layers_) {
            layers_.push_back(layer->clone());
        }
        if (other.loss_function_) {
            loss_function_ = other.loss_function_->clone();
        }
        if (other.optimizer_) {
            optimizer_ = other.optimizer_->clone();
        }
    }

    NeuralNetwork& operator=(const NeuralNetwork& other) {
        if (this != &other) {
            layers_.clear();
            for (const auto& layer : other.layers_) {
                layers_.push_back(layer->clone());
            }
            if (other.loss_function_) {
                loss_function_ = other.loss_function_->clone();
            }
            if (other.optimizer_) {
                optimizer_ = other.optimizer_->clone();
            }
        }
        return *this;
    }

    void add_layer(std::unique_ptr<ILayer<T>> layer) {
        layers_.push_back(std::move(layer));
    }

    void set_loss_function(std::unique_ptr<ILoss<T>> loss_fn) {
        loss_function_ = std::move(loss_fn);
    }

    void set_optimizer(std::unique_ptr<IOptimizer<T>> opt) {
        optimizer_ = std::move(opt);
    }

    Tensor<T, 2> forward(const Tensor<T, 2>& input) {
        Tensor<T, 2> output = input;
        for (auto& layer : layers_) {
            output = layer->forward(output);
        }
        return output;
    }

    T train_step(const Tensor<T, 2>& input, const Tensor<T, 2>& targets) {
        if (!loss_function_ || !optimizer_) {
            throw std::runtime_error("Loss function and optimizer must be set before training");
        }

        // Forward pass
        Tensor<T, 2> predictions = forward(input);

        // Compute loss
        T loss = loss_function_->compute_loss(predictions, targets);

        // Backward pass
        Tensor<T, 2> gradient = loss_function_->compute_gradient(predictions, targets);
        
        for (int i = layers_.size() - 1; i >= 0; --i) {
            gradient = layers_[i]->backward(gradient);
        }

        // Collect parameters and gradients
        std::vector<std::reference_wrapper<Tensor<T, 2>>> parameters;
        std::vector<std::reference_wrapper<Tensor<T, 2>>> gradients;
        
        for (auto& layer : layers_) {
            auto layer_params = layer->get_parameters();
            auto layer_grads = layer->get_gradients();
            
            parameters.insert(parameters.end(), layer_params.begin(), layer_params.end());
            gradients.insert(gradients.end(), layer_grads.begin(), layer_grads.end());
        }

        // Update parameters
        if (!parameters.empty()) {
            optimizer_->update(parameters, gradients);
        }

        return loss;
    }

    size_t num_layers() const {
        return layers_.size();
    }

    // Get layer at index (for inspection)
    const ILayer<T>* get_layer(size_t index) const {
        if (index >= layers_.size()) {
            throw std::out_of_range("Layer index out of range");
        }
        return layers_[index].get();
    }
};

// Factory function to create a binary classifier
template<typename T>
std::unique_ptr<NeuralNetwork<T>> build_binary_classifier(
    const std::vector<size_t>& layer_sizes,
    T learning_rate = T(0.001),
    bool use_adam = true) {
    
    if (layer_sizes.size() < 2) {
        throw std::runtime_error("At least input and output layer sizes must be specified");
    }

    auto network = std::make_unique<NeuralNetwork<T>>();

    // Add hidden layers with ReLU activation
    for (size_t i = 0; i < layer_sizes.size() - 2; ++i) {
        network->add_layer(std::make_unique<DenseLayer<T>>(layer_sizes[i], layer_sizes[i + 1]));
        network->add_layer(std::make_unique<ReLULayer<T>>());
    }

    // Add output layer with sigmoid activation
    network->add_layer(std::make_unique<DenseLayer<T>>(layer_sizes[layer_sizes.size() - 2], layer_sizes.back()));
    network->add_layer(std::make_unique<SigmoidLayer<T>>());

    // Set loss function (binary cross-entropy for binary classification)
    network->set_loss_function(std::make_unique<BinaryCrossEntropyLoss<T>>());

    // Set optimizer
    if (use_adam) {
        network->set_optimizer(std::make_unique<AdamOptimizer<T>>(learning_rate));
    } else {
        network->set_optimizer(std::make_unique<SGDOptimizer<T>>(learning_rate));
    }

    return network;
}

} // namespace utec::nn

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H 