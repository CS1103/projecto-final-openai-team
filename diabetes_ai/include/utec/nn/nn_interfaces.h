//
// Neural network interfaces for layers, optimizers, and loss functions
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NN_INTERFACES_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NN_INTERFACES_H

#include "../algebra/tensor.h"
#include <functional>
#include <memory>

namespace utec::nn {

using namespace utec::algebra;

// Interface for neural network layers
template<typename T>
class ILayer {
public:
    virtual ~ILayer() = default;
    
    // Forward pass - takes input and returns output
    virtual Tensor<T, 2> forward(const Tensor<T, 2>& input) = 0;
    
    // Backward pass - takes gradient from next layer and returns gradient for previous layer
    virtual Tensor<T, 2> backward(const Tensor<T, 2>& gradient) = 0;
    
    // Get trainable parameters (weights and biases)
    virtual std::vector<std::reference_wrapper<Tensor<T, 2>>> get_parameters() = 0;
    
    // Get parameter gradients
    virtual std::vector<std::reference_wrapper<Tensor<T, 2>>> get_gradients() = 0;
    
    // Clone the layer
    virtual std::unique_ptr<ILayer<T>> clone() const = 0;
};

// Interface for optimizers
template<typename T>
class IOptimizer {
public:
    virtual ~IOptimizer() = default;
    
    // Update parameters using computed gradients
    virtual void update(const std::vector<std::reference_wrapper<Tensor<T, 2>>>& parameters,
                       const std::vector<std::reference_wrapper<Tensor<T, 2>>>& gradients) = 0;
    
    // Reset optimizer state (for momentum, etc.)
    virtual void reset() = 0;
    
    // Clone the optimizer
    virtual std::unique_ptr<IOptimizer<T>> clone() const = 0;
};

// Interface for loss functions
template<typename T>
class ILoss {
public:
    virtual ~ILoss() = default;
    
    // Compute loss value
    virtual T compute_loss(const Tensor<T, 2>& predictions, const Tensor<T, 2>& targets) = 0;
    
    // Compute gradient of loss with respect to predictions
    virtual Tensor<T, 2> compute_gradient(const Tensor<T, 2>& predictions, const Tensor<T, 2>& targets) = 0;
    
    // Clone the loss function
    virtual std::unique_ptr<ILoss<T>> clone() const = 0;
};

} // namespace utec::nn

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_NN_INTERFACES_H 