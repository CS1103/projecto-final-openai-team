//
// Loss functions implementation
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NN_LOSS_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NN_LOSS_H

#include "nn_interfaces.h"
#include <cmath>
#include <algorithm>

namespace utec::nn {

// Mean Squared Error loss
template<typename T>
class MSELoss : public ILoss<T> {
public:
    T compute_loss(const Tensor<T, 2>& predictions, const Tensor<T, 2>& targets) override {
        if (predictions.shape() != targets.shape()) {
            throw std::runtime_error("Predictions and targets must have the same shape");
        }

        T total_loss = T(0.0);
        size_t num_samples = predictions.shape()[0];
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            T diff = predictions[i] - targets[i];
            total_loss += diff * diff;
        }
        
        return total_loss / (T(2.0) * num_samples);
    }

    Tensor<T, 2> compute_gradient(const Tensor<T, 2>& predictions, const Tensor<T, 2>& targets) override {
        if (predictions.shape() != targets.shape()) {
            throw std::runtime_error("Predictions and targets must have the same shape");
        }

        Tensor<T, 2> gradient = predictions;
        size_t num_samples = predictions.shape()[0];
        
        for (size_t i = 0; i < gradient.size(); ++i) {
            gradient[i] = (predictions[i] - targets[i]) / num_samples;
        }
        
        return gradient;
    }

    std::unique_ptr<ILoss<T>> clone() const override {
        return std::make_unique<MSELoss<T>>();
    }
};

// Binary Cross Entropy loss
template<typename T>
class BinaryCrossEntropyLoss : public ILoss<T> {
private:
    T epsilon_ = T(1e-7); // Aumentado de 1e-15 a 1e-7 para mayor estabilidad

public:
    T compute_loss(const Tensor<T, 2>& predictions, const Tensor<T, 2>& targets) override {
        if (predictions.shape() != targets.shape()) {
            throw std::runtime_error("Predictions and targets must have the same shape");
        }

        T total_loss = T(0.0);
        size_t num_samples = predictions.shape()[0];
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            T pred = std::max(epsilon_, std::min(T(1.0) - epsilon_, predictions[i]));
            T target = targets[i];
            
            total_loss += -(target * std::log(pred) + (T(1.0) - target) * std::log(T(1.0) - pred));
        }
        
        return total_loss / num_samples;
    }

    Tensor<T, 2> compute_gradient(const Tensor<T, 2>& predictions, const Tensor<T, 2>& targets) override {
        if (predictions.shape() != targets.shape()) {
            throw std::runtime_error("Predictions and targets must have the same shape");
        }

        Tensor<T, 2> gradient = predictions;
        size_t num_samples = predictions.shape()[0];
        
        for (size_t i = 0; i < gradient.size(); ++i) {
            T pred = std::max(epsilon_, std::min(T(1.0) - epsilon_, predictions[i]));
            T target = targets[i];
            
            gradient[i] = (pred - target) / (pred * (T(1.0) - pred) * num_samples);
        }
        
        return gradient;
    }

    std::unique_ptr<ILoss<T>> clone() const override {
        return std::make_unique<BinaryCrossEntropyLoss<T>>();
    }
};

} // namespace utec::nn

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_NN_LOSS_H 