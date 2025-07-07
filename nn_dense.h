//
// Dense (fully connected) layer implementation
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NN_DENSE_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NN_DENSE_H

#include "nn_interfaces.h"
#include <functional>
#include <random>

namespace utec::nn {

template<typename T>
class DenseLayer : public ILayer<T> {
private:
    size_t input_size_;
    size_t output_size_;
    Tensor<T, 2> W_;  // Weight matrix [input_size, output_size]
    Tensor<T, 2> b_;  // Bias vector [1, output_size]
    Tensor<T, 2> dW_; // Weight gradients
    Tensor<T, 2> db_; // Bias gradients
    Tensor<T, 2> last_x_; // Store last input for backward pass

public:
    DenseLayer(size_t input_size, size_t output_size) 
        : input_size_(input_size), output_size_(output_size),
          W_(input_size, output_size),
          b_(1, output_size),
          dW_(input_size, output_size),
          db_(1, output_size) {
        
        // He normal initialization for weights
        std::random_device rd;
        std::mt19937 gen(rd());
        T std_dev = std::sqrt(T(2.0) / input_size);
        std::normal_distribution<T> dist(T(0.0), std_dev);
        
        for (auto& w : W_) {
            w = dist(gen);
        }
        
        // Initialize bias to zero
        b_.fill(T(0.0));
    }

    Tensor<T, 2> forward(const Tensor<T, 2>& input) override {
        // Store input for backward pass
        last_x_ = input;
        
        // Compute z = x * W + b
        Tensor<T,2> z = matrix_product(input, W_);
        
        // Add bias (broadcasting)
        for (size_t i = 0; i < z.shape()[0]; ++i) {
            for (size_t j = 0; j < z.shape()[1]; ++j) {
                z(i, j) += b_(0, j);
            }
        }
        
        return z;
    }

    Tensor<T, 2> backward(const Tensor<T, 2>& dZ) override {
        // Compute gradients for weights and biases
        // dW = X^T * dZ
        Tensor<T,2> dX = matrix_product(dZ, transpose_2d(W_));
        
        // Compute parameter gradients
        dW_ = matrix_product(transpose_2d(last_x_), dZ);
        
        // Compute bias gradients (sum over batch dimension)
        db_.fill(T(0.0));
        for (size_t i = 0; i < dZ.shape()[0]; ++i) {
            for (size_t j = 0; j < dZ.shape()[1]; ++j) {
                db_(0, j) += dZ(i, j);
            }
        }
        
        return dX;
    }

    std::vector<std::reference_wrapper<Tensor<T, 2>>> get_parameters() override {
        return {std::ref(W_), std::ref(b_)};
    }

    std::vector<std::reference_wrapper<Tensor<T, 2>>> get_gradients() override {
        return {std::ref(dW_), std::ref(db_)};
    }

    std::unique_ptr<ILayer<T>> clone() const override {
        auto layer = std::make_unique<DenseLayer<T>>(input_size_, output_size_);
        layer->W_ = W_;
        layer->b_ = b_;
        return layer;
    }

    // Getters for inspection
    const Tensor<T, 2>& weights() const { return W_; }
    const Tensor<T, 2>& biases() const { return b_; }
    size_t input_size() const { return input_size_; }
    size_t output_size() const { return output_size_; }
};

} // namespace utec::nn

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_NN_DENSE_H 