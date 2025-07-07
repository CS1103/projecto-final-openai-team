//
// Activation functions implementation
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NN_ACTIVATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NN_ACTIVATION_H

#include "nn_interfaces.h"
#include <algorithm>
#include <cmath>

namespace utec::nn {

// ReLU activation function
template<typename T>
class ReLULayer : public ILayer<T> {
private:
    Tensor<T, 2> last_input_;

public:
    Tensor<T, 2> forward(const Tensor<T, 2>& input) override {
        last_input_ = input;
        
        Tensor<T, 2> output = input;
        for (auto& val : output) {
            val = std::max(T(0.0), val);
        }
        
        return output;
    }

    Tensor<T, 2> backward(const Tensor<T, 2>& gradient) override {
        Tensor<T, 2> output_gradient = gradient;
        
        for (size_t i = 0; i < output_gradient.size(); ++i) {
            if (last_input_[i] <= T(0.0)) {
                output_gradient[i] = T(0.0);
            }
        }
        
        return output_gradient;
    }

    std::vector<std::reference_wrapper<Tensor<T, 2>>> get_parameters() override {
        return {}; // No trainable parameters
    }

    std::vector<std::reference_wrapper<Tensor<T, 2>>> get_gradients() override {
        return {}; // No gradients to compute
    }

    std::unique_ptr<ILayer<T>> clone() const override {
        return std::make_unique<ReLULayer<T>>();
    }
};

// Sigmoid activation function
template<typename T>
class SigmoidLayer : public ILayer<T> {
private:
    Tensor<T, 2> last_output_;

public:
    Tensor<T, 2> forward(const Tensor<T, 2>& input) override {
        Tensor<T, 2> output = input;
        
        for (auto& val : output) {
            val = T(1.0) / (T(1.0) + std::exp(-val));
        }
        
        last_output_ = output;
        return output;
    }

    Tensor<T, 2> backward(const Tensor<T, 2>& gradient) override {
        Tensor<T, 2> output_gradient = gradient;
        
        for (size_t i = 0; i < output_gradient.size(); ++i) {
            T sigmoid_val = last_output_[i];
            output_gradient[i] *= sigmoid_val * (T(1.0) - sigmoid_val);
        }
        
        return output_gradient;
    }

    std::vector<std::reference_wrapper<Tensor<T, 2>>> get_parameters() override {
        return {}; // No trainable parameters
    }

    std::vector<std::reference_wrapper<Tensor<T, 2>>> get_gradients() override {
        return {}; // No gradients to compute
    }

    std::unique_ptr<ILayer<T>> clone() const override {
        return std::make_unique<SigmoidLayer<T>>();
    }
};

} // namespace utec::nn

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_NN_ACTIVATION_H 