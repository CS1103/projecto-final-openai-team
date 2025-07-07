//
// Optimizer implementations
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NN_OPTIMIZER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NN_OPTIMIZER_H

#include "nn_interfaces.h"
#include <cmath>

namespace utec::nn {

// Stochastic Gradient Descent optimizer
template<typename T>
class SGDOptimizer : public IOptimizer<T> {
private:
    T learning_rate_;

public:
    explicit SGDOptimizer(T learning_rate) : learning_rate_(learning_rate) {}

    void update(const std::vector<std::reference_wrapper<Tensor<T, 2>>>& parameters,
                const std::vector<std::reference_wrapper<Tensor<T, 2>>>& gradients) override {
        
        if (parameters.size() != gradients.size()) {
            throw std::runtime_error("Parameters and gradients must have the same size");
        }

        for (size_t i = 0; i < parameters.size(); ++i) {
            auto& param = parameters[i].get();
            const auto& grad = gradients[i].get();
            
            if (param.shape() != grad.shape()) {
                throw std::runtime_error("Parameter and gradient shapes must match");
            }
            
            for (size_t j = 0; j < param.size(); ++j) {
                param[j] -= learning_rate_ * grad[j];
            }
        }
    }

    void reset() override {
        // SGD has no state to reset
    }

    std::unique_ptr<IOptimizer<T>> clone() const override {
        return std::make_unique<SGDOptimizer<T>>(learning_rate_);
    }

    T learning_rate() const { return learning_rate_; }
    void set_learning_rate(T lr) { learning_rate_ = lr; }
};

// Adam optimizer
template<typename T>
class AdamOptimizer : public IOptimizer<T> {
private:
    T learning_rate_;
    T beta1_;
    T beta2_;
    T epsilon_;
    size_t t_; // Time step counter
    std::vector<Tensor<T, 2>> m_; // First moment estimates
    std::vector<Tensor<T, 2>> v_; // Second moment estimates

public:
    explicit AdamOptimizer(T learning_rate = T(0.001), T beta1 = T(0.9), T beta2 = T(0.999), T epsilon = T(1e-8))
        : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0) {}

    void update(const std::vector<std::reference_wrapper<Tensor<T, 2>>>& parameters,
                const std::vector<std::reference_wrapper<Tensor<T, 2>>>& gradients) override {
        
        if (parameters.size() != gradients.size()) {
            throw std::runtime_error("Parameters and gradients must have the same size");
        }

        // Initialize moment estimates if this is the first update
        if (m_.empty()) {
            m_.reserve(parameters.size());
            v_.reserve(parameters.size());
            
            for (const auto& param : parameters) {
                m_.emplace_back(param.get().shape());
                v_.emplace_back(param.get().shape());
                m_.back().fill(T(0.0));
                v_.back().fill(T(0.0));
            }
        }

        ++t_;

        for (size_t i = 0; i < parameters.size(); ++i) {
            auto& param = parameters[i].get();
            const auto& grad = gradients[i].get();
            auto& m_i = m_[i];
            auto& v_i = v_[i];
            
            if (param.shape() != grad.shape()) {
                throw std::runtime_error("Parameter and gradient shapes must match");
            }
            
            for (size_t j = 0; j < param.size(); ++j) {
                // Update biased first moment estimate
                m_i[j] = beta1_ * m_i[j] + (T(1.0) - beta1_) * grad[j];
                
                // Update biased second raw moment estimate
                v_i[j] = beta2_ * v_i[j] + (T(1.0) - beta2_) * grad[j] * grad[j];
                
                // Compute bias-corrected first moment estimate
                T m_hat = m_i[j] / (T(1.0) - std::pow(beta1_, t_));
                
                // Compute bias-corrected second raw moment estimate
                T v_hat = v_i[j] / (T(1.0) - std::pow(beta2_, t_));
                
                // Update parameters
                param[j] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
            }
        }
    }

    void reset() override {
        t_ = 0;
        for (auto& m : m_) {
            m.fill(T(0.0));
        }
        for (auto& v : v_) {
            v.fill(T(0.0));
        }
    }

    std::unique_ptr<IOptimizer<T>> clone() const override {
        return std::make_unique<AdamOptimizer<T>>(learning_rate_, beta1_, beta2_, epsilon_);
    }

    // Getters
    T learning_rate() const { return learning_rate_; }
    T beta1() const { return beta1_; }
    T beta2() const { return beta2_; }
    T epsilon() const { return epsilon_; }
    size_t time_step() const { return t_; }
    
    // Setters
    void set_learning_rate(T lr) { learning_rate_ = lr; }
};

} // namespace utec::nn

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_NN_OPTIMIZER_H 