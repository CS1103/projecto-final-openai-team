//
// Tensor implementation for neural networks
// Supports multidimensional arrays with broadcasting and matrix operations
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H

#include <array>
#include <vector>
#include <stdexcept>
#include <numeric>
#include <initializer_list>
#include <iterator>
#include <algorithm>
#include <sstream>

namespace utec::algebra {

template<typename T, size_t N>
class Tensor {
private:
    std::array<size_t, N> shape_;
    std::vector<T> data_;

    size_t calculate_size() const {
        return std::accumulate(shape_.begin(), shape_.end(), 1UL, std::multiplies<size_t>());
    }

    size_t calculate_index(const std::array<size_t, N>& indices) const {
        size_t index = 0;
        size_t multiplier = 1;
        for (int i = N - 1; i >= 0; --i) {
            index += indices[i] * multiplier;
            multiplier *= shape_[i];
        }
        return index;
    }

    template<typename... Args>
    size_t calculate_index(Args... indices) const {
        static_assert(sizeof...(indices) == N, "Number of indices must match tensor dimension");
        std::array<size_t, N> idx_array = {static_cast<size_t>(indices)...};
        return calculate_index(idx_array);
    }

    std::array<size_t, N> broadcast_shape(const Tensor& other) const {
        std::array<size_t, N> result_shape = shape_;
        for (size_t i = 0; i < N; ++i) {
            if (shape_[i] != other.shape_[i]) {
                if (shape_[i] == 1) {
                    result_shape[i] = other.shape_[i];
                } else if (other.shape_[i] != 1) {
                    throw std::runtime_error("Incompatible shapes for broadcasting");
                }
            }
        }
        return result_shape;
    }

    bool can_broadcast(const Tensor& other) const {
        for (size_t i = 0; i < N; ++i) {
            if (shape_[i] != other.shape_[i] && shape_[i] != 1 && other.shape_[i] != 1) {
                return false;
            }
        }
        return true;
    }

    size_t broadcast_index(size_t flat_index, const std::array<size_t, N>& broadcast_shape, const std::array<size_t, N>& original_shape) const {
        std::array<size_t, N> indices;
        size_t temp = flat_index;
        
        for (int i = N - 1; i >= 0; --i) {
            indices[i] = temp % broadcast_shape[i];
            temp /= broadcast_shape[i];
        }
        
        for (size_t i = 0; i < N; ++i) {
            if (original_shape[i] == 1 && broadcast_shape[i] > 1) {
                indices[i] = 0;
            }
        }
        
        size_t result_index = 0;
        size_t multiplier = 1;
        for (int i = N - 1; i >= 0; --i) {
            result_index += indices[i] * multiplier;
            multiplier *= original_shape[i];
        }
        return result_index;
    }

public:
    template<typename... Args>
    explicit Tensor(Args... dimensions) : shape_({static_cast<size_t>(dimensions)...}) {
        static_assert(sizeof...(dimensions) == N, "Number of dimensions must match template parameter");
        for (size_t dim : shape_) {
            if (dim == 0) {
                std::ostringstream oss;
                oss << "Invalid dimension: all dimensions must be positive, got dimension size: " << dim;
                throw std::runtime_error(oss.str());
            }
        }
        data_.resize(calculate_size());
    }

    size_t size() const {
        return data_.size();
    }

    const std::array<size_t, N>& shape() const {
        return shape_;
    }

    explicit Tensor(const std::array<size_t, N>& shape) : shape_(shape) {
        data_.resize(calculate_size());
    }

    Tensor() {
        shape_.fill(0);
    }

    void fill(const T& value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    template<typename... Args>
    T& operator()(Args... indices) {
        return data_[calculate_index(indices...)];
    }

    template<typename... Args>
    const T& operator()(Args... indices) const {
        return data_[calculate_index(indices...)];
    }

    T& operator[](size_t index) {
        return data_[index];
    }

    const T& operator[](size_t index) const {
        return data_[index];
    }

    Tensor(std::initializer_list<T> init_list) {
        if constexpr (N == 1) {
            shape_[0] = init_list.size();
            data_ = std::vector<T>(init_list);
        } else {
            throw std::runtime_error("Initializer list constructor only supported for 1D tensors");
        }
    }

    Tensor(std::initializer_list<std::initializer_list<T>> init_list) {
        if constexpr (N == 2) {
            shape_[0] = init_list.size();
            shape_[1] = init_list.size() > 0 ? init_list.begin()->size() : 0;
            
            data_.reserve(calculate_size());
            for (const auto& row : init_list) {
                if (row.size() != shape_[1]) {
                    throw std::runtime_error("All rows must have the same size");
                }
                data_.insert(data_.end(), row.begin(), row.end());
            }
        } else {
            throw std::runtime_error("2D initializer list constructor only supported for 2D tensors");
        }
    }

    Tensor operator+(const Tensor& other) const {
        if (!can_broadcast(other)) {
            throw std::runtime_error("Cannot broadcast tensors for addition");
        }

        if (shape_ == other.shape_) {
            Tensor result = *this;
            for (size_t i = 0; i < data_.size(); ++i) {
                result.data_[i] += other.data_[i];
            }
            return result;
        } else {
            auto broadcast_shape_arr = broadcast_shape(other);
            Tensor result(broadcast_shape_arr);
            
            for (size_t i = 0; i < result.data_.size(); ++i) {
                size_t this_idx = broadcast_index(i, broadcast_shape_arr, shape_);
                size_t other_idx = other.broadcast_index(i, broadcast_shape_arr, other.shape_);
                result.data_[i] = data_[this_idx] + other.data_[other_idx];
            }
            return result;
        }
    }

    Tensor operator-(const Tensor& other) const {
        if (!can_broadcast(other)) {
            throw std::runtime_error("Cannot broadcast tensors for subtraction");
        }

        if (shape_ == other.shape_) {
            Tensor result = *this;
            for (size_t i = 0; i < data_.size(); ++i) {
                result.data_[i] -= other.data_[i];
            }
            return result;
        } else {
            auto broadcast_shape_arr = broadcast_shape(other);
            Tensor result(broadcast_shape_arr);
            
            for (size_t i = 0; i < result.data_.size(); ++i) {
                size_t this_idx = broadcast_index(i, broadcast_shape_arr, shape_);
                size_t other_idx = other.broadcast_index(i, broadcast_shape_arr, other.shape_);
                result.data_[i] = data_[this_idx] - other.data_[other_idx];
            }
            return result;
        }
    }

    Tensor operator*(const Tensor& other) const {
        if (!can_broadcast(other)) {
            throw std::runtime_error("Cannot broadcast tensors for multiplication");
        }

        if (shape_ == other.shape_) {
            Tensor result = *this;
            for (size_t i = 0; i < data_.size(); ++i) {
                result.data_[i] *= other.data_[i];
            }
            return result;
        } else {
            auto broadcast_shape_arr = broadcast_shape(other);
            Tensor result(broadcast_shape_arr);
            
            for (size_t i = 0; i < result.data_.size(); ++i) {
                size_t this_idx = broadcast_index(i, broadcast_shape_arr, shape_);
                size_t other_idx = other.broadcast_index(i, broadcast_shape_arr, other.shape_);
                result.data_[i] = data_[this_idx] * other.data_[other_idx];
            }
            return result;
        }
    }

    Tensor operator+(const T& scalar) const {
        Tensor result = *this;
        for (auto& val : result.data_) {
            val += scalar;
        }
        return result;
    }

    Tensor operator-(const T& scalar) const {
        Tensor result = *this;
        for (auto& val : result.data_) {
            val -= scalar;
        }
        return result;
    }

    Tensor operator*(const T& scalar) const {
        Tensor result = *this;
        for (auto& val : result.data_) {
            val *= scalar;
        }
        return result;
    }

    Tensor operator/(const T& scalar) const {
        Tensor result = *this;
        for (auto& val : result.data_) {
            val /= scalar;
        }
        return result;
    }

    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto cbegin() const { return data_.cbegin(); }
    auto cend() const { return data_.cend(); }

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        if constexpr (N == 1) {
            os << "";
            for (size_t i = 0; i < tensor.shape_[0]; ++i) {
                if (i > 0) os << " ";
                os << tensor.data_[i];
            }
            os << "";
        } else if constexpr (N == 2) {
            os << "{\n";
            for (size_t i = 0; i < tensor.shape_[0]; ++i) {
                for (size_t j = 0; j < tensor.shape_[1]; ++j) {
                    if (j > 0) os << " ";
                    os << tensor(i, j);
                }
                if (i < tensor.shape_[0] - 1) os << "\n";
            }
            os << "\n}";
        } else if constexpr (N == 3) {
            os << "{\n";
            for (size_t i = 0; i < tensor.shape_[0]; ++i) {
                os << "{\n";
                for (size_t j = 0; j < tensor.shape_[1]; ++j) {
                    for (size_t k = 0; k < tensor.shape_[2]; ++k) {
                        if (k > 0) os << " ";
                        os << tensor(i, j, k);
                    }
                    if (j < tensor.shape_[1] - 1) os << "\n";
                }
                os << "\n}";
                if (i < tensor.shape_[0] - 1) os << "\n";
            }
            os << "\n}";
        } else if constexpr (N == 4) {
            os << "{\n";
            for (size_t i = 0; i < tensor.shape_[0]; ++i) {
                os << "{\n";
                for (size_t j = 0; j < tensor.shape_[1]; ++j) {
                    os << "{\n";
                    for (size_t k = 0; k < tensor.shape_[2]; ++k) {
                        for (size_t l = 0; l < tensor.shape_[3]; ++l) {
                            if (l > 0) os << " ";
                            os << tensor(i, j, k, l);
                        }
                        if (k < tensor.shape_[2] - 1) os << "\n";
                    }
                    os << "\n}";
                    if (j < tensor.shape_[1] - 1) os << "\n";
                }
                os << "\n}";
                if (i < tensor.shape_[0] - 1) os << "\n";
            }
            os << "\n}";
        } else {
            // Generic case for higher dimensions - fall back to flat format
            os << "{";
            for (size_t i = 0; i < tensor.data_.size(); ++i) {
                if (i > 0) os << " ";
                os << tensor.data_[i];
            }
            os << "}";
        }
        return os;
    }

    template<typename U, size_t M>
    friend auto transpose_2d(const Tensor<U, M>& tensor) -> Tensor<U, M>;

    template<typename U, size_t M>
    friend auto matrix_product(const Tensor<U, M>& a, const Tensor<U, M>& b) -> Tensor<U, M>;
};

template<typename T, size_t N>
Tensor<T, N> operator+(const T& scalar, const Tensor<T, N>& tensor) {
    return tensor + scalar;
}

template<typename T, size_t N>
Tensor<T, N> operator*(const T& scalar, const Tensor<T, N>& tensor) {
    return tensor * scalar;
}

template<typename T, size_t N>
auto transpose_2d(const Tensor<T, N>& tensor) -> Tensor<T, N> {
    if constexpr (N < 2) {
        throw std::runtime_error("Cannot transpose 1D tensor: need at least 2 dimensions");
    } else {
        auto shape = tensor.shape();
        std::swap(shape[N-2], shape[N-1]);

        Tensor<T, N> result(shape);

        if constexpr (N == 2) {
            for (size_t i = 0; i < tensor.shape()[0]; ++i) {
                for (size_t j = 0; j < tensor.shape()[1]; ++j) {
                    result(j, i) = tensor(i, j);
                }
            }
        } else if constexpr (N == 3) {
            for (size_t i = 0; i < tensor.shape()[0]; ++i) {
                for (size_t j = 0; j < tensor.shape()[1]; ++j) {
                    for (size_t k = 0; k < tensor.shape()[2]; ++k) {
                        result(i, k, j) = tensor(i, j, k);
                    }
                }
            }
        } else if constexpr (N == 4) {
            for (size_t i = 0; i < tensor.shape()[0]; ++i) {
                for (size_t j = 0; j < tensor.shape()[1]; ++j) {
                    for (size_t k = 0; k < tensor.shape()[2]; ++k) {
                        for (size_t l = 0; l < tensor.shape()[3]; ++l) {
                            result(i, j, l, k) = tensor(i, j, k, l);
                        }
                    }
                }
            }
        }

        return result;
    }
}

// Matrix multiplication function
template<typename T, size_t N>
auto matrix_product(const Tensor<T, N>& a, const Tensor<T, N>& b) -> Tensor<T, N> {
    if constexpr (N < 2) {
        throw std::runtime_error("Matrix multiplication requires at least 2D tensors");
    } else {
        auto shape_a = a.shape();
        auto shape_b = b.shape();

        if (shape_a[N-1] != shape_b[N-2]) {
            throw std::runtime_error("Matrix dimensions are incompatible for multiplication");
        }

        for (size_t i = 0; i < N-2; ++i) {
            if (shape_a[i] != shape_b[i]) {
                throw std::runtime_error("Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");
            }
        }

        auto result_shape = shape_a;
        result_shape[N-1] = shape_b[N-1];

        Tensor<T, N> result(result_shape);

        if constexpr (N == 2) {
            size_t rows = shape_a[0];
            size_t cols = shape_b[1];
            size_t inner = shape_a[1];

            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    T sum = T{};
                    for (size_t k = 0; k < inner; ++k) {
                        sum += a(i, k) * b(k, j);
                    }
                    result(i, j) = sum;
                }
            }
        } else if constexpr (N == 3) {
            size_t batch = shape_a[0];
            size_t rows = shape_a[1];
            size_t cols = shape_b[2];
            size_t inner = shape_a[2];

            for (size_t b_idx = 0; b_idx < batch; ++b_idx) {
                for (size_t i = 0; i < rows; ++i) {
                    for (size_t j = 0; j < cols; ++j) {
                        T sum = T{};
                        for (size_t k = 0; k < inner; ++k) {
                            sum += a(b_idx, i, k) * b(b_idx, k, j);
                        }
                        result(b_idx, i, j) = sum;
                    }
                }
            }
        }

        return result;
    }
}

} // namespace utec::algebra

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H 