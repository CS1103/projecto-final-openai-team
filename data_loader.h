//
// Data loader and preprocessor for diabetes prediction dataset
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DATA_LOADER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DATA_LOADER_H

#include "tensor.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <iostream>
#include <set>
#include <numeric>
#include <cmath>

namespace utec::data {

using namespace utec::algebra;

struct DataSplit {
    Tensor<float, 2> X_train;
    Tensor<float, 2> y_train;
    Tensor<float, 2> X_test;
    Tensor<float, 2> y_test;
    
    size_t train_samples;
    size_t test_samples;
    size_t num_features;
    
    void print_info() const {
        std::cout << "📊 División de datos:" << std::endl;
        std::cout << "   🏋️ Entrenamiento: " << train_samples << " muestras" << std::endl;
        std::cout << "   🧪 Prueba: " << test_samples << " muestras" << std::endl;
        std::cout << "   📏 Características: " << num_features << std::endl;
    }
};

class DiabetesDataLoader {
private:
    std::string filename_;
    std::vector<std::vector<std::string>> raw_data_;
    std::vector<std::string> headers_;
    
    // Preprocessing statistics
    std::unordered_map<std::string, float> feature_means_;
    std::unordered_map<std::string, float> feature_stds_;
    std::unordered_map<std::string, std::vector<std::string>> categorical_mappings_;
    
    // Feature indices after preprocessing
    size_t total_features_;
    
public:
    explicit DiabetesDataLoader(const std::string& filename) : filename_(filename) {
        load_csv();
        analyze_data();
    }

private:
    void load_csv() {
        std::ifstream file(filename_);
        if (!file.is_open()) {
            throw std::runtime_error("❌ No se pudo abrir el archivo: " + filename_);
        }
        
        std::string line;
        bool first_line = true;
        
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            std::vector<std::string> row;
            std::stringstream ss(line);
            std::string cell;
            
            while (std::getline(ss, cell, ',')) {
                // Remover espacios en blanco
                cell.erase(0, cell.find_first_not_of(" \t\r\n"));
                cell.erase(cell.find_last_not_of(" \t\r\n") + 1);
                row.push_back(cell);
            }
            
            if (first_line) {
                headers_ = row;
                first_line = false;
                std::cout << "🔍 Headers detectados: ";
                for (const auto& h : headers_) {
                    std::cout << h << " ";
                }
                std::cout << std::endl;
            } else {
                if (row.size() == headers_.size()) {
                    raw_data_.push_back(row);
                }
            }
        }
        
        std::cout << "📁 Cargados " << raw_data_.size() << " registros del archivo " << filename_ << std::endl;
    }
    
    void analyze_data() {
        if (raw_data_.empty()) {
            throw std::runtime_error("❌ No hay datos para analizar");
        }
        
        // Analizar smoking_history para one-hot encoding
        std::set<std::string> smoking_categories;
        for (const auto& row : raw_data_) {
            if (headers_.size() > 4) { // smoking_history está en índice 4
                smoking_categories.insert(row[4]);
            }
        }
        
        categorical_mappings_["smoking_history"] = std::vector<std::string>(smoking_categories.begin(), smoking_categories.end());
        
        std::cout << "🚬 Categorías de smoking_history detectadas: ";
        for (const auto& cat : categorical_mappings_["smoking_history"]) {
            std::cout << "'" << cat << "' ";
        }
        std::cout << std::endl;
        
        // Calcular estadísticas para normalización de variables numéricas
        calculate_normalization_stats();
        
        // Calcular total de características después del preprocessing
        calculate_total_features();
    }
    
    void calculate_normalization_stats() {
        // Índices de las variables numéricas: age(1), bmi(5), HbA1c_level(6), blood_glucose_level(7)
        std::vector<std::pair<std::string, int>> numeric_features = {
            {"age", 1}, {"bmi", 5}, {"HbA1c_level", 6}, {"blood_glucose_level", 7}
        };
        
        for (const auto& [feature_name, idx] : numeric_features) {
            std::vector<float> values;
            
            for (const auto& row : raw_data_) {
                if (idx < static_cast<int>(row.size())) {
                    try {
                        float value = std::stof(row[idx]);
                        values.push_back(value);
                    } catch (const std::exception&) {
                        std::cout << "⚠️ Valor inválido en " << feature_name << ": " << row[idx] << std::endl;
                    }
                }
            }
            
            if (!values.empty()) {
                float mean = std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
                float variance = 0.0f;
                for (float val : values) {
                    variance += (val - mean) * (val - mean);
                }
                variance /= values.size();
                float std_dev = std::sqrt(variance);
                
                feature_means_[feature_name] = mean;
                feature_stds_[feature_name] = std_dev > 1e-8 ? std_dev : 1.0f; // Evitar división por cero
                
                std::cout << "📊 " << feature_name << ": μ=" << mean << ", σ=" << std_dev << std::endl;
            }
        }
    }
    
    void calculate_total_features() {
        // gender (1) + age (1) + hypertension (1) + heart_disease (1) + 
        // smoking_history (one-hot) + bmi (1) + HbA1c_level (1) + blood_glucose_level (1)
        total_features_ = 6 + categorical_mappings_["smoking_history"].size();
        std::cout << "🔢 Total de características después del preprocessing: " << total_features_ << std::endl;
    }
    
    std::vector<float> preprocess_row(const std::vector<std::string>& row) const {
        std::vector<float> processed;
        processed.reserve(total_features_);
        
        // 1. gender: Female=0, Male=1
        if (row[0] == "Male") {
            processed.push_back(1.0f);
        } else {
            processed.push_back(0.0f);
        }
        
        // 2. age (normalizado)
        float age = std::stof(row[1]);
        processed.push_back((age - feature_means_.at("age")) / feature_stds_.at("age"));
        
        // 3. hypertension (ya es 0/1)
        processed.push_back(std::stof(row[2]));
        
        // 4. heart_disease (ya es 0/1)
        processed.push_back(std::stof(row[3]));
        
        // 5. smoking_history (one-hot encoding)
        const std::string& smoking = row[4];
        for (const std::string& category : categorical_mappings_.at("smoking_history")) {
            processed.push_back(smoking == category ? 1.0f : 0.0f);
        }
        
        // 6. bmi (normalizado)
        float bmi = std::stof(row[5]);
        processed.push_back((bmi - feature_means_.at("bmi")) / feature_stds_.at("bmi"));
        
        // 7. HbA1c_level (normalizado)
        float hba1c = std::stof(row[6]);
        processed.push_back((hba1c - feature_means_.at("HbA1c_level")) / feature_stds_.at("HbA1c_level"));
        
        // 8. blood_glucose_level (normalizado)
        float glucose = std::stof(row[7]);
        processed.push_back((glucose - feature_means_.at("blood_glucose_level")) / feature_stds_.at("blood_glucose_level"));
        
        return processed;
    }

public:
    DataSplit split_data(float test_ratio = 0.2f, int random_seed = 42) {
        if (raw_data_.empty()) {
            throw std::runtime_error("❌ No hay datos para dividir");
        }
        
        std::cout << "\n🔄 Preprocesando y dividiendo datos..." << std::endl;
        
        // Preparar datos procesados
        std::vector<std::vector<float>> X_processed;
        std::vector<float> y_processed;
        
        for (const auto& row : raw_data_) {
            try {
                std::vector<float> x_row = preprocess_row(row);
                float y_val = std::stof(row[8]); // diabetes column
                
                X_processed.push_back(x_row);
                y_processed.push_back(y_val);
            } catch (const std::exception& e) {
                std::cout << "⚠️ Error procesando fila, omitiendo: " << e.what() << std::endl;
            }
        }
        
        // Mezclar datos
        std::vector<size_t> indices(X_processed.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        std::mt19937 rng(random_seed);
        std::shuffle(indices.begin(), indices.end(), rng);
        
        // Dividir
        size_t test_size = static_cast<size_t>(X_processed.size() * test_ratio);
        size_t train_size = X_processed.size() - test_size;
        
        DataSplit split;
        split.train_samples = train_size;
        split.test_samples = test_size;
        split.num_features = total_features_;
        
        // Crear tensores
        split.X_train = Tensor<float, 2>(train_size, total_features_);
        split.y_train = Tensor<float, 2>(train_size, 1);
        split.X_test = Tensor<float, 2>(test_size, total_features_);
        split.y_test = Tensor<float, 2>(test_size, 1);
        
        // Llenar tensores de entrenamiento
        for (size_t i = 0; i < train_size; ++i) {
            size_t idx = indices[i];
            for (size_t j = 0; j < total_features_; ++j) {
                split.X_train(i, j) = X_processed[idx][j];
            }
            split.y_train(i, 0) = y_processed[idx];
        }
        
        // Llenar tensores de prueba
        for (size_t i = 0; i < test_size; ++i) {
            size_t idx = indices[train_size + i];
            for (size_t j = 0; j < total_features_; ++j) {
                split.X_test(i, j) = X_processed[idx][j];
            }
            split.y_test(i, 0) = y_processed[idx];
        }
        
        std::cout << "✅ Datos preprocesados y divididos exitosamente" << std::endl;
        split.print_info();
        
        return split;
    }
    
    // Función para procesar una nueva muestra para predicción
    Tensor<float, 2> preprocess_sample(const std::string& gender, float age, int hypertension, 
                                      int heart_disease, const std::string& smoking_history, 
                                      float bmi, float hba1c, float blood_glucose) const {
        
        std::vector<std::string> row = {
            gender, std::to_string(age), std::to_string(hypertension), 
            std::to_string(heart_disease), smoking_history, std::to_string(bmi),
            std::to_string(hba1c), std::to_string(blood_glucose)
        };
        
        std::vector<float> processed = preprocess_row(row);
        
        Tensor<float, 2> sample(1, total_features_);
        for (size_t i = 0; i < total_features_; ++i) {
            sample(0, i) = processed[i];
        }
        
        return sample;
    }
    
    // Getters para información
    size_t num_samples() const { return raw_data_.size(); }
    size_t num_features_after_preprocessing() const { return total_features_; }
    const std::vector<std::string>& get_smoking_categories() const { 
        return categorical_mappings_.at("smoking_history"); 
    }
    
    void print_feature_info() const {
        std::cout << "\n📊 INFORMACIÓN DE CARACTERÍSTICAS:" << std::endl;
        std::cout << "1. gender (binario): Female=0, Male=1" << std::endl;
        std::cout << "2. age (normalizado): μ=" << feature_means_.at("age") << ", σ=" << feature_stds_.at("age") << std::endl;
        std::cout << "3. hypertension (binario): 0/1" << std::endl;
        std::cout << "4. heart_disease (binario): 0/1" << std::endl;
        std::cout << "5. smoking_history (one-hot): ";
        for (const auto& cat : categorical_mappings_.at("smoking_history")) {
            std::cout << "'" << cat << "' ";
        }
        std::cout << std::endl;
        std::cout << "6. bmi (normalizado): μ=" << feature_means_.at("bmi") << ", σ=" << feature_stds_.at("bmi") << std::endl;
        std::cout << "7. HbA1c_level (normalizado): μ=" << feature_means_.at("HbA1c_level") << ", σ=" << feature_stds_.at("HbA1c_level") << std::endl;
        std::cout << "8. blood_glucose_level (normalizado): μ=" << feature_means_.at("blood_glucose_level") << ", σ=" << feature_stds_.at("blood_glucose_level") << std::endl;
        std::cout << "Total: " << total_features_ << " características\n" << std::endl;
    }
    
    void print_dataset_summary() const {
        if (raw_data_.empty()) {
            std::cout << "❌ No hay datos cargados." << std::endl;
            return;
        }
        
        std::cout << "\n📋 RESUMEN DEL DATASET:" << std::endl;
        std::cout << "📊 Total de muestras: " << raw_data_.size() << std::endl;
        
        // Contar casos positivos y negativos
        int positive_cases = 0;
        for (const auto& row : raw_data_) {
            if (row.size() > 8) {
                try {
                    if (std::stof(row[8]) == 1.0f) {
                        positive_cases++;
                    }
                } catch (const std::exception&) {
                    // Ignorar filas con datos inválidos
                }
            }
        }
        
        int negative_cases = raw_data_.size() - positive_cases;
        float positive_ratio = 100.0f * positive_cases / raw_data_.size();
        float negative_ratio = 100.0f * negative_cases / raw_data_.size();
        
        std::cout << "🍭 Casos con diabetes: " << positive_cases << " (" << positive_ratio << "%)" << std::endl;
        std::cout << "💚 Casos sin diabetes: " << negative_cases << " (" << negative_ratio << "%)" << std::endl;
        std::cout << "⚖️ Balance del dataset: " << (positive_ratio > 40 && positive_ratio < 60 ? "Balanceado" : "Desbalanceado") << std::endl;
    }
};

} // namespace utec::data

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_DATA_LOADER_H 