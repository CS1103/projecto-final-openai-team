//
// Main program for diabetes prediction using neural networks
//

#include "data_loader.h"
#include "diabetes_network.h"
#include "model_evaluation.h"
#include "nn_loss.h"
#include "nn_optimizer.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <exception>

using namespace std::chrono;
using namespace utec::data;
using namespace utec::diabetes;
using namespace utec::evaluation;
using namespace utec::nn;
using namespace utec::algebra;

void print_header() {
    std::cout << "████████████████████████████████████████████████████████████████" << std::endl;
    std::cout << "██           PREDICCIÓN DE DIABETES CON REDES NEURONALES     ██" << std::endl;
    std::cout << "██                    Universidad UTEC                       ██" << std::endl;
    std::cout << "██              Proyecto Final - Programación III            ██" << std::endl;
    std::cout << "████████████████████████████████████████████████████████████████" << std::endl;
    std::cout << std::endl;
}

void print_phase(const std::string& phase_name, int phase_number) {
    std::cout << "\n🔄 FASE " << phase_number << ": " << phase_name << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
}

Tensor<float, 2> create_mini_batch(const Tensor<float, 2>& data, size_t start_idx, size_t batch_size) {
    size_t actual_batch_size = std::min(batch_size, data.shape()[0] - start_idx);
    
    Tensor<float, 2> batch(actual_batch_size, data.shape()[1]);
    
    for (size_t i = 0; i < actual_batch_size; ++i) {
        for (size_t j = 0; j < data.shape()[1]; ++j) {
            batch(i, j) = data(start_idx + i, j);
        }
    }
    
    return batch;
}

void train_network(std::unique_ptr<NeuralNetwork<float>>& network, 
                  const DataSplit& data,
                  const DiabetesNetworkConfig& config) {
    
    print_phase("ENTRENAMIENTO DE LA RED NEURONAL", 3);
    
    std::cout << "🚀 Iniciando entrenamiento..." << std::endl;
    std::cout << "📊 Datos de entrenamiento: " << data.train_samples << " muestras" << std::endl;
    std::cout << "⚡ Configuración: " << config.epochs << " épocas, batch size: " << config.batch_size << std::endl;
    
    auto start_time = high_resolution_clock::now();
    
    // Training loop
    for (size_t epoch = 0; epoch < config.epochs; ++epoch) {
        float total_loss = 0.0f;
        size_t num_batches = 0;
        
        // Process data in mini-batches
        for (size_t i = 0; i < data.train_samples; i += config.batch_size) {
            size_t actual_batch_size = std::min(config.batch_size, data.train_samples - i);
            
            // Create mini-batches
            Tensor<float, 2> batch_x = create_mini_batch(data.X_train, i, actual_batch_size);
            Tensor<float, 2> batch_y = create_mini_batch(data.y_train, i, actual_batch_size);
            
            // Train on this batch
            float batch_loss = network->train_step(batch_x, batch_y);
            total_loss += batch_loss;
            num_batches++;
        }
        
        // Print progress
        if ((epoch + 1) % 200 == 0 || epoch == 0) {
            float avg_loss = total_loss / num_batches;
            std::cout << "Época " << std::setw(4) << (epoch + 1) << "/" << config.epochs 
                      << " | Loss: " << std::fixed << std::setprecision(6) << avg_loss 
                      << " | Progreso: " << std::setprecision(1) << (100.0f * (epoch + 1) / config.epochs) << "%" << std::endl;
        }
    }
    
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(end_time - start_time);
    
    std::cout << "✅ Entrenamiento completado en " << duration.count() << " segundos" << std::endl;
}

void evaluate_model(const std::unique_ptr<NeuralNetwork<float>>& network,
                   const DataSplit& data) {
    
    print_phase("EVALUACIÓN DEL MODELO", 4);
    
    std::cout << "🧪 Evaluando modelo en conjunto de prueba..." << std::endl;
    
    auto start_time = high_resolution_clock::now();
    
    // Make predictions on test set
    Tensor<float, 2> predictions = network->forward(data.X_test);
    
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);
    
    std::cout << "⚡ Predicciones generadas en " << duration.count() << " ms" << std::endl;
    std::cout << "📈 Promedio de predicciones: " << duration.count() / static_cast<float>(data.test_samples) 
              << " ms por muestra" << std::endl;
    
    // Evaluate using different thresholds
    ModelEvaluator evaluator;
    
    // Standard evaluation (threshold = 0.5)
    auto metrics = evaluator.evaluate(predictions, data.y_test);
    
    std::cout << "\n🏆 RESULTADOS PRINCIPALES:" << std::endl;
    std::cout << "📊 Accuracy: " << std::fixed << std::setprecision(2) << (metrics.accuracy * 100) << "%" << std::endl;
    std::cout << "🎯 Sensibilidad: " << std::fixed << std::setprecision(2) << (metrics.recall * 100) << "%" << std::endl;
    std::cout << "⚡ Especificidad: " << std::fixed << std::setprecision(2) << (metrics.specificity * 100) << "%" << std::endl;
    std::cout << "⭐ F1-Score: " << std::fixed << std::setprecision(2) << (metrics.f1_score * 100) << "%" << std::endl;
    std::cout << "🏅 Calificación: " << metrics.get_performance_grade() << std::endl;
    
    // Detailed evaluation
    metrics.print_detailed();
    
    // Threshold analysis
    evaluator.print_threshold_analysis(predictions, data.y_test);
}

void demonstrate_predictions(const std::unique_ptr<NeuralNetwork<float>>& network,
                           const DiabetesDataLoader& loader) {
    
    print_phase("DEMOSTRACIÓN DE PREDICCIONES", 5);
    
    std::cout << "🧑‍⚕️ Casos clínicos de ejemplo:" << std::endl;
    
    // Case 1: High risk patient
    std::cout << "\n👨 Caso 1: Paciente de alto riesgo" << std::endl;
    std::cout << "Datos: Hombre, 65 años, hipertensión, sin enfermedad cardíaca, ex-fumador, BMI 28.5, HbA1c 6.2, glucosa 140" << std::endl;
    
    auto sample1 = loader.preprocess_sample("Male", 65, 1, 0, "former", 28.5f, 6.2f, 140.0f);
    auto pred1 = network->forward(sample1);
    float risk1 = pred1(0, 0) * 100;
    
    std::cout << "🎯 Predicción: " << std::fixed << std::setprecision(1) << risk1 << "% de probabilidad de diabetes" << std::endl;
    std::cout << "📋 Recomendación: " << (risk1 > 50 ? "❗ Evaluación médica urgente" : "✅ Monitoreo regular") << std::endl;
    
    // Case 2: Low risk patient
    std::cout << "\n👩 Caso 2: Paciente de bajo riesgo" << std::endl;
    std::cout << "Datos: Mujer, 28 años, sin hipertensión, sin enfermedad cardíaca, nunca fumó, BMI 22.0, HbA1c 5.1, glucosa 85" << std::endl;
    
    auto sample2 = loader.preprocess_sample("Female", 28, 0, 0, "never", 22.0f, 5.1f, 85.0f);
    auto pred2 = network->forward(sample2);
    float risk2 = pred2(0, 0) * 100;
    
    std::cout << "🎯 Predicción: " << std::fixed << std::setprecision(1) << risk2 << "% de probabilidad de diabetes" << std::endl;
    std::cout << "📋 Recomendación: " << (risk2 > 50 ? "❗ Evaluación médica urgente" : "✅ Riesgo bajo, seguimiento rutinario") << std::endl;
    
    // Case 3: Medium risk patient
    std::cout << "\n👨 Caso 3: Paciente de riesgo medio" << std::endl;
    std::cout << "Datos: Hombre, 45 años, sin hipertensión, sin enfermedad cardíaca, fumador actual, BMI 26.0, HbA1c 5.8, glucosa 110" << std::endl;
    
    auto sample3 = loader.preprocess_sample("Male", 45, 0, 0, "current", 26.0f, 5.8f, 110.0f);
    auto pred3 = network->forward(sample3);
    float risk3 = pred3(0, 0) * 100;
    
    std::cout << "🎯 Predicción: " << std::fixed << std::setprecision(1) << risk3 << "% de probabilidad de diabetes" << std::endl;
    std::cout << "📋 Recomendación: " << (risk3 > 50 ? "⚠️ Evaluación médica recomendada" : "⚠️ Monitoreo frecuente") << std::endl;
}

void compare_models(const DataSplit& data) {
    print_phase("COMPARACIÓN DE MODELOS", 6);
    
    std::cout << "🆚 Comparando diferentes arquitecturas de red neuronal..." << std::endl;
    
    std::vector<std::pair<std::string, DiabetesNetworkConfig>> configs = {
        {"Simple", get_simple_config()},
        {"Standard", get_standard_config()},
        {"Optimized", get_optimized_config()}
    };
    
    std::cout << "\n📊 RESULTADOS DE COMPARACIÓN:" << std::endl;
    std::cout << std::setw(12) << "Modelo" << std::setw(12) << "Accuracy" << std::setw(12) << "F1-Score" 
              << std::setw(15) << "Sensibilidad" << std::setw(15) << "Especificidad" << std::setw(12) << "Tiempo" << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    
    for (const auto& [name, config] : configs) {
        auto start_time = high_resolution_clock::now();
        
        // Adjust config for actual number of features
        auto adjusted_config = adjust_config_for_features(config, data.num_features);
        auto network = create_diabetes_classifier<float>(adjusted_config);
        
        // Quick training (reduced epochs for comparison)
        auto quick_config = adjusted_config;
        quick_config.epochs = 300;
        
        train_network(network, data, quick_config);
        
        // Evaluate
        auto predictions = network->forward(data.X_test);
        ModelEvaluator evaluator;
        auto metrics = evaluator.evaluate(predictions, data.y_test);
        
        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<seconds>(end_time - start_time);
        
        std::cout << std::setw(12) << name 
                  << std::setw(11) << std::fixed << std::setprecision(1) << (metrics.accuracy * 100) << "%"
                  << std::setw(11) << std::fixed << std::setprecision(1) << (metrics.f1_score * 100) << "%"
                  << std::setw(14) << std::fixed << std::setprecision(1) << (metrics.recall * 100) << "%"
                  << std::setw(14) << std::fixed << std::setprecision(1) << (metrics.specificity * 100) << "%"
                  << std::setw(10) << duration.count() << "s" << std::endl;
    }
}

int main() {
    try {
        print_header();
        
        // Phase 1: Load and preprocess data
        print_phase("CARGA Y PREPROCESAMIENTO DE DATOS", 1);
        
        std::cout << "📁 Cargando dataset de diabetes..." << std::endl;
        DiabetesDataLoader loader("diabetes_prediction_dataset.csv");
        
        loader.print_dataset_summary();
        loader.print_feature_info();
        
        auto data = loader.split_data(0.2f, 42);
        
        // Phase 2: Network configuration
        print_phase("CONFIGURACIÓN DE LA RED NEURONAL", 2);
        
        auto config = get_optimized_config();
        config = adjust_config_for_features(config, data.num_features);
        
        auto network = create_diabetes_classifier<float>(config);
        
        // Phase 3: Training
        train_network(network, data, config);
        
        // Phase 4: Evaluation
        evaluate_model(network, data);
        
        // Phase 5: Demonstration
        demonstrate_predictions(network, loader);
        
        // Phase 6: Model comparison
        // compare_models(data);  // Comentado para ahorrar tiempo
        
        std::cout << "\n🎉 ¡ANÁLISIS COMPLETADO EXITOSAMENTE!" << std::endl;
        std::cout << "📝 El modelo está listo para ser usado en predicción de diabetes." << std::endl;
        std::cout << "⚠️ IMPORTANTE: Este modelo es solo para fines educativos." << std::endl;
        std::cout << "🏥 Para uso clínico real, se requiere validación médica adicional." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ ERROR: " << e.what() << std::endl;
        std::cerr << "🔧 Verifique que el archivo 'diabetes_prediction_dataset.csv' esté en el directorio actual." << std::endl;
        return 1;
    }
    
    return 0;
} 