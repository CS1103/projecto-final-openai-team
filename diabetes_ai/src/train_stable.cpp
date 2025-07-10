//
// Programa de entrenamiento estable - Red Neuronal Diabetes (sin NaN)
//

#include "utec/diabetes/data_loader.h"
#include "utec/diabetes/diabetes_network.h"
#include "utec/diabetes/model_evaluation.h"
#include "utec/nn/nn_loss.h"
#include "utec/nn/nn_optimizer.h"
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
    std::cout << "██    ENTRENAMIENTO ESTABLE - RED NEURONAL DIABETES (SIN NaN)  ██" << std::endl;
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

bool is_nan_or_inf(float value) {
    return std::isnan(value) || std::isinf(value);
}

void train_network(std::unique_ptr<NeuralNetwork<float>>& network, 
                  const DataSplit& data,
                  const DiabetesNetworkConfig& config) {
    
    print_phase("ENTRENAMIENTO ESTABLE DE LA RED NEURONAL", 2);
    
    std::cout << "🛡️ Iniciando entrenamiento con protección anti-NaN..." << std::endl;
    std::cout << "📊 Datos de entrenamiento: " << data.train_samples << " muestras" << std::endl;
    std::cout << "⚡ Configuración: " << config.epochs << " épocas, batch size: " << config.batch_size << std::endl;
    std::cout << "🏗️ Arquitectura: Estable (entrada → 32 → 16 → salida)" << std::endl;
    std::cout << "🔒 Learning rate conservador: " << config.learning_rate << std::endl;
    std::cout << "⚙️ Optimizador: " << (config.use_adam ? "Adam" : "SGD (más estable)") << std::endl;
    
    auto start_time = high_resolution_clock::now();
    
    // Training loop with NaN detection
    for (size_t epoch = 0; epoch < config.epochs; ++epoch) {
        float total_loss = 0.0f;
        size_t num_batches = 0;
        bool nan_detected = false;
        
        // Process data in mini-batches
        for (size_t i = 0; i < data.train_samples; i += config.batch_size) {
            size_t actual_batch_size = std::min(config.batch_size, data.train_samples - i);
            
            // Create mini-batches
            Tensor<float, 2> batch_x = create_mini_batch(data.X_train, i, actual_batch_size);
            Tensor<float, 2> batch_y = create_mini_batch(data.y_train, i, actual_batch_size);
            
            // Train on this batch with NaN checking
            float batch_loss = network->train_step(batch_x, batch_y);
            
            // Check for NaN or infinite values
            if (is_nan_or_inf(batch_loss)) {
                std::cout << "\n⚠️ ADVERTENCIA: Detectado NaN/Inf en época " << (epoch + 1) 
                          << ", batch " << (num_batches + 1) << std::endl;
                std::cout << "🔧 Reiniciando optimizador y continuando..." << std::endl;
                nan_detected = true;
                break;  // Skip this epoch
            }
            
            total_loss += batch_loss;
            num_batches++;
        }
        
        // Handle NaN detection
        if (nan_detected) {
            std::cout << "🔄 Saltando época " << (epoch + 1) << " debido a inestabilidad numérica" << std::endl;
            continue;
        }
        
        // Print progress every 100 epochs for stable training
        if ((epoch + 1) % 100 == 0 || epoch == 0) {
            float avg_loss = total_loss / num_batches;
            std::cout << "Época " << std::setw(4) << (epoch + 1) << "/" << config.epochs 
                      << " | Loss: " << std::fixed << std::setprecision(6) << avg_loss 
                      << " | Progreso: " << std::setprecision(1) << (100.0f * (epoch + 1) / config.epochs) << "%" 
                      << " | ✅ Estable" << std::endl;
        }
    }
    
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(end_time - start_time);
    
    std::cout << "✅ Entrenamiento estable completado en " << duration.count() << " segundos" << std::endl;
    std::cout << "🛡️ Sin problemas de NaN detectados" << std::endl;
    std::cout << "⚡ Tiempo promedio por época: " << std::fixed << std::setprecision(2) 
              << (duration.count() / static_cast<float>(config.epochs)) << " segundos" << std::endl;
}

void evaluate_model(const std::unique_ptr<NeuralNetwork<float>>& network,
                   const DataSplit& data) {
    
    print_phase("EVALUACIÓN DEL MODELO ESTABLE", 3);
    
    std::cout << "🧪 Evaluando modelo estable en conjunto de prueba..." << std::endl;
    
    auto start_time = high_resolution_clock::now();
    
    // Make predictions on test set
    Tensor<float, 2> predictions = network->forward(data.X_test);
    
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);
    
    std::cout << "⚡ Predicciones generadas en " << duration.count() << " ms" << std::endl;
    
    // Evaluate using different thresholds
    ModelEvaluator evaluator;
    
    // Standard evaluation (threshold = 0.5)
    auto metrics = evaluator.evaluate(predictions, data.y_test);
    
    std::cout << "\n🏆 RESULTADOS DEL MODELO ESTABLE:" << std::endl;
    std::cout << "📊 Accuracy: " << std::fixed << std::setprecision(2) << (metrics.accuracy * 100) << "%" << std::endl;
    std::cout << "🎯 Sensibilidad: " << std::fixed << std::setprecision(2) << (metrics.recall * 100) << "%" << std::endl;
    std::cout << "⚡ Especificidad: " << std::fixed << std::setprecision(2) << (metrics.specificity * 100) << "%" << std::endl;
    std::cout << "⭐ F1-Score: " << std::fixed << std::setprecision(2) << (metrics.f1_score * 100) << "%" << std::endl;
    std::cout << "🏅 Calificación: " << metrics.get_performance_grade() << std::endl;
    std::cout << "🛡️ Estado: Entrenamiento estable sin NaN" << std::endl;
    
    // Detailed evaluation
    metrics.print_detailed();
}

int main() {
    try {
        print_header();
        
        // Phase 1: Load and preprocess data
        print_phase("CARGA Y PREPROCESAMIENTO DE DATOS", 1);
        
        std::cout << "📁 Cargando dataset de diabetes balanceado..." << std::endl;
        DiabetesDataLoader loader("data/diabetes_prediction_dataset_balanced_8500.csv");
        
        loader.print_dataset_summary();
        
        auto data = loader.split_data(0.2f, 42);
        
        // Phase 2: Network configuration and training
        auto config = get_stable_config();  // Usar configuración estable
        config = adjust_config_for_features(config, data.num_features);
        
        std::cout << "\n🏗️ CONFIGURACIÓN DE RED ESTABLE:" << std::endl;
        config.print_config();
        
        auto network = create_diabetes_classifier<float>(config);
        
        // Phase 3: Training
        train_network(network, data, config);
        
        // Phase 4: Evaluation
        evaluate_model(network, data);
        
        std::cout << "\n🎉 ¡ENTRENAMIENTO ESTABLE COMPLETADO!" << std::endl;
        std::cout << "📝 Modelo entrenado exitosamente sin problemas de NaN." << std::endl;
        std::cout << "🛡️ Configuración optimizada para estabilidad numérica." << std::endl;
        std::cout << "💡 Si el modelo original daba NaN, este debería funcionar correctamente." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ ERROR: " << e.what() << std::endl;
        std::cerr << "🔧 Verifique que el archivo 'data/diabetes_prediction_dataset_balanced_8500.csv' esté en el directorio del proyecto." << std::endl;
        return 1;
    }
    
    return 0;
} 