//
// Programa de entrenamiento - Red Neuronal Diabetes (500 épocas)
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
    std::cout << "██      ENTRENAMIENTO RÁPIDO - RED NEURONAL DIABETES (500)     ██" << std::endl;
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
    
    print_phase("ENTRENAMIENTO DE LA RED NEURONAL", 2);
    
    std::cout << "🚀 Iniciando entrenamiento rápido..." << std::endl;
    std::cout << "📊 Datos de entrenamiento: " << data.train_samples << " muestras" << std::endl;
    std::cout << "⚡ Configuración: " << config.epochs << " épocas, batch size: " << config.batch_size << std::endl;
    std::cout << "🏗️ Arquitectura: Simple (entrada → 32 → salida)" << std::endl;
    
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
        
        // Print progress every 50 epochs for faster training
        if ((epoch + 1) % 50 == 0 || epoch == 0) {
            float avg_loss = total_loss / num_batches;
            std::cout << "Época " << std::setw(3) << (epoch + 1) << "/" << config.epochs 
                      << " | Loss: " << std::fixed << std::setprecision(6) << avg_loss 
                      << " | Progreso: " << std::setprecision(1) << (100.0f * (epoch + 1) / config.epochs) << "%" << std::endl;
        }
    }
    
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(end_time - start_time);
    
    std::cout << "✅ Entrenamiento completado en " << duration.count() << " segundos" << std::endl;
    std::cout << "⚡ Tiempo promedio por época: " << std::fixed << std::setprecision(2) 
              << (duration.count() / static_cast<float>(config.epochs)) << " segundos" << std::endl;
}

void evaluate_model(const std::unique_ptr<NeuralNetwork<float>>& network,
                   const DataSplit& data) {
    
    print_phase("EVALUACIÓN DEL MODELO", 3);
    
    std::cout << "🧪 Evaluando modelo en conjunto de prueba..." << std::endl;
    
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
    
    std::cout << "\n🏆 RESULTADOS PRINCIPALES (500 ÉPOCAS):" << std::endl;
    std::cout << "📊 Accuracy: " << std::fixed << std::setprecision(2) << (metrics.accuracy * 100) << "%" << std::endl;
    std::cout << "🎯 Sensibilidad: " << std::fixed << std::setprecision(2) << (metrics.recall * 100) << "%" << std::endl;
    std::cout << "⚡ Especificidad: " << std::fixed << std::setprecision(2) << (metrics.specificity * 100) << "%" << std::endl;
    std::cout << "⭐ F1-Score: " << std::fixed << std::setprecision(2) << (metrics.f1_score * 100) << "%" << std::endl;
    std::cout << "🏅 Calificación: " << metrics.get_performance_grade() << std::endl;
    
    // Detailed evaluation
    metrics.print_detailed();
}

int main() {
    try {
        print_header();
        
        // Phase 1: Load and preprocess data
        print_phase("CARGA Y PREPROCESAMIENTO DE DATOS", 1);
        
        std::cout << "📁 Cargando dataset de diabetes balanceado..." << std::endl;
        DiabetesDataLoader loader("diabetes_prediction_dataset_balanced_8500.csv");
        
        loader.print_dataset_summary();
        
        auto data = loader.split_data(0.2f, 42);
        
        // Phase 2: Network configuration and training
        auto config = get_fast_500_config();
        config = adjust_config_for_features(config, data.num_features);
        
        std::cout << "\n🏗️ CONFIGURACIÓN DE RED (500 ÉPOCAS):" << std::endl;
        config.print_config();
        
        auto network = create_diabetes_classifier<float>(config);
        
        // Phase 3: Training
        train_network(network, data, config);
        
        // Phase 4: Evaluation
        evaluate_model(network, data);
        
        std::cout << "\n🎉 ¡ENTRENAMIENTO RÁPIDO COMPLETADO!" << std::endl;
        std::cout << "📝 Modelo entrenado con " << config.epochs << " épocas." << std::endl;
        std::cout << "⏱️ Este fue un entrenamiento rápido para pruebas." << std::endl;
        std::cout << "📈 Para mejor rendimiento, considera usar más épocas." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ ERROR: " << e.what() << std::endl;
        std::cerr << "🔧 Verifique que el archivo 'diabetes_prediction_dataset_balanced_8500.csv' esté en el directorio actual." << std::endl;
        return 1;
    }
    
    return 0;
} 