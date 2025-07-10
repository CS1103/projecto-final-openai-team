//
// Programa de entrenamiento - Red Neuronal Diabetes (1000 épocas)
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
    std::cout << "██      ENTRENAMIENTO MEDIO - RED NEURONAL DIABETES (1000)     ██" << std::endl;
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
    
    std::cout << "🚀 Iniciando entrenamiento medio..." << std::endl;
    std::cout << "📊 Datos de entrenamiento: " << data.train_samples << " muestras" << std::endl;
    std::cout << "⚡ Configuración: " << config.epochs << " épocas, batch size: " << config.batch_size << std::endl;
    std::cout << "🏗️ Arquitectura: Simple (entrada → 32 → salida)" << std::endl;
    std::cout << "⏱️ Tiempo estimado: ~2x más que entrenamiento de 500 épocas" << std::endl;
    
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
        
        // Print progress every 100 epochs for medium training
        if ((epoch + 1) % 100 == 0 || epoch == 0) {
            float avg_loss = total_loss / num_batches;
            std::cout << "Época " << std::setw(4) << (epoch + 1) << "/" << config.epochs 
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
    
    std::cout << "\n🏆 RESULTADOS PRINCIPALES (1000 ÉPOCAS):" << std::endl;
    std::cout << "📊 Accuracy: " << std::fixed << std::setprecision(2) << (metrics.accuracy * 100) << "%" << std::endl;
    std::cout << "🎯 Sensibilidad: " << std::fixed << std::setprecision(2) << (metrics.recall * 100) << "%" << std::endl;
    std::cout << "⚡ Especificidad: " << std::fixed << std::setprecision(2) << (metrics.specificity * 100) << "%" << std::endl;
    std::cout << "⭐ F1-Score: " << std::fixed << std::setprecision(2) << (metrics.f1_score * 100) << "%" << std::endl;
    std::cout << "🏅 Calificación: " << metrics.get_performance_grade() << std::endl;
    
    // Detailed evaluation
    metrics.print_detailed();
}

void demonstrate_predictions(const std::unique_ptr<NeuralNetwork<float>>& network,
                           const DiabetesDataLoader& loader) {
    
    print_phase("DEMOSTRACIÓN DE PREDICCIONES", 4);
    
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
        auto config = get_medium_1000_config();
        config = adjust_config_for_features(config, data.num_features);
        
        std::cout << "\n🏗️ CONFIGURACIÓN DE RED (1000 ÉPOCAS):" << std::endl;
        config.print_config();
        
        auto network = create_medium_1000_diabetes_classifier<float>();
        
        // Phase 3: Training
        train_network(network, data, config);
        
        // Phase 4: Evaluation
        evaluate_model(network, data);
        
        // Phase 5: Demonstration (with fewer examples)
        demonstrate_predictions(network, loader);
        
        std::cout << "\n🎉 ¡ENTRENAMIENTO MEDIO COMPLETADO!" << std::endl;
        std::cout << "📝 Modelo entrenado con " << config.epochs << " épocas." << std::endl;
        std::cout << "⚖️ Balance entre velocidad y rendimiento." << std::endl;
        std::cout << "📈 Debería mostrar mejor rendimiento que 500 épocas." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ ERROR: " << e.what() << std::endl;
        std::cerr << "🔧 Verifique que el archivo 'diabetes_prediction_dataset_balanced_8500.csv' esté en el directorio actual." << std::endl;
        return 1;
    }
    
    return 0;
} 