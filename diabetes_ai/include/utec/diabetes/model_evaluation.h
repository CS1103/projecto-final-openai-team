//
// Model evaluation metrics for diabetes prediction
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_MODEL_EVALUATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_MODEL_EVALUATION_H

#include "../algebra/tensor.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>

namespace utec::evaluation {

using namespace utec::algebra;

// Confusion matrix structure
struct ConfusionMatrix {
    size_t true_positives;
    size_t true_negatives;
    size_t false_positives;
    size_t false_negatives;
    
    size_t total() const {
        return true_positives + true_negatives + false_positives + false_negatives;
    }
    
    void print() const {
        std::cout << "\n📊 MATRIZ DE CONFUSIÓN:" << std::endl;
        std::cout << "                 Predicción" << std::endl;
        std::cout << "                No    Sí" << std::endl;
        std::cout << "Real    No   " << std::setw(5) << true_negatives << " " << std::setw(5) << false_positives << std::endl;
        std::cout << "        Sí   " << std::setw(5) << false_negatives << " " << std::setw(5) << true_positives << std::endl;
    }
};

// Classification metrics
struct ClassificationMetrics {
    float accuracy;
    float precision;
    float recall;        // También conocido como sensibilidad
    float specificity;
    float f1_score;
    float auc_roc;       // Area Under the Curve - Receiver Operating Characteristic
    ConfusionMatrix confusion_matrix;
    
    void print_detailed() const {
        std::cout << "\n🏆 MÉTRICAS DE EVALUACIÓN DETALLADAS:" << std::endl;
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
        
        confusion_matrix.print();
        
        std::cout << "\n📈 MÉTRICAS GENERALES:" << std::endl;
        std::cout << "📊 Accuracy:     " << std::fixed << std::setprecision(4) << accuracy << " (" << (accuracy * 100) << "%)" << std::endl;
        std::cout << "🎯 Precision:    " << std::fixed << std::setprecision(4) << precision << " (" << (precision * 100) << "%)" << std::endl;
        std::cout << "⚡ Recall:       " << std::fixed << std::setprecision(4) << recall << " (" << (recall * 100) << "%)" << std::endl;
        std::cout << "🛡️ Specificity:  " << std::fixed << std::setprecision(4) << specificity << " (" << (specificity * 100) << "%)" << std::endl;
        std::cout << "⭐ F1-Score:     " << std::fixed << std::setprecision(4) << f1_score << " (" << (f1_score * 100) << "%)" << std::endl;
        
        std::cout << "\n🏥 INTERPRETACIÓN MÉDICA:" << std::endl;
        print_medical_interpretation();
        
        std::cout << "\n💡 RECOMENDACIONES:" << std::endl;
        print_recommendations();
    }
    
    void print_medical_interpretation() const {
        std::cout << "🩺 Sensibilidad (Recall): " << (recall * 100) << "% - ";
        if (recall >= 0.9) {
            std::cout << "Excelente detección de casos positivos de diabetes" << std::endl;
        } else if (recall >= 0.8) {
            std::cout << "Buena detección de casos positivos de diabetes" << std::endl;
        } else if (recall >= 0.7) {
            std::cout << "Aceptable detección de casos positivos, pero puede mejorar" << std::endl;
        } else {
            std::cout << "❌ CRÍTICO: Muchos casos de diabetes no detectados" << std::endl;
        }
        
        std::cout << "🛡️ Especificidad: " << (specificity * 100) << "% - ";
        if (specificity >= 0.9) {
            std::cout << "Excelente en evitar falsos positivos" << std::endl;
        } else if (specificity >= 0.8) {
            std::cout << "Buena en evitar falsos positivos" << std::endl;
        } else {
            std::cout << "⚠️ Puede generar muchas falsas alarmas" << std::endl;
        }
        
        std::cout << "⚖️ Balance Sensibilidad-Especificidad: ";
        float balance_diff = std::abs(recall - specificity);
        if (balance_diff <= 0.05) {
            std::cout << "Muy balanceado ✅" << std::endl;
        } else if (balance_diff <= 0.1) {
            std::cout << "Bien balanceado ✅" << std::endl;
        } else {
            std::cout << "Desbalanceado ⚠️" << std::endl;
        }
    }
    
    void print_recommendations() const {
        if (recall < 0.8) {
            std::cout << "📈 Mejorar sensibilidad: Ajustar umbral de decisión hacia abajo" << std::endl;
        }
        
        if (specificity < 0.8) {
            std::cout << "🎯 Mejorar especificidad: Ajustar umbral de decisión hacia arriba" << std::endl;
        }
        
        if (f1_score < 0.8) {
            std::cout << "🔧 F1-Score bajo: Considerar más datos de entrenamiento o ajuste de hiperparámetros" << std::endl;
        }
        
        if (accuracy < 0.85) {
            std::cout << "📚 Accuracy bajo: Revisar calidad de datos y arquitectura del modelo" << std::endl;
        }
        
        // Medical recommendations
        if (recall >= 0.9 && specificity >= 0.85) {
            std::cout << "🏆 MODELO CLÍNICAMENTE ACEPTABLE para screening de diabetes" << std::endl;
        } else if (recall >= 0.85) {
            std::cout << "🩺 Modelo útil para screening inicial, requiere confirmación médica" << std::endl;
        } else {
            std::cout << "⚠️ Modelo NO recomendado para uso clínico sin mejoras significativas" << std::endl;
        }
    }
    
    std::string get_performance_grade() const {
        float overall_score = (accuracy + f1_score + ((recall + specificity) / 2)) / 3;
        
        if (overall_score >= 0.95) return "A+ (Excelente)";
        if (overall_score >= 0.90) return "A (Muy Bueno)";
        if (overall_score >= 0.85) return "B+ (Bueno)";
        if (overall_score >= 0.80) return "B (Aceptable)";
        if (overall_score >= 0.75) return "C+ (Regular)";
        if (overall_score >= 0.70) return "C (Necesita Mejora)";
        return "D (Inaceptable)";
    }
};

class ModelEvaluator {
private:
    float threshold_;

public:
    explicit ModelEvaluator(float threshold = 0.5f) : threshold_(threshold) {}
    
    void set_threshold(float threshold) {
        threshold_ = threshold;
    }
    
    float get_threshold() const {
        return threshold_;
    }
    
    ConfusionMatrix compute_confusion_matrix(const Tensor<float, 2>& predictions, 
                                           const Tensor<float, 2>& targets) const {
        if (predictions.shape() != targets.shape()) {
            throw std::runtime_error("Las predicciones y objetivos deben tener la misma forma");
        }
        
        ConfusionMatrix cm = {0, 0, 0, 0};
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            bool predicted_positive = predictions[i] >= threshold_;
            bool actual_positive = targets[i] >= 0.5f;
            
            if (predicted_positive && actual_positive) {
                cm.true_positives++;
            } else if (!predicted_positive && !actual_positive) {
                cm.true_negatives++;
            } else if (predicted_positive && !actual_positive) {
                cm.false_positives++;
            } else {
                cm.false_negatives++;
            }
        }
        
        return cm;
    }
    
    ClassificationMetrics evaluate(const Tensor<float, 2>& predictions, 
                                 const Tensor<float, 2>& targets) const {
        
        ConfusionMatrix cm = compute_confusion_matrix(predictions, targets);
        
        ClassificationMetrics metrics;
        metrics.confusion_matrix = cm;
        
        // Calculate basic metrics
        float tp = static_cast<float>(cm.true_positives);
        float tn = static_cast<float>(cm.true_negatives);
        float fp = static_cast<float>(cm.false_positives);
        float fn = static_cast<float>(cm.false_negatives);
        
        // Accuracy
        metrics.accuracy = (tp + tn) / (tp + tn + fp + fn);
        
        // Precision
        metrics.precision = (tp + fp > 0) ? tp / (tp + fp) : 0.0f;
        
        // Recall (Sensitivity)
        metrics.recall = (tp + fn > 0) ? tp / (tp + fn) : 0.0f;
        
        // Specificity
        metrics.specificity = (tn + fp > 0) ? tn / (tn + fp) : 0.0f;
        
        // F1 Score
        metrics.f1_score = (metrics.precision + metrics.recall > 0) ?
            2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall) : 0.0f;
        
        // Placeholder for AUC-ROC (simplified calculation)
        metrics.auc_roc = calculate_simple_auc(predictions, targets);
        
        return metrics;
    }
    
    float calculate_simple_auc(const Tensor<float, 2>& predictions, 
                              const Tensor<float, 2>& targets) const {
        // Simplified AUC calculation - in a real implementation, you'd want a more sophisticated approach
        std::vector<std::pair<float, float>> pairs;
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            pairs.emplace_back(predictions[i], targets[i]);
        }
        
        // Sort by prediction score
        std::sort(pairs.begin(), pairs.end());
        
        float auc = 0.0f;
        size_t positive_count = 0;
        size_t negative_count = 0;
        
        for (const auto& pair : pairs) {
            if (pair.second >= 0.5f) {
                positive_count++;
            } else {
                negative_count++;
            }
        }
        
        if (positive_count == 0 || negative_count == 0) {
            return 0.5f; // No discrimination possible
        }
        
        // Simple approximation
        size_t correctly_ranked = 0;
        for (size_t i = 0; i < pairs.size(); ++i) {
            if (pairs[i].second >= 0.5f) { // Positive case
                for (size_t j = 0; j < i; ++j) {
                    if (pairs[j].second < 0.5f) { // Negative case ranked lower
                        correctly_ranked++;
                    }
                }
            }
        }
        
        auc = static_cast<float>(correctly_ranked) / (positive_count * negative_count);
        return auc;
    }
    
    // Threshold analysis for optimal cutoff point
    struct ThresholdAnalysis {
        float threshold;
        float sensitivity;
        float specificity;
        float f1_score;
        float accuracy;
        float youden_index; // sensitivity + specificity - 1
        
        void print() const {
            std::cout << "Threshold: " << std::fixed << std::setprecision(3) << threshold
                      << " | Sens: " << std::setprecision(3) << sensitivity
                      << " | Spec: " << std::setprecision(3) << specificity
                      << " | F1: " << std::setprecision(3) << f1_score
                      << " | Acc: " << std::setprecision(3) << accuracy
                      << " | Youden: " << std::setprecision(3) << youden_index << std::endl;
        }
    };
    
    std::vector<ThresholdAnalysis> analyze_thresholds(const Tensor<float, 2>& predictions, 
                                                    const Tensor<float, 2>& targets,
                                                    size_t num_thresholds = 20) const {
        std::vector<ThresholdAnalysis> analyses;
        
        for (size_t i = 0; i <= num_thresholds; ++i) {
            float threshold = static_cast<float>(i) / num_thresholds;
            
            // Temporarily change threshold
            ModelEvaluator temp_evaluator(threshold);
            auto metrics = temp_evaluator.evaluate(predictions, targets);
            
            ThresholdAnalysis analysis;
            analysis.threshold = threshold;
            analysis.sensitivity = metrics.recall;
            analysis.specificity = metrics.specificity;
            analysis.f1_score = metrics.f1_score;
            analysis.accuracy = metrics.accuracy;
            analysis.youden_index = metrics.recall + metrics.specificity - 1.0f;
            
            analyses.push_back(analysis);
        }
        
        return analyses;
    }
    
    float find_optimal_threshold(const Tensor<float, 2>& predictions, 
                                const Tensor<float, 2>& targets,
                                const std::string& criterion = "youden") const {
        
        auto analyses = analyze_thresholds(predictions, targets, 100);
        
        float best_threshold = 0.5f;
        float best_score = -1.0f;
        
        for (const auto& analysis : analyses) {
            float score;
            
            if (criterion == "youden") {
                score = analysis.youden_index;
            } else if (criterion == "f1") {
                score = analysis.f1_score;
            } else if (criterion == "accuracy") {
                score = analysis.accuracy;
            } else {
                score = analysis.youden_index; // Default
            }
            
            if (score > best_score) {
                best_score = score;
                best_threshold = analysis.threshold;
            }
        }
        
        return best_threshold;
    }
    
    void print_threshold_analysis(const Tensor<float, 2>& predictions, 
                                const Tensor<float, 2>& targets) const {
        std::cout << "\n🔍 ANÁLISIS DE UMBRALES:" << std::endl;
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
        
        auto analyses = analyze_thresholds(predictions, targets, 10);
        
        for (const auto& analysis : analyses) {
            analysis.print();
        }
        
        float optimal_youden = find_optimal_threshold(predictions, targets, "youden");
        float optimal_f1 = find_optimal_threshold(predictions, targets, "f1");
        
        std::cout << "\n🎯 UMBRALES ÓPTIMOS:" << std::endl;
        std::cout << "Youden Index: " << optimal_youden << std::endl;
        std::cout << "F1-Score: " << optimal_f1 << std::endl;
    }
};

} // namespace utec::evaluation

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_MODEL_EVALUATION_H 