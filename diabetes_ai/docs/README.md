[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Lj3YlzJp)
# Proyecto Final 2025-1: AI Neural Network - Predicción de Diabetes
## **CS2013 Programación III** · Informe Final

## **Descripción**

Implementación de una red neuronal multicapa en C++ para la predicción de diabetes mellitus, utilizando un dataset médico real balanceado. El proyecto demuestra la aplicación práctica de redes neuronales artificiales en el diagnóstico médico asistido.

## **Video**

Link: https://www.youtube.com/watch?v=mR9xgN0fn60

## Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Investigación teórica](#1-investigación-teórica)
4. [Diseño e implementación](#2-diseño-e-implementación)
5. [Ejecución](#3-ejecución)
6. [Análisis del rendimiento](#4-análisis-del-rendimiento)
7. [Trabajo en equipo](#5-trabajo-en-equipo)
8. [Conclusiones](#6-conclusiones)
9. [Bibliografía](#7-bibliografía)
10. [Licencia](#licencia)
---

## Datos generales

* **Tema**: Redes Neuronales para Predicción de Diabetes
* **Grupo**: `open-ai-team`
* **Integrantes**:

  * Francis Huerta Roque – 202310535 (Responsable de investigación teórica, desarrollo de la arquitectura)
  * Nicolas Fabian Riquero Urteaga – 202410759 (Implementación del modelo)
  * Maquera Quispe, Luis Fernando – 202410621 (Pruebas, benchmarking, documentación y demo)
  * Jossy Abigail Gamonal Retuerto - 202310643 (Implementación del modelo)

> *Nota: Reemplazar nombres y roles reales.*

---

## Requisitos e instalación

1. **Compilador**: C++20 compatible (GCC 11+, Clang 13+, MSVC 2022+)
2. **Herramientas de build**:
   * CMake 3.16+
   * Make o Ninja (Linux/macOS)
   * Visual Studio 2022 (Windows)
3. **Dependencias**: Ninguna (solo STL de C++)
4. **Instalación**:

   ```bash
   git clone https://github.com/CS1103/projecto-final-openai-team.git
   cd projecto-final-openai-team/diabetes_ai
   mkdir build && cd build
   cmake ..
   cmake --build . --config Release
   
   # Ejecutar entrenamiento estable
   ./bin/stable_train
   ```

---

## 1. Investigación teórica  

### 🎯 Objetivo

Explorar fundamentos y arquitecturas de redes neuronales, comprendiendo su evolución, funcionamiento interno y algoritmos de entrenamiento.

---

### 🧠 1.1 Historia y evolución de las redes neuronales

Las redes neuronales artificiales (ANNs) nacieron como una idea inspirada en la estructura biológica del cerebro humano.  
El primer modelo matemático fue propuesto por **McCulloch y Pitts** en 1943, quienes demostraron que una red simple de neuronas artificiales podía representar funciones lógicas básicas [3].

En 1958, el psicólogo **Frank Rosenblatt** desarrolló el *perceptrón*, un algoritmo capaz de clasificar entradas linealmente separables.  
Aunque fue un avance innovador para su tiempo, presentaba limitaciones: no podía resolver problemas como la función XOR.  
Estas carencias llevaron a una disminución del interés en los años 70.

El renacimiento llegó en los años 80 con la introducción del algoritmo de **retropropagación del error (backpropagation)**, que permitió entrenar redes con múltiples capas ocultas (*MLP*) [5].  
Este avance técnico marcó un antes y un después, haciendo posible modelar relaciones no lineales de forma más eficiente.

A partir de la década de 2010, el panorama cambió radicalmente:

- El crecimiento del *big data*.  
- La aparición de hardware especializado (como las GPUs).  
- Y la necesidad de resolver tareas más complejas...

...hicieron que las redes neuronales profundas (*deep learning*) tomaran el protagonismo en aplicaciones del mundo real [1].

> 🚀 Hoy en día, las ANNs son clave en tecnologías como **ChatGPT**, sistemas de reconocimiento facial, vehículos autónomos y diagnósticos médicos asistidos por IA.

---

### 🏗️ 1.2 Principales arquitecturas: MLP, CNN y RNN

Las arquitecturas de redes neuronales definen cómo se organizan las neuronas artificiales y qué tipo de datos pueden procesar eficientemente.

---

#### 🔹 MLP (Perceptrón Multicapa)

El MLP es la base de muchas redes modernas.  
Está compuesto por varias capas de neuronas (una capa de entrada, una o más capas ocultas y una capa de salida), donde cada neurona está conectada a todas las de la capa siguiente [2].  
Es ideal para tareas de clasificación, regresión y reconocimiento de patrones cuando los datos no tienen estructura espacial ni secuencial.

> 🧠 A pesar de su simplicidad, los MLP pueden aproximar funciones complejas si se entrenan correctamente y con suficientes capas.

---

#### 🔹 CNN (Convolutional Neural Networks)

Las CNNs están especialmente diseñadas para trabajar con datos estructurados en forma de matrices, como imágenes.  
Utilizan capas convolucionales que aplican filtros para detectar características locales (bordes, texturas, colores, formas), lo que reduce el número de parámetros necesarios y mejora el rendimiento [1].

**Aplicaciones destacadas de las CNNs:**
- 📷 Reconocimiento facial en tiempo real.  
- 🩺 Clasificación de imágenes médicas (tumores, fracturas, etc.).  
- 🚗 Sistemas de visión en vehículos autónomos.

---

#### 🔹 RNN (Recurrent Neural Networks)

Las RNNs están diseñadas para procesar secuencias de datos.  
A diferencia de las redes tradicionales, poseen conexiones recurrentes que les permiten "recordar" información previa [2].

**Usos comunes de las RNNs:**
- 📝 Procesamiento de lenguaje natural.  
- 🌐 Traducción automática.  
- 💬 Análisis de sentimientos.  
- 📈 Predicción de series temporales (finanzas, clima, etc.).

> 🧬 Existen versiones mejoradas como **LSTM** y **GRU**, que solucionan problemas como el desvanecimiento del gradiente y permiten memorizar secuencias más largas.

---

### ⚙️ 1.3 Algoritmos de entrenamiento: backpropagation y optimizadores

El entrenamiento de una red neuronal consiste en ajustar sus parámetros (pesos y sesgos) para minimizar la diferencia entre la salida esperada y la obtenida.  
Para lograr esto, se utilizan dos elementos clave:

---

#### 🔄 Backpropagation

Es un algoritmo que aplica la **regla de la cadena** del cálculo diferencial para distribuir el error de salida hacia las capas anteriores.  
Cada peso se actualiza en función del gradiente de la función de pérdida respecto a ese peso, permitiendo que la red aprenda patrones complejos [5].

> 💡 Su impacto fue tal que permitió pasar de redes simples a redes profundas con múltiples capas ocultas.

---

#### ⚙️ Optimizadores

El **optimizador** es el encargado de decidir cómo actualizar los pesos de la red durante el proceso de aprendizaje.  
Algunos de los más conocidos y utilizados son:

- **SGD (Stochastic Gradient Descent):**  
  Actualiza los pesos usando solo una muestra o mini-lote. Es eficiente pero puede ser sensible al *learning rate*.

- **Adam (Adaptive Moment Estimation):**  
  Combina ideas de *momentum* y adaptación dinámica. Es robusto, eficiente y muy popular en la práctica [4].

- **RMSprop y Adagrad:**  
  Ideales para datos dispersos o ruidosos. Ajustan la tasa de aprendizaje de manera adaptativa según la frecuencia de actualización de cada parámetro.

> ✅ Una buena elección del optimizador y de la tasa de aprendizaje (*learning rate*) puede marcar la diferencia entre una red que converge eficientemente y otra que nunca llega a aprender correctamente.

---

## 2. Diseño e implementación

### 🏗️ 2.1 Arquitectura de la solución

#### **Patrones de diseño implementados**
- **Factory Pattern**: Para la creación de diferentes configuraciones de red neuronal específicas para diabetes
- **Strategy Pattern**: Para optimizadores (SGD y Adam) intercambiables 
- **Template Pattern**: Para componentes genéricos de álgebra tensorial
- **Interface/Abstract**: Para capas, optimizadores y funciones de pérdida

#### **Estructura modular del proyecto**
```
diabetes_ai/
├── include/utec/                    # Headers organizados por módulos
│   ├── algebra/                     # Álgebra tensorial
│   │   └── tensor.h                 # Implementación de Tensor<T, Rank>
│   ├── nn/                          # Red neuronal: capas, activaciones, optimizadores
│   │   ├── nn_interfaces.h          # Interfaces base (ILayer, IOptimizer, ILoss)
│   │   ├── nn_dense.h               # Capa densa (fully connected)
│   │   ├── nn_activation.h          # Funciones de activación (ReLU, Sigmoid, Tanh)
│   │   ├── nn_loss.h                # Funciones de pérdida (Binary Cross Entropy)
│   │   ├── nn_optimizer.h           # Optimizadores (SGD, Adam)
│   │   └── neural_network.h         # Red neuronal principal
│   └── diabetes/                    # Lógica específica del problema
│       ├── data_loader.h            # Carga y preprocesamiento de datos
│       ├── diabetes_network.h       # Configuraciones específicas para diabetes
│       └── model_evaluation.h       # Métricas de evaluación médica
├── src/                             # Código fuente principal
│   ├── main.cpp                     # Programa ejecutable completo
│   └── train_stable.cpp             # Entrenamiento estable (anti-NaN)
├── data/                            # Datasets
│   ├── diabetes_prediction_dataset.csv           # Dataset original
│   └── diabetes_prediction_dataset_balanced_8500.csv  # Dataset balanceado
├── scripts/                         # Scripts auxiliares en Python
│   ├── Downsampling.py              # Reducción del dataset
│   └── datos.py                     # Manipulación del dataset
├── bin/                             # Ejecutables compilados
│   ├── programa                     # Ejecutable principal
│   └── stable_train                 # Entrenamiento estable
└── docs/                            # Documentación
    ├── README.md
    └── INSTRUCCIONES_ENTRENAMIENTO.md
```

#### **Arquitectura de red neuronal implementada**
- **Tipo**: Perceptrón Multicapa (MLP) para clasificación binaria
- **Arquitectura**: `12 → 32 → 16 → 1`
  - **Capa de entrada**: 12 características preprocesadas
  - **Capas ocultas**: 32 y 16 neuronas con activación ReLU
  - **Capa de salida**: 1 neurona con activación Sigmoid (probabilidad de diabetes)

#### **Características técnicas**
- **Lenguaje**: C++20 con templates
- **Gestión de memoria**: Smart pointers y RAII
- **Estabilidad numérica**: Protección anti-NaN y configuraciones conservadoras
- **Modularidad**: Interfaces bien definidas para extensibilidad

---

### 📊 2.2 Dataset y preprocesamiento

#### **Dataset de diabetes**
- **Fuente**: Diabetes Prediction Dataset balanceado
- **Tamaño**: 17,000 registros (8,500 con diabetes, 8,500 sin diabetes)
- **Balance**: Perfectamente balanceado (50%-50%)

#### **Características del dataset**
| Característica | Descripción | Tipo |
|----------------|-------------|------|
| `gender` | Género del paciente | Categórica (Male/Female) |
| `age` | Edad en años | Numérica (μ=50.66, σ=21.45) |
| `hypertension` | Presencia de hipertensión | Binaria (0/1) |
| `heart_disease` | Enfermedad cardíaca | Binaria (0/1) |
| `smoking_history` | Historia de tabaquismo | Categórica (6 categorías) |
| `bmi` | Índice de masa corporal | Numérica (μ=29.43, σ=7.39) |
| `HbA1c_level` | Nivel de hemoglobina A1c | Numérica (μ=6.16, σ=1.28) |
| `blood_glucose_level` | Nivel de glucosa en sangre | Numérica (μ=163.24, σ=56.97) |
| `diabetes` | Diagnóstico de diabetes (objetivo) | Binaria (0/1) |

#### **Preprocesamiento implementado**
1. **Variables categóricas**:
   - `gender`: Codificación binaria (Female=0, Male=1)
   - `smoking_history`: One-hot encoding (6 categorías: 'No Info', 'current', 'ever', 'former', 'never', 'not current')

2. **Variables numéricas**:
   - Normalización Z-score: `(x - μ) / σ`
   - Aplicada a: age, bmi, HbA1c_level, blood_glucose_level

3. **División de datos**:
   - **Entrenamiento**: 80% (13,600 muestras)
   - **Prueba**: 20% (3,400 muestras)
   - **Semilla aleatoria**: 42 (reproducibilidad)

---

### 🛠️ 2.3 Manual de uso

#### **Compilación**
```bash
# Crear directorio de build
mkdir build && cd build

# Configurar con CMake
cmake ..

# Compilar (Release optimizado)
cmake --build . --config Release
```

#### **Ejecución**
```bash
# Programa principal completo
./bin/programa

# Entrenamiento estable (recomendado)
./bin/stable_train
```

#### **Configuraciones disponibles**
- **Simple**: Red 12→32→1, 500 épocas, SGD
- **Standard**: Red 12→64→32→16→1, 1500 épocas, Adam
- **Ultra-Stable**: Red 12→32→16→1, 1500 épocas, SGD conservador

---

### 🧪  2.4 Casos de prueba implementados

#### **Tests unitarios conceptuales**
- ✅ **Carga de datos**: Verificación de integridad del dataset
- ✅ **Preprocesamiento**: Validación de normalización y encoding
- ✅ **Forward pass**: Propagación correcta a través de capas
- ✅ **Backward pass**: Cálculo correcto de gradientes
- ✅ **Estabilidad numérica**: Detección y manejo de NaN/Inf

#### **Tests de integración**
- ✅ **Entrenamiento completo**: Convergencia en 1500 épocas
- ✅ **Evaluación**: Métricas médicas precisas
- ✅ **Casos clínicos**: Predicciones en pacientes sintéticos

---

## 3. Ejecución y resultados

### 🚀 3.1 Proceso de entrenamiento

#### **Configuración utilizada (Ultra-Stable)**
- **Arquitectura**: 12 → 32 → 16 → 1
- **Optimizador**: SGD (más estable que Adam)
- **Learning rate**: 0.0005 (conservador)
- **Épocas**: 1,500
- **Batch size**: 64
- **Función de pérdida**: Binary Cross Entropy
- **Activaciones**: ReLU (capas ocultas), Sigmoid (salida)

#### **Progreso del entrenamiento**
```
Época    1/1500 | Loss: 0.770434 | Progreso: 0.1% | ✅ Estable
Época  100/1500 | Loss: 0.339640 | Progreso: 6.7% | ✅ Estable
Época  500/1500 | Loss: 0.320597 | Progreso: 33.3% | ✅ Estable
Época 1000/1500 | Loss: 0.314698 | Progreso: 66.7% | ✅ Estable
Época 1500/1500 | Loss: 0.309518 | Progreso: 100.0% | ✅ Estable
```

#### **Estadísticas de entrenamiento**
- ⏱️ **Tiempo total**: 1,791 segundos (≈30 minutos)
- ⚡ **Tiempo por época**: 1.19 segundos promedio
- 🛡️ **Estabilidad**: Sin problemas de NaN detectados
- 📊 **Convergencia**: Pérdida decrece consistentemente

---

### 📈 3.2 Evaluación en conjunto de prueba

#### **Matriz de confusión**
```
                 Predicción
                No    Sí
Real    No    1423   283
        Sí     214  1480
```

#### **Métricas principales**
| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Accuracy** | 85.38% | Precisión general del modelo |
| **Sensibilidad (Recall)** | 87.37% | Detección de casos positivos |
| **Especificidad** | 83.41% | Evitar falsos positivos |
| **Precisión** | 83.95% | Confiabilidad de predicciones positivas |
| **F1-Score** | 85.62% | Balance entre precisión y recall |
| **Calificación** | B+ (Bueno) | Evaluación general del modelo |

---

### 🏥 3.3 Interpretación médica

#### **Análisis clínico**
- 🩺 **Sensibilidad (87.37%)**: Buena detección de casos positivos de diabetes
- 🛡️ **Especificidad (83.41%)**: Buena capacidad para evitar falsos positivos  
- ⚖️ **Balance**: Muy balanceado entre sensibilidad y especificidad
- 💊 **Uso clínico**: Útil para screening inicial, requiere confirmación médica

#### **Casos clínicos demostrativos**
1. **Paciente alto riesgo**: Hombre, 65 años, hipertensión, ex-fumador, BMI 28.5
   - **Predicción**: ~75% probabilidad de diabetes
   - **Recomendación**: Evaluación médica urgente

2. **Paciente bajo riesgo**: Mujer, 28 años, sin comorbilidades, BMI 22.0
   - **Predicción**: ~15% probabilidad de diabetes
   - **Recomendación**: Seguimiento rutinario

---

## 4. Análisis del rendimiento

### ⚡ 4.1 Métricas de rendimiento

#### **Rendimiento computacional**
- **Tiempo de entrenamiento**: 1,791 segundos (29.85 minutos)
- **Tiempo de predicción**: 93 ms para 3,400 muestras
- **Velocidad de predicción**: 0.027 ms por muestra
- **Uso de memoria**: Optimizado con smart pointers

#### **Convergencia del modelo**
- **Épocas necesarias**: 1,500 (configuración conservadora)
- **Estabilidad**: 100% (sin episodios de NaN)
- **Pérdida final**: 0.309518 (Binary Cross Entropy)
- **Tendencia**: Convergencia consistente y estable

---

### 📊 4.2 Comparación de configuraciones

| Configuración | Arquitectura | Entrenamiento | Accuracy | F1-Score | Estabilidad |
|---------------|--------------|---------------|----------|----------|-------------|
| Simple | 12→32→1 | 500 épocas | ~82% | ~81% | Alta |
| Standard | 12→64→32→16→1 | 1500 épocas | ~87% | ~86% | Media |
| **Ultra-Stable** | **12→32→16→1** | **1500 épocas** | **85.38%** | **85.62%** | **Muy Alta** |

---

### ✅ 4.3 Ventajas y limitaciones

#### **Ventajas**
- ✅ **Implementación desde cero**: Control total sobre la arquitectura
- ✅ **Estabilidad numérica**: Robusto ante problemas de NaN
- ✅ **Modularidad**: Fácil extensión y mantenimiento
- ✅ **Sin dependencias externas**: Solo STL de C++
- ✅ **Rendimiento clínicamente útil**: Sensibilidad y especificidad balanceadas

#### **Limitaciones**
- ❌ **Sin paralelización**: Entrenamiento secuencial
- ❌ **Dataset limitado**: 17K muestras (pequeño para deep learning)
- ❌ **Arquitectura simple**: Solo MLP, sin CNNs o RNNs
- ❌ **Optimización manual**: Sin autotuning de hiperparámetros

---

### 🚀 4.4 Mejoras futuras

#### **Optimizaciones técnicas**
1. **Paralelización**: OpenMP para entrenamiento multi-core
2. **Optimización matemática**: Uso de BLAS para multiplicaciones matriciales
3. **GPU Computing**: Implementación CUDA para aceleración
4. **Memoria**: Pool de memoria para reducir allocations

#### **Mejoras del modelo**
1. **Regularización**: Dropout y weight decay para prevenir overfitting
2. **Arquitecturas avanzadas**: Batch normalization y residual connections
3. **Ensemble methods**: Combinación de múltiples modelos
4. **Autotuning**: Búsqueda automática de hiperparámetros

#### **Extensiones médicas**
1. **Más datos**: Integrar datasets adicionales de diabetes
2. **Características avanzadas**: Análisis de imágenes médicas
3. **Predicción temporal**: Series de tiempo para progresión de diabetes
4. **Explicabilidad**: SHAP values para interpretación médica

---

## 5. Trabajo en equipo

### 👥 5.1 Distribución de tareas

| Tarea | Responsable | Descripción | Estado |
|-------|-------------|-------------|--------|
| Investigación teórica | Equipo completo | Fundamentos de redes neuronales y diabetes | ✅ Completado |
| Diseño de arquitectura | Equipo completo | Estructura modular y patrones de diseño | ✅ Completado |
| Implementación core | Equipo completo | Tensor, capas, optimizadores | ✅ Completado |
| Módulo de diabetes | Equipo completo | Data loader y evaluación médica | ✅ Completado |
| Testing y validación | Equipo completo | Casos de prueba y estabilidad | ✅ Completado |
| Documentación | Equipo completo | README, comentarios y manuales | ✅ Completado |

---

### 🤝 5.2 Metodología de trabajo

#### **Herramientas utilizadas**
- **Control de versiones**: Git con GitHub
- **Compilación**: CMake multiplataforma
- **Documentación**: Markdown con emojis descriptivos
- **Testing**: Validación manual y casos clínicos

#### **Proceso de desarrollo**
1. **Investigación**: Revisión de literatura sobre diabetes y ML
2. **Diseño**: Arquitectura modular con interfaces claras
3. **Implementación**: Desarrollo incremental con testing continuo
4. **Validación**: Pruebas con dataset real de diabetes
5. **Optimización**: Configuración estable y anti-NaN
6. **Documentación**: Manual completo y casos de uso

---

## 6. Conclusiones

### 🎯 6.1 Logros alcanzados

#### **Objetivos técnicos cumplidos**
- ✅ **Red neuronal desde cero**: Implementación completa en C++ sin librerías externas
- ✅ **Problema real**: Predicción de diabetes con dataset médico real  
- ✅ **Arquitectura modular**: Código extensible y mantenible
- ✅ **Estabilidad numérica**: Sistema robusto sin problemas de NaN
- ✅ **Rendimiento clínico**: Accuracy 85.38%, F1-Score 85.62%

#### **Objetivos académicos cumplidos**
- ✅ **Comprensión profunda**: Backpropagation implementado desde cero
- ✅ **Programación avanzada**: Templates, RAII, smart pointers
- ✅ **Trabajo en equipo**: Desarrollo colaborativo exitoso
- ✅ **Documentación**: Manual técnico y médico completo

---

### 📚 6.2 Aprendizajes obtenidos

#### **Conocimientos técnicos**
- 🧠 **Redes neuronales**: Comprensión profunda de MLP y backpropagation
- 💻 **C++ avanzado**: Templates, gestión de memoria, arquitectura modular
- 📊 **Machine Learning**: Preprocesamiento, evaluación, métricas médicas
- 🏥 **Aplicaciones médicas**: Importancia de sensibilidad/especificidad

#### **Habilidades blandas**
- 🤝 **Colaboración**: Trabajo efectivo en equipo distribuido
- 📝 **Documentación**: Comunicación técnica clara y precisa
- 🔍 **Debugging**: Identificación y solución de problemas numéricos
- ⏰ **Gestión de tiempo**: Planificación y entrega de proyecto completo

---

### 🔬 6.3 Evaluación del modelo

#### **Fortalezas del modelo**
- 🎯 **Balance**: Sensibilidad y especificidad bien balanceadas
- 🛡️ **Estabilidad**: Entrenamiento consistente sin fallos numéricos
- 🏥 **Utilidad clínica**: Útil para screening inicial de diabetes
- 📈 **Convergencia**: Aprendizaje efectivo en dataset balanceado

#### **Áreas de mejora identificadas**
- 🔧 **Arquitectura**: Explorar redes más profundas o complejas
- 📊 **Datos**: Ampliar dataset con más características médicas
- ⚡ **Rendimiento**: Optimizar para datasets más grandes
- 🧠 **Explicabilidad**: Añadir interpretación de decisiones

---

### 🚀 6.4 Recomendaciones futuras

#### **Para uso médico**
1. **Validación clínica**: Pruebas con médicos especialistas
2. **Datos adicionales**: Incorporar historiales médicos completos
3. **Regulación**: Cumplir estándares médicos y de privacidad
4. **Interfaz médica**: GUI intuitiva para personal sanitario

#### **Para desarrollo técnico**
1. **Escalabilidad**: Migrar a frameworks como PyTorch para datasets grandes
2. **Productización**: API REST para integración en sistemas médicos
3. **Monitoring**: Sistema de monitoreo de performance en producción
4. **A/B Testing**: Comparación con modelos comerciales existentes

#### **Para investigación académica**
1. **Paper científico**: Publicar resultados en conferencia de ML médico
2. **Datasets públicos**: Contribuir con código open-source
3. **Benchmarking**: Comparar con estado del arte en diabetes prediction
4. **Extensiones**: Aplicar metodología a otras enfermedades

---

##  7. Bibliografía 

[1] Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," *Nature*, vol. 521, no. 7553, pp. 436–444, 2015.  
[2] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*, MIT Press, 2016.  
[3] S. Haykin, *Neural Networks and Learning Machines*, 3rd ed., Pearson, 2009.  
[4] C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006.  
[5] D. E. Rumelhart, G. E. Hinton, and R. J. Williams, "Learning representations by back-propagating errors," *Nature*, vol. 323, pp. 533–536, 1986.  
[6] American Diabetes Association, "Classification and Diagnosis of Diabetes," *Diabetes Care*, vol. 44, Supplement 1, pp. S15-S33, 2021.  
[7] P. Saeedi et al., "Global and regional diabetes prevalence estimates for 2019 and projections for 2030 and 2045," *Diabetes Research and Clinical Practice*, vol. 157, 2019.  
[8] A. Géron, *Hands-On Machine Learning with Scikit-Learn and TensorFlow*, 2nd ed., O'Reilly Media, 2019.

---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
