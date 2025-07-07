[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Lj3YlzJp)
# Proyecto Final 2025-1: AI Neural Network
## **CS2013 Programación III** · Informe Final

### **Descripción**

> Ejemplo: Implementación de una red neuronal multicapa en C++ para clasificación de dígitos manuscritos.

### Contenidos

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

### Datos generales

* **Tema**: Redes Neuronales en AI
* **Grupo**: `group_3_custom_name`
* **Integrantes**:

  * Alumno A – 209900001 (Responsable de investigación teórica)
  * Alumno B – 209900002 (Desarrollo de la arquitectura)
  * Alumno C – 209900003 (Implementación del modelo)
  * Alumno D – 209900004 (Pruebas y benchmarking)
  * Alumno E – 209900005 (Documentación y demo)

> *Nota: Reemplazar nombres y roles reales.*

---

### Requisitos e instalación

1. **Compilador**: GCC 11 o superior
2. **Dependencias**:

   * CMake 3.18+
   * Eigen 3.4
   * \[Otra librería opcional]
3. **Instalación**:

   ```bash
   git clone https://github.com/EJEMPLO/proyecto-final.git
   cd proyecto-final
   mkdir build && cd build
   cmake ..
   make
   ```

> *Ejemplo de repositorio y comandos, ajustar según proyecto.*

---

### 1. Investigación teórica

* **Objetivo**: Explorar fundamentos y arquitecturas de redes neuronales.
* **Contenido de ejemplo**:

## 🧠 1. Historia y evolución de las redes neuronales

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


## 🏗️ 2. Principales arquitecturas: MLP, CNN y RNN

Las arquitecturas de redes neuronales definen cómo se organizan las neuronas artificiales y qué tipo de datos pueden procesar eficientemente.

---

### 🔹 MLP (Perceptrón Multicapa)

El MLP es la base de muchas redes modernas. Está compuesto por varias capas de neuronas (una capa de entrada, una o más capas ocultas y una capa de salida), donde cada neurona está conectada a todas las de la capa siguiente [2].  
Es ideal para tareas de clasificación, regresión y reconocimiento de patrones cuando los datos no tienen estructura espacial ni secuencial.

> 🧠 A pesar de su simplicidad, los MLP pueden aproximar funciones complejas si se entrenan correctamente y con suficientes capas.

---

### 🔹 CNN (Convolutional Neural Networks)

Las CNNs están especialmente diseñadas para trabajar con datos estructurados en forma de matrices, como imágenes.  
Utilizan capas convolucionales que aplican filtros para detectar características locales (bordes, texturas, colores, formas), lo que reduce el número de parámetros necesarios y mejora el rendimiento [1].

**Aplicaciones destacadas de las CNNs:**
- 📷 Reconocimiento facial en tiempo real.  
- 🩺 Clasificación de imágenes médicas (tumores, fracturas, etc.).  
- 🚗 Sistemas de visión en vehículos autónomos.

---

### 🔹 RNN (Recurrent Neural Networks)

Las RNNs están diseñadas para procesar secuencias de datos.  
A diferencia de las redes tradicionales, poseen conexiones recurrentes que les permiten “recordar” información previa [2].

**Usos comunes de las RNNs:**
- 📝 Procesamiento de lenguaje natural.  
- 🌐 Traducción automática.  
- 💬 Análisis de sentimientos.  
- 📈 Predicción de series temporales (finanzas, clima, etc.).

> 🧬 Existen versiones mejoradas como **LSTM** y **GRU**, que solucionan problemas como el desvanecimiento del gradiente y permiten memorizar secuencias más largas.

---

## ⚙️ 3. Algoritmos de entrenamiento: backpropagation y optimizadores

El entrenamiento de una red neuronal consiste en ajustar sus parámetros (pesos y sesgos) para minimizar la diferencia entre la salida esperada y la obtenida.  
Para lograr esto, se utilizan dos elementos clave:

---

### 🔄 Backpropagation

Es un algoritmo que aplica la **regla de la cadena** del cálculo diferencial para distribuir el error de salida hacia las capas anteriores.  
Cada peso se actualiza en función del gradiente de la función de pérdida respecto a ese peso, permitiendo que la red aprenda patrones complejos [5].

> 💡 Su impacto fue tal que permitió pasar de redes simples a redes profundas con múltiples capas ocultas.

---

### ⚙️ Optimizadores

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

### 2. Diseño e implementación

#### 2.1 Arquitectura de la solución

* **Patrones de diseño**: ejemplo: Factory para capas, Strategy para optimizadores.
* **Estructura de carpetas (ejemplo)**:

  ```
  proyecto-final/
  ├── src/
  │   ├── layers/
  │   ├── optimizers/
  │   └── main.cpp
  ├── tests/
  └── docs/
  ```

#### 2.2 Manual de uso y casos de prueba

* **Cómo ejecutar**: `./build/neural_net_demo input.csv output.csv`
* **Casos de prueba**:

  * Test unitario de capa densa.
  * Test de función de activación ReLU.
  * Test de convergencia en dataset de ejemplo.

> *Personalizar rutas, comandos y casos reales.*

---

### 3. Ejecución

> **Demo de ejemplo**: Video/demo alojado en `docs/demo.mp4`.
> Pasos:
>
> 1. Preparar datos de entrenamiento (formato CSV).
> 2. Ejecutar comando de entrenamiento.
> 3. Evaluar resultados con script de validación.

---

### 4. Análisis del rendimiento

* **Métricas de ejemplo**:

  * Iteraciones: 1000 épocas.
  * Tiempo total de entrenamiento: 2m30s.
  * Precisión final: 92.5%.
* **Ventajas/Desventajas**:

  * * Código ligero y dependencias mínimas.
  * – Sin paralelización, rendimiento limitado.
* **Mejoras futuras**:

  * Uso de BLAS para multiplicaciones (Justificación).
  * Paralelizar entrenamiento por lotes (Justificación).

---

### 5. Trabajo en equipo

| Tarea                     | Miembro  | Rol                       |
| ------------------------- | -------- | ------------------------- |
| Investigación teórica     | Alumno A | Documentar bases teóricas |
| Diseño de la arquitectura | Alumno B | UML y esquemas de clases  |
| Implementación del modelo | Alumno C | Código C++ de la NN       |
| Pruebas y benchmarking    | Alumno D | Generación de métricas    |
| Documentación y demo      | Alumno E | Tutorial y video demo     |

> *Actualizar con tareas y nombres reales.*

---

### 6. Conclusiones

* **Logros**: Implementar NN desde cero, validar en dataset de ejemplo.
* **Evaluación**: Calidad y rendimiento adecuados para propósito académico.
* **Aprendizajes**: Profundización en backpropagation y optimización.
* **Recomendaciones**: Escalar a datasets más grandes y optimizar memoria.

---

### 7. Bibliografía

> *Actualizar con bibliografia utilizada, al menos 4 referencias bibliograficas y usando formato IEEE de referencias bibliograficas.*

---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
