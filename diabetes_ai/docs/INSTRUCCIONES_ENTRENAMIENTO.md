# 🚀 Instrucciones de Compilación y Ejecución

## 📋 Programas Disponibles

Tienes **4 programas** diferentes para entrenar redes neuronales:

1. **`diabetes_predictor`** - Programa completo (2000 épocas, optimizado)
2. **`train_500_epochs`** - Entrenamiento rápido (500 épocas, simple)
3. **`train_1000_epochs`** - Entrenamiento medio (1000 épocas, simple)
4. **`train_stable`** - Entrenamiento estable (1500 épocas, ultra-estable)

## ⚙️ Configuraciones de Red

### 🏃‍♂️ Configuración Rápida (500 épocas)
- **Arquitectura**: 13 → 32 → 1 (simple)
- **Learning rate**: 0.01
- **Épocas**: 500
- **Batch size**: 64
- **Optimizador**: SGD
- **Tiempo estimado**: ~30-60 segundos

### 🚶‍♂️ Configuración Media (1000 épocas)
- **Arquitectura**: 13 → 32 → 1 (simple)
- **Learning rate**: 0.01
- **Épocas**: 1000
- **Batch size**: 64
- **Optimizador**: SGD
- **Tiempo estimado**: ~1-2 minutos

### 🛡️ Configuración Estable (1500 épocas)
- **Arquitectura**: 13 → 32 → 16 → 1 (ultra-estable)
- **Learning rate**: 0.0005 (conservador)
- **Épocas**: 1500
- **Batch size**: 64
- **Optimizador**: SGD (máxima estabilidad)
- **Tiempo estimado**: ~2-3 minutos
- **Características**:
  - ✅ Detección automática de inestabilidades
  - ✅ Manejo robusto de errores numéricos
  - ✅ Configuración conservadora
  - ✅ Ideal para máxima confiabilidad

## 🔨 Compilación

### Requisitos
- Compilador C++ con soporte para C++17
- CMake (opcional, pero recomendado)

### Método 1: Compilación con CMake (Recomendado)

```bash
# Configurar el proyecto
cmake -B build -S .

# Compilar todos los ejecutables
cmake --build build --config Release
```

### Método 2: Compilación de programas específicos
```bash
# Programa estable
cmake --build build --target train_stable

# Programa rápido
cmake --build build --target train_500

# Programa medio
cmake --build build --target train_1000

# Programa completo
cmake --build build --target diabetes_predictor
```

### Método 3: Compilación Manual

Si prefieres compilar manualmente:

```bash
# Programa estable (recomendado)
g++ -std=c++17 -O2 -Wall -Wextra -o train_stable train_stable.cpp

# Programa rápido
g++ -std=c++17 -O2 -Wall -Wextra -o train_500 train_500_epochs.cpp

# Programa medio
g++ -std=c++17 -O2 -Wall -Wextra -o train_1000 train_1000_epochs.cpp

# Programa completo
g++ -std=c++17 -O2 -Wall -Wextra -o programa main.cpp
```

## 🚀 Ejecución

### Ejecutar directamente

```bash
# Entrenamiento estable (máxima confiabilidad)
./build/train_stable

# Entrenamiento rápido (500 épocas)
./build/train_500_epochs

# Entrenamiento medio (1000 épocas)
./build/train_1000_epochs

# Programa completo (2000 épocas)
./build/diabetes_predictor
```

### Usar targets de CMake
```bash
# Ejecutar entrenamiento estable
cmake --build build --target run_stable

# Ejecutar entrenamiento rápido
cmake --build build --target run_500

# Ejecutar entrenamiento medio
cmake --build build --target run_1000

# Ejecutar programa completo
cmake --build build --target run
```

## 📁 Archivos Necesarios

Asegúrate de tener estos archivos en el directorio del proyecto:
- `diabetes_prediction_dataset_balanced_8500.csv` (dataset principal)
- Todos los archivos `.h` (headers)
- Los archivos `.cpp`:
  - `main.cpp`
  - `train_500_epochs.cpp`
  - `train_1000_epochs.cpp`
  - `train_stable.cpp`

## 🛠️ Comandos Útiles

### Ver información del proyecto
```bash
cmake --build build --target info
```

### Limpiar y recompilar
```bash
cmake --build build --target rebuild
```

### Compilar en modo Release (más rápido)
```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Compilar en modo Debug (para desarrollo)
```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

## 📊 Comparación de Rendimiento

| Programa | Épocas | Tiempo Aprox. | Precisión Esperada | Estabilidad | Uso Recomendado |
|----------|--------|---------------|-------------------|-------------|-----------------|
| `train_500_epochs` | 500 | 30-60 seg | ~75-80% | ⚡ Rápido | Pruebas rápidas |
| `train_1000_epochs` | 1000 | 1-2 min | ~80-85% | 🔄 Normal | Balance tiempo/calidad |
| `train_stable` | 1500 | 2-3 min | ~80-87% | 🛡️ **Ultra-estable** | Máxima confiabilidad |
| `diabetes_predictor` | 2000 | 3-5 min | ~85-90% | ⚖️ Avanzado | Máxima precisión |

## 🔧 Solución de Problemas

### Problemas de estabilidad numérica
- Usa `train_stable` para máxima confiabilidad
- Este programa maneja automáticamente inestabilidades numéricas

### Archivos no encontrados
- Verifica que el dataset CSV esté en el directorio de ejecución
- Confirma que todos los archivos `.h` estén presentes

### Problemas de compilación
- Verifica que tengas un compilador C++17 instalado
- Prueba la compilación manual si CMake presenta problemas

### Rendimiento lento
- Compila en modo `Release` para mejor rendimiento
- Usa programas más rápidos (500 o 1000 épocas) para pruebas

## 🎯 Recomendaciones de Uso

### Para máxima confiabilidad
- **Usa `train_stable`** como primera opción
- Este programa está optimizado para estabilidad y robustez
- Ideal cuando necesitas resultados consistentes

### Para casos específicos
1. **Desarrollo y pruebas**: `train_500_epochs` para iteración rápida
2. **Balance**: `train_1000_epochs` para buen compromiso tiempo/calidad
3. **Máxima precisión**: `diabetes_predictor` cuando el tiempo no es crítico

## 📈 Ejemplo de Ejecución

### Flujo de trabajo recomendado

```bash
# 1. Compilar todos los programas
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 2. Comenzar con entrenamiento estable
./build/train_stable

# 3. Comparar con otros programas si es necesario
./build/train_500_epochs
./build/train_1000_epochs
./build/diabetes_predictor
```

## 💡 Tips

- El programa estable usa configuración conservadora para máxima confiabilidad
- Los entrenamientos muestran progreso cada 50-100 iteraciones
- Los resultados pueden variar entre ejecuciones debido a inicialización aleatoria
- Para mejores resultados, ejecuta el programa varias veces y promedia
- Compila en modo Release para mejor rendimiento

## 🎛️ Configuraciones Avanzadas

### Modos de compilación
- **Release**: Optimizado para velocidad (`-O2`)
- **Debug**: Incluye información de depuración (`-g`)

### Personalización
- Todos los programas pueden modificarse editando los archivos `.cpp`
- Las configuraciones de red están en `diabetes_network.h`
- Los parámetros se pueden ajustar según necesidades específicas 