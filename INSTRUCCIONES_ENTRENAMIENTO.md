# 🚀 Instrucciones de Compilación y Ejecución

## 📋 Programas Disponibles

Ahora tienes **3 programas** diferentes para entrenar redes neuronales:

1. **`diabetes_predictor`** - Programa completo (2000 épocas, optimizado)
2. **`train_500_epochs`** - Entrenamiento rápido (500 épocas, simple)
3. **`train_1000_epochs`** - Entrenamiento medio (1000 épocas, simple)

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

## 🔨 Compilación

### Opción 1: Compilar todos los programas
```bash
# Configurar el proyecto
cmake -B build -S .

# Compilar todos los ejecutables
cmake --build build --config Release

# O compilar en modo Debug
cmake --build build --config Debug
```

### Opción 2: Compilar programas específicos
```bash
# Solo el programa de 500 épocas
cmake --build build --target train_500

# Solo el programa de 1000 épocas
cmake --build build --target train_1000

# Programa completo original
cmake --build build --target diabetes_predictor
```

## 🚀 Ejecución

### Opción 1: Ejecutar directamente
```bash
# Ejecutar entrenamiento rápido (500 épocas)
./build/train_500_epochs

# Ejecutar entrenamiento medio (1000 épocas)
./build/train_1000_epochs

# Ejecutar programa completo (2000 épocas)
./build/diabetes_predictor
```

### Opción 2: Usar targets de CMake
```bash
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
- Los archivos `.cpp` que acabamos de crear

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

| Programa | Épocas | Tiempo Aprox. | Precisión Esperada | Uso Recomendado |
|----------|--------|---------------|-------------------|-----------------|
| `train_500_epochs` | 500 | 30-60 seg | ~75-80% | Pruebas rápidas |
| `train_1000_epochs` | 1000 | 1-2 min | ~80-85% | Balance tiempo/calidad |
| `diabetes_predictor` | 2000 | 3-5 min | ~85-90% | Máxima precisión |

## 🔧 Solución de Problemas

### Error: No encuentra el dataset
```bash
❌ ERROR: No se pudo abrir el archivo: diabetes_prediction_dataset_balanced_8500.csv
```
**Solución**: Asegúrate de que el archivo CSV esté en el directorio donde ejecutas el programa.

### Error de compilación
```bash
❌ Missing required header file: [archivo].h
```
**Solución**: Verifica que todos los archivos `.h` estén en el directorio del proyecto.

### El programa se ejecuta muy lento
- Compila en modo `Release` para mejor rendimiento
- Los programas de 500 y 1000 épocas son más rápidos que el completo

## 🎯 Recomendaciones de Uso

1. **Para desarrollo**: Usa `train_500_epochs` para pruebas rápidas
2. **Para demostración**: Usa `train_1000_epochs` como balance
3. **Para producción**: Usa `diabetes_predictor` para máxima calidad

## 📈 Ejemplo de Ejecución

```bash
# 1. Compilar todo
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 2. Ejecutar entrenamiento rápido
./build/train_500_epochs

# 3. Comparar con entrenamiento medio
./build/train_1000_epochs

# 4. Ver la diferencia en precisión y tiempo
```

## 💡 Tips

- Los entrenamientos mostrarán progreso cada 50 (500 épocas) o 100 (1000 épocas) iteraciones
- Todos usan la misma arquitectura simple para comparación justa
- Los resultados pueden variar ligeramente entre ejecuciones debido a la inicialización aleatoria
- Para mejores resultados, ejecuta el programa varias veces y promedia los resultados 