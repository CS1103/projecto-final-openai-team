# 🚀 Instrucciones para Ejecutar el Proyecto en CLion

## 📋 **Requisitos Previos**

1. **CLion 2021.3 o superior**
2. **CMake 3.16 o superior** (incluido con CLion)
3. **Compilador C++17** compatible:
   - **Windows**: MinGW-w64, MSVC 2019+, o Clang
   - **Linux**: GCC 9+, Clang 10+
   - **macOS**: Xcode Command Line Tools

## 🔧 **Configuración Inicial en CLion**

### **Paso 1: Abrir el Proyecto**
```
1. Abrir CLion
2. File → Open → Seleccionar la carpeta del proyecto
3. CLion detectará automáticamente el CMakeLists.txt
4. Hacer clic en "Open as Project"
```

### **Paso 2: Configurar el Compilador (si es necesario)**
```
1. File → Settings (o CLion → Preferences en macOS)
2. Build, Execution, Deployment → Toolchains
3. Verificar que se detecte correctamente el compilador C++
4. Si no se detecta automáticamente:
   - En Windows: Configurar MinGW o MSVC
   - En Linux/macOS: Verificar que gcc/clang esté instalado
```

### **Paso 3: Configurar CMake**
```
1. File → Settings → Build, Execution, Deployment → CMake
2. Verificar que esté configurado el profile "Debug" y "Release"
3. Si es necesario, agregar flags adicionales en "CMake options":
   -DCMAKE_BUILD_TYPE=Debug (para modo debug)
   -DCMAKE_BUILD_TYPE=Release (para modo release)
```

## 🏗️ **Compilación del Proyecto**

### **Opción 1: Compilación Automática**
```
1. CLion compilará automáticamente al abrir el proyecto
2. Esperar a que aparezca "CMake generation finished" en la barra inferior
3. Si hay errores, aparecerán en la ventana "CMake" (inferior)
```

### **Opción 2: Compilación Manual**
```
1. Build → Rebuild Project (Ctrl+Shift+F9)
2. O hacer clic en el ícono 🔨 en la barra de herramientas
```

### **Verificar Compilación Exitosa**
```
✅ Sin errores en la ventana "CMake"
✅ Target visible en el panel "CMake" (lateral derecho):
   - diabetes_predictor (ejecutable principal)
```

## 🚀 **Ejecución del Proyecto**

### **Método 1: Ejecutar desde CLion**
```
1. Seleccionar "diabetes_predictor" en el dropdown de configuraciones (parte superior)
2. Hacer clic en el botón ▶️ Run (o presionar Shift+F10)
3. El programa se ejecutará en la consola integrada de CLion
```

### **Método 2: Ejecutar con Debugging**
```
1. Seleccionar "diabetes_predictor" en el dropdown
2. Hacer clic en el botón 🐛 Debug (o presionar Shift+F9)
3. Esto habilitará breakpoints y debugging completo
```

## 🔍 **Resolución de Problemas Comunes**

### **Error: "CMake generation failed"**
```
✅ Solución:
1. Tools → CMake → Reset Cache and Reload Project
2. Verificar que CMake esté instalado: cmake --version en terminal
3. Verificar que el compilador C++ esté disponible
```

### **Error: "No valid toolchain found"**
```
✅ Solución Windows:
1. Instalar MinGW-w64: https://www.mingw-w64.org/
2. O instalar Visual Studio Build Tools
3. Reiniciar CLion después de la instalación

✅ Solución Linux:
sudo apt install build-essential cmake

✅ Solución macOS:
xcode-select --install
```

### **Error: "Cannot find dataset file"**
```
✅ Solución:
1. Verificar que diabetes_prediction_dataset.csv esté en la carpeta raíz
2. Working directory en Run Configuration debe ser la carpeta del proyecto
3. Run → Edit Configurations → Verificar "Working directory"
```

### **Error de Includes o Headers no encontrados**
```
✅ Solución:
1. Build → Clean (limpiar proyecto)
2. Tools → CMake → Reset Cache and Reload Project
3. File → Invalidate Caches and Restart
```

### **Problemas con Templates o STL**
```
✅ Solución:
1. Verificar que C++17 esté habilitado en CMakeLists.txt
2. File → Settings → Editor → Code Style → C/C++ → Set Standard to "C++17"
3. Build → Rebuild Project
```

## 📊 **Configuraciones de Ejecución**

### **Configuración Principal: diabetes_predictor**
```
- Ejecutable: diabetes_predictor
- Working directory: [Project Root]
- Program arguments: (ninguno necesario)
- Environment variables: (ninguna necesaria)
```

## 🎯 **Funcionalidades de CLion Útiles**

### **Debugging Avanzado**
```
1. Colocar breakpoints haciendo clic en el margen izquierdo
2. Variables automáticamente visibles en modo debug
3. Step Over (F8), Step Into (F7), Step Out (Shift+F8)
```

### **Navegación de Código**
```
1. Ctrl+Click en función → Ir a definición
2. Ctrl+Shift+F → Buscar en todo el proyecto
3. Ctrl+N → Buscar archivos por nombre
```

### **Autocompletado Inteligente**
```
1. CLion sugiere automáticamente funciones y variables
2. Ctrl+Space → Forzar autocompletado
3. Alt+Enter → Quick fixes para errores
```

## 📈 **Resultados Esperados**

Al ejecutar exitosamente `diabetes_predictor`, deberías ver:

```
████████████████████████████████████████████████████████████████
██           PREDICCIÓN DE DIABETES CON REDES NEURONALES     ██
████████████████████████████████████████████████████████████████

📊 FASE 1: CARGA Y PREPROCESAMIENTO DE DATOS
Datos cargados: 100000 muestras, 9 características
...
🏆 RESULTADOS PRINCIPALES:
📊 Accuracy: 92.50%
🎯 Sensibilidad: 89.20%
⚡ Especificidad: 94.80%
⭐ F1-Score: 91.10%
```

## 🆘 **Soporte Adicional**

Si continúas teniendo problemas:

1. **Verificar logs detallados**:
   - View → Tool Windows → CMake
   - View → Tool Windows → Build

2. **Contactar soporte**:
   - Incluir versión de CLion: Help → About
   - Incluir versión de compilador
   - Incluir mensajes de error específicos

3. **Recursos útiles**:
   - Documentación CLion: https://www.jetbrains.com/help/clion/
   - CMake documentation: https://cmake.org/documentation/

---

¡Proyecto listo para ejecutarse en CLion! 🎉 