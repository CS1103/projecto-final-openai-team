cmake_minimum_required(VERSION 3.16)
project(diabetes_prediction_nn LANGUAGES CXX)

# Configuración del estándar C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Configuración de compilador
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Wall -Wextra")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /W4 /DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /W4")
endif()

# Configurar tipo de build por defecto
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Archivos de cabecera con nuevas rutas
set(ALGEBRA_HEADERS
    include/utec/algebra/tensor.h
)

set(NN_HEADERS
    include/utec/nn/nn_interfaces.h
    include/utec/nn/nn_dense.h
    include/utec/nn/nn_activation.h
    include/utec/nn/nn_loss.h
    include/utec/nn/nn_optimizer.h
    include/utec/nn/neural_network.h
)

set(DIABETES_HEADERS
    include/utec/diabetes/data_loader.h
    include/utec/diabetes/diabetes_network.h
    include/utec/diabetes/model_evaluation.h
)

set(ALL_HEADERS ${ALGEBRA_HEADERS} ${NN_HEADERS} ${DIABETES_HEADERS})

# Archivos fuente con nuevas rutas
set(MAIN_SOURCE_FILES
    src/main.cpp
)

set(TRAIN_STABLE_SOURCE
    src/train_stable.cpp
)

# Verificar que todos los archivos existen
foreach(header ${ALL_HEADERS})
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${header}")
        message(FATAL_ERROR "Missing required header file: ${header}")
    endif()
endforeach()

foreach(source ${MAIN_SOURCE_FILES})
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
        message(FATAL_ERROR "Missing required source file: ${source}")
    endif()
endforeach()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${TRAIN_STABLE_SOURCE}")
    message(FATAL_ERROR "Missing required source file: ${TRAIN_STABLE_SOURCE}")
endif()

# Crear el ejecutable principal
add_executable(diabetes_predictor ${MAIN_SOURCE_FILES} ${ALL_HEADERS})

# Crear ejecutable para entrenamiento estable (anti-NaN)
add_executable(train_stable ${TRAIN_STABLE_SOURCE} ${ALL_HEADERS})

# Configurar directorio de inclusión para todos los ejecutables
# Usar include/ como directorio base para que los #include "utec/..." funcionen
target_include_directories(diabetes_predictor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(train_stable PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Propiedades de los targets
set_target_properties(diabetes_predictor PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "programa"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
)

set_target_properties(train_stable PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "stable_train"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
)

# Verificar que los datasets existen
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/diabetes_prediction_dataset.csv")
    message(WARNING "Dataset file not found: data/diabetes_prediction_dataset.csv")
    message(WARNING "Make sure to place the dataset in the data/ directory")
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/diabetes_prediction_dataset_balanced_8500.csv")
    message(WARNING "Balanced dataset file not found: data/diabetes_prediction_dataset_balanced_8500.csv")
    message(WARNING "Make sure to place the balanced dataset in the data/ directory")
endif()

# Configuración específica para Windows
if(WIN32)
    # Configurar para que funcione en Windows
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_definitions(diabetes_predictor PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_options(diabetes_predictor PRIVATE /utf-8)
        
        target_compile_definitions(train_stable PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_options(train_stable PRIVATE /utf-8)
    endif()
endif()

# Habilitar coloreado de sintaxis para todos los archivos
set_property(DIRECTORY PROPERTY
    VS_STARTUP_PROJECT diabetes_predictor
)

# Configurar working directory para ejecución
set_property(TARGET diabetes_predictor PROPERTY
    VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_property(TARGET train_stable PROPERTY
    VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Target para mostrar información
add_custom_target(info
    COMMAND ${CMAKE_COMMAND} -E echo "🔧 Información del proyecto:"
    COMMAND ${CMAKE_COMMAND} -E echo "   - Compilador: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
    COMMAND ${CMAKE_COMMAND} -E echo "   - C++ Standard: ${CMAKE_CXX_STANDARD}"
    COMMAND ${CMAKE_COMMAND} -E echo "   - Build Type: ${CMAKE_BUILD_TYPE}"
    COMMAND ${CMAKE_COMMAND} -E echo "   - Estructura: Nueva organización modular"
    COMMAND ${CMAKE_COMMAND} -E echo "   - Headers: ${CMAKE_CURRENT_LIST_LENGTH} archivos organizados por módulos"
    COMMAND ${CMAKE_COMMAND} -E echo "   - Ejecutables disponibles:"
    COMMAND ${CMAKE_COMMAND} -E echo "     * programa (ejecutable principal completo)"
    COMMAND ${CMAKE_COMMAND} -E echo "     * stable_train (entrenamiento estable - sin NaN)"
    COMMAND ${CMAKE_COMMAND} -E echo "   - Targets disponibles: run, run_stable, rebuild"
    VERBATIM
)

# Target para ejecutar el programa principal
add_custom_target(run
    COMMAND programa
    DEPENDS diabetes_predictor
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "🚀 Ejecutando programa principal..."
    VERBATIM
)

# Target para ejecutar entrenamiento estable (anti-NaN)
add_custom_target(run_stable
    COMMAND stable_train
    DEPENDS train_stable
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "🛡️ Ejecutando entrenamiento estable (sin NaN)..."
    VERBATIM
)

# Target para limpiar y rebuilder
add_custom_target(rebuild
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target diabetes_predictor
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target train_stable
    COMMENT "🔄 Limpiando y recompilando todos los ejecutables..."
    VERBATIM
)

# Target para compilación rápida (solo stable_train)
add_custom_target(quick
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target train_stable
    COMMENT "⚡ Compilación rápida de entrenamiento estable..."
    VERBATIM
)

# Target para verificar estructura
add_custom_target(check_structure
    COMMAND ${CMAKE_COMMAND} -E echo "📁 Verificando estructura del proyecto:"
    COMMAND ${CMAKE_COMMAND} -E echo "   ✅ Headers organizados en include/utec/"
    COMMAND ${CMAKE_COMMAND} -E echo "   ✅ Código fuente en src/"
    COMMAND ${CMAKE_COMMAND} -E echo "   ✅ Datos en data/"
    COMMAND ${CMAKE_COMMAND} -E echo "   ✅ Scripts en scripts/"
    COMMAND ${CMAKE_COMMAND} -E echo "   ✅ Ejecutables en bin/"
    COMMAND ${CMAKE_COMMAND} -E echo "   ✅ Documentación en docs/"
    VERBATIM
) 