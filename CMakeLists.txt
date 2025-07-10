cmake_minimum_required(VERSION 3.16)
project(diabetes_prediction_nn LANGUAGES CXX)

# Configuración del estándar C++
set(CMAKE_CXX_STANDARD 17)
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

# Archivos de cabecera
set(HEADER_FILES
    tensor.h
    nn_interfaces.h
    nn_dense.h
    nn_activation.h
    nn_loss.h
    nn_optimizer.h
    neural_network.h
    data_loader.h
    diabetes_network.h
    model_evaluation.h
)

# Archivos fuente
set(SOURCE_FILES
    main.cpp
)

# Archivos fuente para entrenamientos específicos
set(TRAIN_500_SOURCE
    train_500_epochs.cpp
)

set(TRAIN_1000_SOURCE
    train_1000_epochs.cpp
)

# Verificar que todos los archivos existen
foreach(header ${HEADER_FILES})
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${header}")
        message(FATAL_ERROR "Missing required header file: ${header}")
    endif()
endforeach()

foreach(source ${SOURCE_FILES})
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
        message(FATAL_ERROR "Missing required source file: ${source}")
    endif()
endforeach()

# Crear el ejecutable principal
add_executable(diabetes_predictor ${SOURCE_FILES} ${HEADER_FILES})

# Crear ejecutable para entrenamiento de 500 épocas
add_executable(train_500 ${TRAIN_500_SOURCE} ${HEADER_FILES})

# Crear ejecutable para entrenamiento de 1000 épocas
add_executable(train_1000 ${TRAIN_1000_SOURCE} ${HEADER_FILES})

# Configurar directorio de inclusión para todos los ejecutables
target_include_directories(diabetes_predictor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(train_500 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(train_1000 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Propiedades de los targets
set_target_properties(diabetes_predictor PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "diabetes_predictor"
)

set_target_properties(train_500 PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "train_500_epochs"
)

set_target_properties(train_1000 PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "train_1000_epochs"
)

# Verificar que el dataset existe
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/diabetes_prediction_dataset.csv")
    message(WARNING "Dataset file not found: diabetes_prediction_dataset.csv")
    message(WARNING "Make sure to place the dataset in the project root directory")
endif()

# Configuración específica para Windows
if(WIN32)
    # Configurar para que funcione en Windows
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_definitions(diabetes_predictor PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_options(diabetes_predictor PRIVATE /utf-8)
        
        target_compile_definitions(train_500 PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_options(train_500 PRIVATE /utf-8)
        
        target_compile_definitions(train_1000 PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_options(train_1000 PRIVATE /utf-8)
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

# Target para mostrar información
add_custom_target(info
    COMMAND ${CMAKE_COMMAND} -E echo "🔧 Información del proyecto:"
    COMMAND ${CMAKE_COMMAND} -E echo "   - Compilador: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
    COMMAND ${CMAKE_COMMAND} -E echo "   - C++ Standard: ${CMAKE_CXX_STANDARD}"
    COMMAND ${CMAKE_COMMAND} -E echo "   - Build Type: ${CMAKE_BUILD_TYPE}"
    COMMAND ${CMAKE_COMMAND} -E echo "   - Ejecutables disponibles:"
    COMMAND ${CMAKE_COMMAND} -E echo "     * diabetes_predictor (programa completo)"
    COMMAND ${CMAKE_COMMAND} -E echo "     * train_500_epochs (entrenamiento rápido - 500 épocas)"
    COMMAND ${CMAKE_COMMAND} -E echo "     * train_1000_epochs (entrenamiento medio - 1000 épocas)"
    COMMAND ${CMAKE_COMMAND} -E echo "   - Targets disponibles: run, run_500, run_1000, rebuild"
    VERBATIM
)

# Target para ejecutar el programa principal
add_custom_target(run
    COMMAND diabetes_predictor
    DEPENDS diabetes_predictor
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "🚀 Ejecutando diabetes_predictor..."
    VERBATIM
)

# Target para ejecutar entrenamiento de 500 épocas
add_custom_target(run_500
    COMMAND train_500_epochs
    DEPENDS train_500
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "⚡ Ejecutando entrenamiento rápido (500 épocas)..."
    VERBATIM
)

# Target para ejecutar entrenamiento de 1000 épocas
add_custom_target(run_1000
    COMMAND train_1000_epochs
    DEPENDS train_1000
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "🔥 Ejecutando entrenamiento medio (1000 épocas)..."
    VERBATIM
)

# Target para limpiar y rebuilder
add_custom_target(rebuild
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target diabetes_predictor
    COMMENT "🔄 Limpiando y recompilando..."
    VERBATIM
)

# Mostrar configuración
message(STATUS "🎯 Configuración del proyecto:")
message(STATUS "   - Proyecto: ${PROJECT_NAME}")
message(STATUS "   - Compilador: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "   - C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "   - Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "   - Directorio fuente: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "   - Directorio build: ${CMAKE_CURRENT_BINARY_DIR}")

# Configurar CLion para debugging
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "🐛 Modo Debug habilitado - CLion debugging disponible")
endif()

# Instalar (opcional)
install(TARGETS diabetes_predictor train_500 train_1000
    RUNTIME DESTINATION bin
    COMPONENT applications
) 