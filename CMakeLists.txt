cmake_minimum_required(VERSION 3.10)
project(neural_wrapper LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(neural_wrapper SHARED
    src/tensor.cpp
    src/nn/layers/dense_layer.cpp
    src/api.cpp
)

# Para exportar a DLL/.so com nome "neural_wrapper"
set_target_properties(neural_wrapper PROPERTIES
    OUTPUT_NAME "neural_wrapper"
    PREFIX ""
)