cmake_minimum_required(VERSION 3.15)
project(NeuralWrapper LANGUAGES CXX)

# -----------------------
# Global Settings
# -----------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

# -----------------------
# Neural Lib
# -----------------------
file(GLOB LIB_SRCS
  src/*.cpp
  src/nn/activation/*.cpp
  src/nn/layers/*.cpp
  src/nn/loss/*.cpp
  src/nn/*.cpp
  src/training/optimizer/*.cpp
  src/training/trainer/*.cpp
)

add_library(nn SHARED ${LIB_SRCS})
target_include_directories(nn PUBLIC ${PROJECT_SOURCE_DIR}/include)

set_target_properties(nn PROPERTIES
  OUTPUT_NAME "nn"
  PREFIX ""
)

# -----------------------
# Option to build examples
# -----------------------
option(BUILD_EXAMPLES "Build example executables" ON)

if(BUILD_EXAMPLES)
  file(GLOB EXAMPLES_SRC examples/*.cpp)
  foreach(EXAMPLE_SRC ${EXAMPLES_SRC})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SRC} NAME_WE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SRC})
    target_link_libraries(${EXAMPLE_NAME} nn)
  endforeach()
endif()

# -----------------------
# Installation rules
# -----------------------
install(TARGETS nn
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)