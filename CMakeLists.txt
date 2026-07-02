cmake_minimum_required(VERSION 3.15)
project(NeuralWrapper LANGUAGES CXX)

# -----------------------
# Global Settings
# -----------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

# -----------------------
# Source Files
# -----------------------
file(GLOB_RECURSE LIB_SRCS CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

# -----------------------
# Library Target
# -----------------------
if(EMSCRIPTEN)
  message(STATUS "Building for WebAssembly")

  # Common link options for both JS and MJS targets
  set(WASM_COMMON_LINK_OPTIONS
    "-O3"
    "-sWASM=1"
    "-sALLOW_MEMORY_GROWTH=1"
    "-sEXPORTED_FUNCTIONS=@${CMAKE_CURRENT_SOURCE_DIR}/exported_functions.json"
    "-sEXPORTED_RUNTIME_METHODS=['HEAPF32','HEAPU8','HEAP32','ccall','cwrap']"
    "-sNO_DISABLE_EXCEPTION_CATCHING"
    "--no-entry"
  )

  # 1. Legacy Target (CommonJS - nn.js)
  add_executable(nn ${LIB_SRCS})
  set_target_properties(nn PROPERTIES
    OUTPUT_NAME "nn"
    PREFIX ""
    SUFFIX ".js"
  )
  target_compile_options(nn PRIVATE -O3)
  target_link_options(nn PRIVATE
    ${WASM_COMMON_LINK_OPTIONS}
    "-sMODULARIZE=1"
    "-sEXPORT_NAME=createNNModule"
  )

  # 2. Modern Target (ES Module - nn.mjs)
  add_executable(nn_mjs ${LIB_SRCS})
  set_target_properties(nn_mjs PROPERTIES
    OUTPUT_NAME "nn"
    PREFIX ""
    SUFFIX ".mjs"
  )
  target_compile_options(nn_mjs PRIVATE -O3)
  target_link_options(nn_mjs PRIVATE
    ${WASM_COMMON_LINK_OPTIONS}
    "-sEXPORT_ES6=1"
    "-sUSE_ES6_IMPORT_META=1"
  )

else()
  message(STATUS "Building native library")

  if(MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  endif()

  add_library(nn SHARED ${LIB_SRCS})
  target_include_directories(nn PUBLIC ${PROJECT_SOURCE_DIR}/include)

  set_target_properties(nn PROPERTIES
    OUTPUT_NAME "nn"
    PREFIX ""
  )

  target_compile_options(nn PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
  )

  # -----------------------
  # Examples
  # -----------------------
  option(BUILD_EXAMPLES "Build example executables" ON)

  if(BUILD_EXAMPLES)
    file(GLOB EXAMPLES_SRC CONFIGURE_DEPENDS examples/*.cpp)
    foreach(EXAMPLE_SRC ${EXAMPLES_SRC})
      get_filename_component(EXAMPLE_NAME ${EXAMPLE_SRC} NAME_WE)
      add_executable(${EXAMPLE_NAME} ${EXAMPLE_SRC})
      target_link_libraries(${EXAMPLE_NAME} PRIVATE nn)
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

endif()