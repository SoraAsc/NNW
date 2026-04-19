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
# Source Files
# -----------------------
file(GLOB LIB_SRCS
  src/*.cpp
  src/nn/activation/*.cpp
  src/nn/layers/*.cpp
  src/nn/loss/*.cpp
  src/nn/*.cpp
  src/training/optimizer/*.cpp
  src/training/trainer/*.cpp
  src/rl/*.cpp
)

# -----------------------
# Library Target
# -----------------------
if(EMSCRIPTEN)
  message(STATUS "Building for WebAssembly")
  
  add_executable(nn ${LIB_SRCS})

  set_target_properties(nn PROPERTIES
    OUTPUT_NAME "nn"
    PREFIX ""
    SUFFIX ".js"
  )

  target_link_options(nn PRIVATE
    "-O3"
    "-sWASM=1"
    "-sMODULARIZE=1"
    "-sALLOW_MEMORY_GROWTH=1"
    "-sEXPORT_NAME=createNNModule"
    "-sEXPORTED_FUNCTIONS=_nn_create_model,_nn_free_model,_nn_add_dense,_nn_get_input_dim,_nn_get_output_dim,_nn_create_trainer,_nn_free_trainer,_nn_train_fit,_nn_predict,_rl_create_qtable,_rl_free_qtable,_rl_get_qtable,_rl_set_qtable,_rl_get_qtable_states,_rl_get_qtable_actions,_rl_create_agent,_rl_free_agent,_rl_choose_agent_action,_rl_update_agent,_rl_set_agent_policy,_rl_set_agent_epsilon_decay,_rl_update_agent_epsilon_step,_rl_update_agent_epsilon_episode,_rl_get_agent_epsilon,_rl_reset_agent_epsilon,_rl_set_agent_training,_rl_get_agent_training,_rl_get_agent_learning_rate,_rl_set_agent_learning_rate,_rl_get_agent_discount_factor,_rl_set_agent_discount_factor,_rl_get_agent_qtable,_rl_set_agent_reward_clip,_rl_set_agent_reward_normalization,_rl_get_agent_average_reward,_rl_get_agent_last_reward,_rl_get_agent_episode_count,_rl_get_agent_last_episode_length,_rl_get_agent_average_episode_length,_rl_notify_agent_episode_end,_rl_qtable_save,_rl_qtable_load,_rl_save_agent,_rl_load_agent"
    "--no-entry"
  )
else()
  message(STATUS "Building native library")

  add_library(nn SHARED ${LIB_SRCS})
  target_include_directories(nn PUBLIC ${PROJECT_SOURCE_DIR}/include)

  set_target_properties(nn PROPERTIES
    OUTPUT_NAME "nn"
    PREFIX ""
  )
endif()

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