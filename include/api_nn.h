#pragma once

#include <cstddef>

#define API

#ifdef __cplusplus
extern "C" {
#endif
  // Opaque handlers
  typedef struct NN_Model NN_Model;
  typedef struct NN_Trainer NN_Trainer;

  // Public Type
  typedef enum { NN_ACT_LINEAR = 0, NN_ACT_RELU = 1, NN_ACT_SIGMOID = 2, NN_ACT_TANH = 3 } NN_Activation;
  typedef enum { RL_ACT_NONE = 0, RL_ACT_LINEAR = 1, RL_ACT_RELU = 2, RL_ACT_SIGMOID = 3, RL_ACT_TANH = 4 } RL_Activation;
  typedef enum { NN_OPT_SGD = 0, NN_OPT_ADAMW = 1 } NN_Optimizer;
  typedef enum { NN_LOSS_MSE = 0 } NN_Loss;

  // Trainer config (C struct)
  typedef struct {
    size_t epochs;
    size_t batch_size;
    int shuffle; // 0=false, non-zero=true
    float learning_rate;
  } NN_TrainerConfig; 

  // Model
  API NN_Model* nn_create_model(size_t input_dim);
  API void nn_free_model(NN_Model* model);
  API void nn_add_dense(NN_Model* model, size_t units, NN_Activation act);

  API size_t nn_get_input_dim(const NN_Model* model);
  API size_t nn_get_output_dim(const  NN_Model* model);

  API NN_Model* rl_model_create(size_t input_dim);
  API void rl_model_free(NN_Model* model);
  API void rl_model_add_dense(NN_Model* model, size_t input_dim, size_t units, RL_Activation act);

  API size_t rl_model_get_input_dim(const NN_Model* model);
  API size_t rl_model_get_output_dim(const NN_Model* model);
  API void* nn_model_get_internal(NN_Model* model);

  // Trainer
  API NN_Trainer* nn_create_trainer(NN_Model* model, NN_Optimizer opt, NN_Loss loss, const NN_TrainerConfig* cfg);
  API void nn_free_trainer(NN_Trainer* trainer);

  API void nn_train_fit(NN_Trainer* trainer, const float* x, size_t n_samples, size_t x_dim, const float* y, size_t y_dim);
  API void nn_predict(const NN_Model* model, const float* x, size_t n_samples, size_t x_dim, float* out, size_t y_dim);

#ifdef __cplusplus
}
#endif
