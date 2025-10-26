#pragma once

#ifdef _WIN32
  #ifdef BUILDING_DLL
    #define API __declspec(dllexport)
  #else
    #define API __declspec(dllimport)
  #endif
#else
  #define API __attribute__((visibility("default")))
#endif

#include "training/trainer/trainer.h"
#include "nn/layers/dense_layer.h"

extern "C" {
  // Opaque handler
  typedef struct NN_Model NN_Model;
  typedef struct NN_Trainer NN_Trainer;

  // Public Type
  typedef enum { NN_ACT_LINEAR = 0, NN_ACT_RELU = 1, NN_ACT_SIGMOID = 2, NN_ACT_TANH = 3 } NN_Activation;
  typedef enum { NN_OPT_SGD = 0, NN_OPT_ADAMW = 1 } NN_Optimizer;
  typedef enum { NN_LOSS_MSE = 0 } NN_Loss;

  // Model
  API NN_Model* nn_create_model(size_t input_dim);
  API void nn_free_model(NN_Model* model);
  API void nn_add_dense(NN_Model* model, size_t units, NN_Activation act);

  API size_t nn_get_input_dim(const NN_Model* model);
  API size_t nn_get_output_dim(const  NN_Model* model);

  // Trainer
  API NN_Trainer* nn_create_trainer(NN_Model* model, NN_Optimizer opt, NN_Loss loss, const TrainerConfig* cfg);
  API void nn_free_trainer(NN_Trainer* trainer);

  API void nn_train_fit(NN_Trainer* trainer, const float* x, size_t n_samples, size_t x_dim, const float* y, size_t y_dim);
  API void nn_predict(const NN_Model* model, const float* x, size_t n_samples, size_t x_dim, float* out, size_t y_dim);
  
  // // Models Handle
  // API Model* create_model();
  // API float* predict_model(Model* model, Tensor* input);
  // API void free_model(Model* model);
  
  // // Layers Handleout_features
  // API void add_dense_layer(Model* model, size_t units, ActivationType act_type);
  
  // // Trainer Handle
  // API Trainer* create_trainer(Model* model, OptimizerType opt_type, LossType loss_type, TrainerConfig cfg);
  // API Model* train(Trainer* trainer, const std::vector<Tensor> &inputs, const std::vector<Tensor> &targets); 
  // API void free_trainer(Trainer* trainer);
  // // Agent Handle
}