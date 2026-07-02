#include "api_nn.h"

#include "nn/layers/dense_layer.h"
#include "nn/loss/mse_loss.h"
#include "nn/model.h"
#include "nn/tensor.h"
#include "nn/training/optimizer/adamw_optimizer.h"
#include "nn/training/optimizer/sgd_optimizer.h"
#include "nn/training/trainer.h"

#include <cstring>
#include <vector>

// Types
struct NN_Model { Model impl; size_t input_dim = 0; size_t output_dim = 0; };
struct NN_Trainer { Trainer* impl = nullptr; ~NN_Trainer() { delete impl; } };

// Model
NN_Model* nn_create_model(size_t input_dim) {
  NN_Model* model = new NN_Model();
  model->input_dim = input_dim;
  return model;
}

void nn_free_model(NN_Model* model) { delete model; }

void nn_add_dense(NN_Model* model, size_t units, NN_Activation act) {
  ActivationType a = ActivationType::NONE;
  switch (act)
  {
    case NN_ACT_LINEAR: a = ActivationType::NONE; break;
    case NN_ACT_RELU: a = ActivationType::RELU; break;  
    case NN_ACT_TANH: a = ActivationType::TANH; break;
    case NN_ACT_SIGMOID: a = ActivationType::SIGMOID; break;
  }
  size_t in = model->output_dim;
  if(model->impl.layers().empty()) in = model->input_dim;
  model->impl.add_layer(new DenseLayer(in, units, a));
  model->output_dim = units;
}

size_t nn_get_input_dim(const NN_Model* model) { return model->input_dim; }
size_t nn_get_output_dim(const  NN_Model* model) { return model->output_dim; }

NN_Model* rl_model_create(size_t input_dim) {
  return nn_create_model(input_dim);
}

void rl_model_free(NN_Model* model) { nn_free_model(model); }

void rl_model_add_dense(NN_Model* model, size_t input_dim, size_t units, RL_Activation act) {
  NN_Activation a = NN_ACT_LINEAR;
  switch (act)
  {
    case RL_ACT_NONE:
    case RL_ACT_LINEAR: a = NN_ACT_LINEAR; break;
    case RL_ACT_RELU: a = NN_ACT_RELU; break;
    case RL_ACT_TANH: a = NN_ACT_TANH; break;
    case RL_ACT_SIGMOID: a = NN_ACT_SIGMOID; break;
  }
  (void)input_dim;
  nn_add_dense(model, units, a);
}

size_t rl_model_get_input_dim(const NN_Model* model) { return nn_get_input_dim(model); }
size_t rl_model_get_output_dim(const NN_Model* model) { return nn_get_output_dim(model); }
void* nn_model_get_internal(NN_Model* model) { return model ? &model->impl : nullptr; }

// Trainer
NN_Trainer* nn_create_trainer(NN_Model* model, NN_Optimizer opt, NN_Loss loss, const NN_TrainerConfig* cfg) {
  const float learning_rate = cfg ? cfg->learning_rate : 1e-3f;
  Optimizer* optimizer = nullptr;
  switch (opt)
  {
    case NN_OPT_ADAMW: optimizer = new AdamW(model->impl, learning_rate); break;
    default: case NN_OPT_SGD: optimizer = new SGD(model->impl, learning_rate); break;
  }
  Loss* lossfn = nullptr;
  switch (loss)
  {
    default: lossfn = new MSELoss(); break;
  }

  TrainerConfig tcfg;
  if (cfg) {
    tcfg.epochs = cfg->epochs;
    tcfg.batch_size = cfg->batch_size;
    tcfg.shuffle = cfg->shuffle != 0;
  }

  NN_Trainer* trainer = new NN_Trainer();
  trainer->impl = new Trainer(&model->impl, lossfn, optimizer, tcfg);
  return trainer;
}

void nn_free_trainer(NN_Trainer* trainer) { delete trainer; }

// Helper
static Tensor make_tensor_from_row(const float* base, size_t stride_elems, size_t row, size_t dim) {
  Tensor t({dim});
  std::memcpy(t.data(), base + row*stride_elems, sizeof(float)*dim);
  return t;
}

void nn_train_fit(NN_Trainer* trainer, const float* x, size_t n_samples, size_t x_dim, const float* y, size_t y_dim) {
  std::vector<Tensor> vin; vin.reserve(n_samples);
  std::vector<Tensor> vtar; vtar.reserve(n_samples);

  for(size_t i = 0; i < n_samples; ++i) {
    vin.emplace_back(make_tensor_from_row(x, x_dim, i, x_dim));
    vtar.emplace_back(make_tensor_from_row(y, y_dim, i, y_dim));
  }

  trainer->impl->train(vin, vtar);
}

void nn_predict(const NN_Model* model, const float* x, size_t n_samples, size_t x_dim, float* out, size_t y_dim) {
  Tensor batch_input({n_samples, x_dim});
  std::memcpy(batch_input.data(), x, sizeof(float) * n_samples * x_dim);
  
  Tensor batch_output = const_cast<Model&>(model->impl).forward(batch_input);
  std::memcpy(out, batch_output.data(), sizeof(float) * n_samples * y_dim);
}
