#include "api.h"
#include <cstddef>
#include "nn/loss/loss_mse.h"
#include "training/optimizer/optimizer_adamw.h"
#include "training/optimizer/optimizer_sgd.h"
#include <iostream>
#include <cstring>

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

// Trainer
NN_Trainer* nn_create_trainer(NN_Model* model, NN_Optimizer opt, NN_Loss loss, const TrainerConfig* cfg) {
  Optimizer* optimizer = nullptr;
  switch (opt)
  {
    case NN_OPT_ADAMW: optimizer = new AdamW(model->impl); break;
    default: case NN_OPT_SGD: optimizer = new SGD(model->impl, 0.5f); break;
  }
  Loss* lossfn = nullptr;
  switch (loss)
  {
    default: lossfn = new MSELoss(); break;
  }

  TrainerConfig tcfg;
  tcfg.epochs = cfg ? cfg->epochs : 3000;
  tcfg.batch_size = cfg ? cfg->batch_size : 4;
  tcfg.shuffle = cfg ? cfg->shuffle : true;

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

static Tensor make_tensor_from_row2d(const float* base, size_t stride_elems, size_t row, size_t dim) {
  Tensor t({1, dim});
  std::memcpy(t.data(), base + row*stride_elems, sizeof(float)*dim);
  return t;
}


void nn_train_fit(NN_Trainer* trainer, const float* x, size_t n_samples, size_t x_dim, const float* y, size_t y_dim) {
  const Model* model = trainer->impl->get_model();
  std::vector<Tensor> vin; vin.reserve(n_samples);
  std::vector<Tensor> vtar; vtar.reserve(n_samples);

  for(size_t i = 0; i < n_samples; ++i) {
    vin.emplace_back(make_tensor_from_row(x, x_dim, i, x_dim));
    vtar.emplace_back(make_tensor_from_row(y, y_dim, i, y_dim));
  }

  trainer->impl->train(vin, vtar);
}

void nn_predict(const NN_Model* model, const float* x, size_t n_samples, size_t x_dim, float* out, size_t y_dim) {
  for(size_t i = 0; i < n_samples; ++i) {
    Tensor in = make_tensor_from_row2d(x, x_dim, i, x_dim);
    Tensor pred = const_cast<Model&>(model->impl).forward(in);
    std::memcpy(out + i * y_dim, pred.data(), sizeof(float) * y_dim);
  }
}

// Model* create_model() {
//   return new Model();
// }

// float* predict_model(Model* model, Tensor* input) {
//   return model->forward(*input).data();
// }

// void free_model(Model* model) {
//   delete model;
// }

// void add_dense_layer(Model* model, size_t units, ActivationType act_type) {
//   if(model->layers().empty()) model->add_layer(new DenseLayer(units, units, act_type));
//   else model->add_layer(new DenseLayer(model->layers().back()->get_output_size(), units, act_type));
// }

// Trainer* create_trainer(Model* model, OptimizerType opt_type, LossType loss_type, TrainerConfig cfg) {
//   if (!model) return nullptr;

//   try {
//     Optimizer* optimizer = nullptr;
//     Loss* loss = nullptr;

//     // --- Optimizer ---
//     switch (opt_type) {
//       case OptimizerType::ADAMW:
//         optimizer = new AdamW(*model);
//         break;
//       case OptimizerType::SGD:
//         optimizer = new SGD(*model, 0.5f);
//         break;
//       default:
//         throw std::runtime_error("Unsupported optimizer type");
//     }

//     // --- Loss Function ---
//     switch (loss_type) {
//       case LossType::MSE:
//         loss = new MSELoss();
//         break;
//       default:
//         throw std::runtime_error("Unsupported loss type");
//     }

//     return new Trainer(model, loss, optimizer, cfg);
//   } catch (const std::exception& e) {
//     std::cerr << "[API] Error creating trainer: " << e.what() << std::endl;
//     return nullptr;
//   }
// }

// Model* train(Trainer* trainer, const std::vector<Tensor> &inputs, const std::vector<Tensor> &targets) {
//   trainer->train(inputs, targets);
//   return trainer->get_model();
// } 

// void free_trainer(Trainer* trainer) {
//   delete trainer;
// }