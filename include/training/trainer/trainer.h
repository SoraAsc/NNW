#pragma once
#include "nn/model.h"
#include "nn/loss/loss.h"
#include "training/optimizer/optimizer.h"
#include <random>

struct TrainerConfig {
  size_t epochs = 10;
  size_t batch_size = 32;
  bool shuffle = true;
};

class Trainer {
public:
  Trainer(Model* model, Loss* loss, Optimizer* optimizer, const TrainerConfig& config = TrainerConfig());
  ~Trainer();

  // Vectors of samples (each sample is a Tensor). They must be of the same length.
  void train(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets);

  // Train on a single batch, return the average loss for the batch
  float train_batch(const std::vector<Tensor>& batch_inputs, const std::vector<Tensor>& batch_targets);
private:
  Model* m_model;
  Loss* m_loss;
  Optimizer* m_optimizer;
  TrainerConfig m_config;
  std::mt19937 m_rng;

  Tensor stack_batch(const std::vector<Tensor>& samples, size_t start, size_t end);
};