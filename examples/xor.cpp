#include <iostream>
#include <vector>
#include <memory>

#include "tensor.h"
#include "nn/layers/dense_layer.h"
#include "nn/model.h"
#include "nn/loss/loss_mse.h"
#include "training/optimizer/optimizer_sgd.h"
#include "training/optimizer/optimizer_adamw.h"
#include "training/trainer/trainer.h"

int main() {
  // Tiny network: 2 -> 2 (Tanh) -> 1 (Sigmoid)
  Model model;
  model.add_layer(new DenseLayer(2, 2, ActivationType::TANH));
  model.add_layer(new DenseLayer(2, 1, ActivationType::SIGMOID));

  // Loss and optimizer
  MSELoss loss;
  SGD opt(model, 0.5f);
  // AdamW opt(model);
  // Trainer config
  TrainerConfig cfg;
  cfg.epochs = 3000;
  cfg.batch_size = 4;
  cfg.shuffle = true;

  Trainer trainer(&model, &loss, &opt, cfg);

  std::vector<Tensor> X, Y;
  X.reserve(4); Y.reserve(4);

  std::vector<std::vector<float>> inputs = {
    {0.0f, 0.0f},
    {0.0f, 1.0f},
    {1.0f, 0.0f},
    {1.0f, 1.0f}
  };

  std::vector<float> targets = {0.0f, 1.0f, 1.0f, 0.0f};

  for (size_t i = 0; i < inputs.size(); ++i) {
    X.emplace_back(inputs[i], std::vector<size_t>{2}); // shape [2]
    Y.emplace_back(std::vector<float>{targets[i]}, std::vector<size_t>{1});
  }

  std::cout << "Starting training (XOR)\n";
  trainer.train(X, Y);

  std::cout << "\nFinal Predictions:\n";
  for (size_t i = 0; i < X.size(); ++i) {
    Tensor sample_batch({1, 2});
    for (size_t j = 0; j < 2; ++j) sample_batch.data()[j] = X[i].data()[j];
    Tensor out = model.forward(sample_batch);
    float p = out.data()[0];

    std::cout << "X = [" << X[i].data()[0] << ", " << X[i].data()[1]
              << "] -> " << p << " (rounded: " << (p > 0.5f ? 1 : 0) << ")\n";
  }

  return 0;
}
