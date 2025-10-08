#pragma once
#include "layers/layer.h"

class Model {
public:
  Model() = default;
  ~Model();

  void add_layer(Layer* layer);
  Tensor forward(const Tensor& input); // Forward through all layers (each layer sequentially)
  Tensor backward(const Tensor& grad_output); // Backprop through all layers in reverse order
  void update(float learning_rate); // Apply update to all layers
  void clear_state();

  std::vector<Layer*>& layers();
private:
  std::vector<Layer*> m_layers;
};