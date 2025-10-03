#pragma once
#include "nn/layers/layer.h"

class DenseLayer : public Layer {
public:
  DenseLayer(const size_t in_features, const size_t out_features);
  Tensor forward(const Tensor& input) override;
  Tensor backward(const Tensor& grad_output) override;
  void update(float learning_rate) override;
  std::string info() override;
  std::string detailed_info() override;
  ~DenseLayer();
private:
  Tensor weights; // [out_features, in_features]
  Tensor biases;  // [out_features]
  Tensor input_cache; // Cache input for backward pass

  Tensor grad_weights; // Gradient w.r.t. weights
  Tensor grad_biases;  // Gradient w.r.t. biases
};
