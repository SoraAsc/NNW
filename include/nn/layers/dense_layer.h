#pragma once
#include "nn/layers/layer.h"  
#include "nn/activation/activation.h"

class DenseLayer : public Layer {
public:
  DenseLayer(const size_t in_features, const size_t out_features, ActivationType act_type = ActivationType::NONE);
  Tensor forward(const Tensor& input) override; // X: [batch_size, in_features] or [in_features] (1D)
  Tensor backward(const Tensor& grad_output) override; // compute gradients and return grad_input
  void update(float learning_rate) override; // apply update to weights and biases
  void zero_grad() override; // clear stored gradients
  std::vector<std::pair<Tensor*, Tensor*>> get_parameters() override; // weights/biases and their gradients
  std::string info() override;
  std::string detailed_info() override;
  ~DenseLayer();


private:
  Tensor weights; // [out_features, in_features]
  Tensor biases;  // [out_features]
  Tensor input_cache; // Cache input for backward pass
  Tensor output_cache; // Cache output after activation for backward pass
  
  Tensor grad_weights; // Gradient w.r.t. weights
  Tensor grad_biases;  // Gradient w.r.t. biases

  ActivationType activation_type;
  std::unique_ptr<Activation> activation;
};
