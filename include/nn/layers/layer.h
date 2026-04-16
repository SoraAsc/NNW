#pragma once
#include "tensor.h"
#include <string>

class Layer {
public:
  virtual Tensor forward(const Tensor& input) = 0;
  virtual Tensor backward(const Tensor& grad_output) = 0;
  virtual void update(float learning_rate) = 0;
  virtual void zero_grad() = 0;
  virtual std::vector<std::pair<Tensor*, Tensor*>> get_parameters() = 0;
  virtual std::string info() = 0;
  virtual std::string detailed_info() = 0;
  virtual ~Layer() = default;

};

