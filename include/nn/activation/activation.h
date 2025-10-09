#pragma once
#include "tensor.h"
#include <memory>

enum class ActivationType {
  NONE = 0,
  RELU,
  SIGMOID,
  TANH,
  SOFTMAX
};

class Activation {
public:
  virtual ~Activation() = default;
  virtual void forward(Tensor& data) = 0; // In-place operation
  virtual void backward(Tensor& grad, const Tensor& grad_output) = 0;   // modify grad in-place based on grad_output: grad := grad * f'(input)

  virtual const char* name() const = 0;
};

// Factory function to create activation by type
std::unique_ptr<Activation> create_activation(ActivationType t);