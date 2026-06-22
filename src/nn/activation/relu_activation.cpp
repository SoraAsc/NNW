#include "nn/activation/relu_activation.h"

#include <stdexcept>

void ReLUActivation::forward(Tensor& data) {
  const size_t total_size = data.numel();
  float* values = data.data();
  for (size_t index = 0; index < total_size; ++index) {
    values[index] = (values[index] > 0.0f) ? values[index] : 0.0f;
  }
}

void ReLUActivation::backward(Tensor& grad, const Tensor& grad_output) {
  if (grad.shape() != grad_output.shape()) {
    throw std::invalid_argument("Shapes do not match in ReLU backward");
  }

  const size_t total_size = grad.numel();
  float* gradient = grad.data();
  const float* output = grad_output.data();
  for (size_t index = 0; index < total_size; ++index) {
    gradient[index] *= (output[index] > 0.0f) ? 1.0f : 0.0f;
  }
}

const char* ReLUActivation::name() const {
  return "ReLU";
}
