#include "nn/activation/tanh_activation.h"

#include <cmath>
#include <stdexcept>

void TanhActivation::forward(Tensor& data) {
  const size_t total_size = data.numel();
  float* values = data.data();
  for (size_t index = 0; index < total_size; ++index) {
    values[index] = std::tanh(values[index]);
  }
}

void TanhActivation::backward(Tensor& grad, const Tensor& grad_output) {
  if (grad.shape() != grad_output.shape()) {
    throw std::invalid_argument("Shapes do not match in Tanh backward");
  }

  const size_t total_size = grad.numel();
  float* gradient = grad.data();
  const float* output = grad_output.data();
  for (size_t index = 0; index < total_size; ++index) {
    const float value = output[index];
    gradient[index] *= (1.0f - value * value);
  }
}

const char* TanhActivation::name() const {
  return "Tanh";
}
