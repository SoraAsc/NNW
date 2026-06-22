#include "nn/activation/sigmoid_activation.h"

#include <cmath>
#include <stdexcept>

void SigmoidActivation::forward(Tensor& data) {
  const size_t total_size = data.numel();
  float* values = data.data();
  for (size_t index = 0; index < total_size; ++index) {
    values[index] = 1.0f / (1.0f + std::exp(-values[index]));
  }
}

void SigmoidActivation::backward(Tensor& grad, const Tensor& grad_output) {
  if (grad.shape() != grad_output.shape()) {
    throw std::invalid_argument("Shapes do not match in Sigmoid backward");
  }

  const size_t total_size = grad.numel();
  float* gradient = grad.data();
  const float* output = grad_output.data();
  for (size_t index = 0; index < total_size; ++index) {
    const float value = output[index];
    gradient[index] *= value * (1.0f - value);
  }
}

const char* SigmoidActivation::name() const {
  return "Sigmoid";
}
