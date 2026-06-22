#include "nn/activation/activation.h"
#include "nn/activation/relu_activation.h"
#include "nn/activation/sigmoid_activation.h"
#include "nn/activation/tanh_activation.h"
#include <stdexcept>

std::unique_ptr<Activation> create_activation(ActivationType t) {
  switch(t) {
    case ActivationType::RELU:
      return std::make_unique<ReLUActivation>();
    case ActivationType::SIGMOID:
      return std::make_unique<SigmoidActivation>();
    case ActivationType::TANH:
      return std::make_unique<TanhActivation>();
    case ActivationType::SOFTMAX:
      throw std::runtime_error("Softmax activation not implemented yet");
    case ActivationType::NONE:
      return nullptr;
    default:
      throw std::invalid_argument("Unknown activation type");
  }
}
