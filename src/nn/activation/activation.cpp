#include "nn/activation/activation.h"
#include <stdexcept>
#include <cmath>

struct ReLUActivation : public Activation {
  void forward(Tensor& data) override {
    size_t total_size = data.numel();
    float* d = data.data();
    for (size_t i = 0; i < total_size; ++i) d[i] = (d[i] > 0.0f) ? d[i] : 0.0f;
  }

  void backward(Tensor& grad, const Tensor& grad_output) override {
    if (grad.shape() != grad_output.shape())
      throw std::invalid_argument("Shapes do not match in ReLU backward");
    
    size_t total_size = grad.numel();
    float* g = grad.data();
    const float* go = grad_output.data();
    for (size_t i = 0; i < total_size; ++i) g[i] *= (go[i] > 0) ? 1.0f : 0.0f;
  }

  const char* name() const override { return "ReLU"; }
};

struct SigmoidActivation : public Activation {
  void forward(Tensor& data) override {
    size_t total_size = data.numel();
    float* d = data.data();
    for (size_t i = 0; i < total_size; ++i) d[i] = 1.0f / (1.0f + std::exp(-d[i]));
  }

  void backward(Tensor& grad, const Tensor& grad_output) override {
    if (grad.shape() != grad_output.shape())
      throw std::invalid_argument("Shapes do not match in Sigmoid backward");
    
    size_t total_size = grad.numel();
    float* g = grad.data();
    const float* go = grad_output.data();
    for (size_t i = 0; i < total_size; ++i) {  
      float o = go[i];
      g[i] *= o * (1.0f - o);
    }
  }

  const char* name() const override { return "Sigmoid"; }
};

struct TanhActivation : public Activation {
  void forward(Tensor& data) override {
    size_t total_size = data.numel();
    float* d = data.data();
    for (size_t i = 0; i < total_size; ++i) d[i] = std::tanh(d[i]);
  }

  void backward(Tensor& grad, const Tensor& grad_output) override {
    if (grad.shape() != grad_output.shape())
      throw std::invalid_argument("Shapes do not match in Tanh backward");
    
    size_t total_size = grad.numel();
    float* g = grad.data();
    const float* go = grad_output.data();
    for (size_t i = 0; i < total_size; ++i) {  
      float o = go[i];
      g[i] *= (1.0f - o * o);
    }
  }

  const char* name() const override { return "Tanh"; }
};

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