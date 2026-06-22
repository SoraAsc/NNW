#pragma once

#include "nn/activation/activation.h"

class ReLUActivation final : public Activation {
public:
  void forward(Tensor& data) override;
  void backward(Tensor& grad, const Tensor& grad_output) override;
  const char* name() const override;
};
