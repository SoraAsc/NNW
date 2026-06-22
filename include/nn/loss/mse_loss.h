#pragma once
#include "nn/loss/loss.h"

class MSELoss: public Loss {
public:
  float forward(const Tensor& predicted, const Tensor& target) override;
  Tensor backward(const Tensor& predicted, const Tensor& target) override;
  const char* name() const override { return "Mean Squared Error"; }
};