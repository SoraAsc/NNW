#pragma once
#include "tensor.h"

class Loss {
public:
  virtual float forward(const Tensor& predicted, const Tensor& target) = 0;
  virtual Tensor backward(const Tensor& predicted, const Tensor& target) = 0;
  virtual const char* name() const = 0;
  virtual ~Loss() = default;
};