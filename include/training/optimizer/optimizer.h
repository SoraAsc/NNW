#pragma once
#include "nn/model.h"

class Optimizer {
public:
  virtual void step() = 0;
  virtual void zero_grad() = 0;
  virtual ~Optimizer() = default;
};