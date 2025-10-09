#pragma once
#include "training/optimizer/optimizer.h"

class SGD : public Optimizer {
public:
  SGD(Model& model, float learning_rate);
  void step() override;
  void zero_grad() override;

private:
  Model& m_model;
  float m_learning_rate;
};
