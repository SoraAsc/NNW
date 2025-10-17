#pragma once
#include "training/optimizer/optimizer.h"
#include <unordered_map>

struct AdamWMoments {
  Tensor m;
  Tensor v;
  AdamWMoments() = default;
  AdamWMoments(const std::vector<size_t>& shape) : m(shape), v(shape) {
    m.zero();
    v.zero();
  }
};

class AdamW : public Optimizer {
public:
  AdamW(Model& model, float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 1e-2f);
  void step() override;
  void zero_grad() override;

private:
  Model& m_model;
  float m_learning_rate;
  float m_beta1;
  float m_beta2;
  float m_eps;
  float m_weight_decay;
  size_t m_t; // Steps counter
  struct Moments { Tensor m; Tensor v; };

  std::unordered_map<Tensor*, AdamWMoments> m_moments;
  void ensure_moments(Tensor* param);
};
