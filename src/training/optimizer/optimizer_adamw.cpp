#include "training/optimizer/optimizer_adamw.h"
#include <cmath>
#include <stdexcept>

AdamW::AdamW(Model& model, float lr, float beta1, float beta2, float eps, float weight_decay)
  : m_model(model),
    m_learning_rate(lr),
    m_beta1(beta1),
    m_beta2(beta2),
    m_eps(eps),
    m_weight_decay(weight_decay),
    m_t(0)
{
  if (m_learning_rate <= 0.0f) throw std::invalid_argument("AdamW: lr must be > 0");
}

void AdamW::ensure_moments(Tensor* param) {
  if (m_moments.find(param) == m_moments.end())
    m_moments.emplace(param, AdamWMoments(param->shape()));
}

void AdamW::zero_grad() {
  m_model.clear_state();
}

void AdamW::step() {
  m_t += 1;
  const float one_minus_beta1 = 1.0f - m_beta1;
  const float one_minus_beta2 = 1.0f - m_beta2;
  // bias correction denominators
  const float bias_correction1 = 1.0f - std::pow(m_beta1, (double)m_t);
  const float bias_correction2 = 1.0f - std::pow(m_beta2, (double)m_t);

  for (auto* layer : m_model.layers()) {
    auto params = layer->get_parameters();
    for (auto &pg : params) {
      Tensor* p = pg.first;
      Tensor* g = pg.second;
      if (!p || !g) continue;
      size_t n = p->numel();
      if (n == 0) continue;
      if (g->numel() != n) continue;

      ensure_moments(p);
      AdamWMoments &mom = m_moments.at(p);

      float* p_data = p->data();
      float* g_data = g->data();
      float* m_data = mom.m.data();
      float* v_data = mom.v.data();

      for (size_t i = 0; i < n; ++i) {
        float grad = g_data[i];

        // update biased first and second moments
        m_data[i] = m_beta1 * m_data[i] + one_minus_beta1 * grad;
        v_data[i] = m_beta2 * v_data[i] + one_minus_beta2 * (grad * grad);

        // bias-corrected estimates
        float m_hat = m_data[i] / bias_correction1;
        float v_hat = v_data[i] / bias_correction2;

        float denom = std::sqrt(v_hat) + m_eps;

        // decoupled weight decay (AdamW)
        float wd_term = m_weight_decay * p_data[i];

        float update = (m_hat / denom) + wd_term;

        // parameter update
        p_data[i] -= m_learning_rate * update;
      }
    }
  }
}
  

