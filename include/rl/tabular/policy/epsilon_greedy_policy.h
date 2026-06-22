#pragma once

#include "policy.h"

#include <random>

class EpsilonGreedyPolicy final : public Policy {
public:
  explicit EpsilonGreedyPolicy(float epsilon = 0.1f);

  std::size_t select_action(const float* q_values, std::size_t num_actions, bool training = true) override;

  void set_epsilon(float epsilon) { m_epsilon = epsilon; }
  float get_epsilon() const { return m_epsilon; }

private:
  float m_epsilon;
  std::mt19937 m_rng{std::random_device{}()};
};
