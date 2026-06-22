#pragma once

#include "policy.h"

class GreedyPolicy final : public Policy {
public:
  std::size_t select_action(const float* q_values, std::size_t num_actions, bool training = true) override;
};
