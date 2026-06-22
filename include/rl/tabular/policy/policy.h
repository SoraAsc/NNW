#pragma once

#include <cstddef>

class Policy {
public:
  virtual ~Policy() = default;

  virtual std::size_t select_action(const float* q_values, std::size_t num_actions, bool training = true) = 0;
};
