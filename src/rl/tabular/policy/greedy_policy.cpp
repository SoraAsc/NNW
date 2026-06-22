#include "rl/tabular/policy/greedy_policy.h"

std::size_t GreedyPolicy::select_action(const float* q_values, std::size_t num_actions, bool training) {
  (void)training;
  std::size_t best_action = 0;
  float best_value = q_values[0];

  for (std::size_t index = 1; index < num_actions; ++index) {
    if (q_values[index] > best_value) {
      best_value = q_values[index];
      best_action = index;
    }
  }

  return best_action;
}
