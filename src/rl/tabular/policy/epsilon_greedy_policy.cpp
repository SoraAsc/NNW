#include "rl/tabular/policy/epsilon_greedy_policy.h"

#include <algorithm>
#include <cmath>
#include <vector>

static constexpr float EPS_TOL = 1e-6f;

EpsilonGreedyPolicy::EpsilonGreedyPolicy(float epsilon)
  : m_epsilon(epsilon) {}

std::size_t EpsilonGreedyPolicy::select_action(const float* q_values, std::size_t num_actions, bool training) {
  if (!training) {
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

  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  if (dist(m_rng) < m_epsilon) {
    std::uniform_int_distribution<std::size_t> action_dist(0, num_actions - 1);
    return action_dist(m_rng);
  }

  float best_value = q_values[0];
  for (std::size_t index = 1; index < num_actions; ++index) {
    if (q_values[index] > best_value) {
      best_value = q_values[index];
    }
  }

  std::vector<std::size_t> best_indices;
  best_indices.reserve(num_actions);
  for (std::size_t index = 0; index < num_actions; ++index) {
    if (std::fabs(q_values[index] - best_value) <= EPS_TOL) {
      best_indices.push_back(index);
    }
  }

  if (best_indices.empty()) {
    return 0;
  }

  std::uniform_int_distribution<std::size_t> tie_dist(0, best_indices.size() - 1);
  return best_indices[tie_dist(m_rng)];
}
