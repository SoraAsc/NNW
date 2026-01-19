#include "rl/policy.h"
#include <algorithm>
#include <cmath>

static constexpr float EPS_TOL = 1e-6f;

size_t GreedyPolicy::select_action(const float* q_values, size_t num_actions) {
  size_t best_action = 0;
  float best_value = q_values[0];
  
  for (size_t i = 1; i < num_actions; ++i) {
    if (q_values[i] > best_value) {
      best_value = q_values[i];
      best_action = i;
    }
  }
  
  return best_action;
}

EpsilonGreedyPolicy::EpsilonGreedyPolicy(float epsilon) 
  : m_epsilon(epsilon) {}

size_t EpsilonGreedyPolicy::select_action(const float* q_values, size_t num_actions) {
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  
  if (dist(m_rng) < m_epsilon) {
    std::uniform_int_distribution<size_t> action_dist(0, num_actions - 1);
    return action_dist(m_rng);
  }
  
  // Find best value
  float best_value = q_values[0];
  for (size_t i = 1; i < num_actions; ++i) {
    if (q_values[i] > best_value)
      best_value = q_values[i];
  }

  // Collect all indices within tolerance of best_value (tie-breaking)
  std::vector<size_t> best_indices;
  best_indices.reserve(num_actions);
  for (size_t i = 0; i < num_actions; ++i) {
    if (std::fabs(q_values[i] - best_value) <= EPS_TOL)
      best_indices.push_back(i);
  }

  if (best_indices.empty()) return 0;

  std::uniform_int_distribution<size_t> tie_dist(0, best_indices.size() - 1);
  return best_indices[tie_dist(m_rng)];
}
