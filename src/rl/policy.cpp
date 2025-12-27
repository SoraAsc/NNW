#include "rl/policy.h"
#include <algorithm>

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
