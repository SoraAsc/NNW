#include "rl/q-learning-agent.h"
#include <algorithm>
#include <cmath>
#include "rl/policy.h"

QLearningAgent::QLearningAgent(size_t states_num,
                               size_t actions_num,
                               float learning_rate,
                               float discount_factor)
  : m_learning_rate(learning_rate),
    m_discount_factor(discount_factor),
    m_actions_num(actions_num),
    m_training(true) {
  
  m_q_table = std::make_unique<QTable>(states_num, actions_num);
  m_policy = std::make_unique<EpsilonGreedyPolicy>(0.1f);
}

size_t QLearningAgent::choose_action(size_t state) {
  std::vector<float> q_values(m_actions_num);
  for (size_t a = 0; a < m_actions_num; ++a)
    q_values[a] = m_q_table->get(state, a);

  if (!m_training) {
    float best_value = q_values[0];
    for (size_t i = 1; i < m_actions_num; ++i) best_value = std::max(best_value, q_values[i]);

    std::vector<size_t> best_indices;
    for (size_t i = 0; i < m_actions_num; ++i) {
      if (std::fabs(q_values[i] - best_value) <= 1e-6f) best_indices.push_back(i);
    }
    if (best_indices.empty()) return 0;
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, best_indices.size() - 1);
    return best_indices[dist(rng)];
  }

  return m_policy->select_action(q_values.data(), m_actions_num);
}

void QLearningAgent::update(size_t state, size_t action, float reward, 
                            size_t next_state, bool done) {
  if (!m_training) return;

  // Q-Learning: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
  float current_q = m_q_table->get(state, action);
  float max_next_q = done ? 0.0f : get_max_qvalue(next_state);
  float new_q = current_q + m_learning_rate * (reward + m_discount_factor * max_next_q - current_q);

  m_q_table->set(state, action, new_q);
}

float QLearningAgent::get_max_qvalue(size_t state) {
  float max_q = -std::numeric_limits<float>::infinity();
  
  for (size_t a = 0; a < m_actions_num; ++a) {
    float q_value = m_q_table->get(state, a);
    max_q = std::max(max_q, q_value);
  }
  
  return max_q;
}

void QLearningAgent::set_epsilon(float epsilon) {
  if (!m_policy) return;
  if (auto eps = dynamic_cast<EpsilonGreedyPolicy*>(m_policy.get()))
    eps->set_epsilon(epsilon);
}

float QLearningAgent::get_epsilon() const {
  if (!m_policy) return 0.0f;
  if (auto eps = dynamic_cast<const EpsilonGreedyPolicy*>(m_policy.get()))
    return eps->get_epsilon();
  return 0.0f;
}
