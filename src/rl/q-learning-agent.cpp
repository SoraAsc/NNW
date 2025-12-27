#include "rl/q-learning-agent.h"
#include <algorithm>
#include <cmath>

QLearningAgent::QLearningAgent(size_t states_num,
                               size_t actions_num,
                               float learning_rate,
                               float discount_factor)
  : m_learning_rate(learning_rate),
    m_discount_factor(discount_factor),
    m_actions_num(actions_num) {
  
  m_q_table = std::make_unique<QTable>(states_num, actions_num);
  m_policy = std::make_unique<EpsilonGreedyPolicy>(0.1f);
}

size_t QLearningAgent::choose_action(size_t state) {
  std::vector<float> q_values(m_actions_num);
  for (size_t a = 0; a < m_actions_num; ++a)
    q_values[a] = m_q_table->get(state, a);
  
  return m_policy->select_action(q_values.data(), m_actions_num);
}

void QLearningAgent::update(size_t state, size_t action, float reward, 
                            size_t next_state, bool done) {
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
