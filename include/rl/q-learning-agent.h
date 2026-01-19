#pragma once
#include "rl/q-table.h"
#include "rl/policy.h"
#include <memory>

class QLearningAgent {
public:
  QLearningAgent(size_t states_num, 
                 size_t actions_num,
                 float learning_rate = 0.1f,
                 float discount_factor = 0.99f);
  
  ~QLearningAgent() = default;
  
  size_t choose_action(size_t state);
  
  void update(size_t state, size_t action, float reward, size_t next_state, bool done = false);
  
  // Getters
  float get_learning_rate() const { return m_learning_rate; }
  float get_discount_factor() const { return m_discount_factor; }
  QTable* get_qtable() { return m_q_table.get(); }
  
  // Setters
  void set_learning_rate(float lr) { m_learning_rate = lr; }
  void set_discount_factor(float gamma) { m_discount_factor = gamma; }
  void set_policy(std::unique_ptr<Policy> policy) { m_policy = std::move(policy); }

  // Epsilon helpers - effective only if current policy is EpsilonGreedyPolicy
  void set_epsilon(float epsilon);
  float get_epsilon() const;

private:
  std::unique_ptr<QTable> m_q_table;
  std::unique_ptr<Policy> m_policy;
  float m_learning_rate;
  float m_discount_factor;
  size_t m_actions_num;
  
  float get_max_qvalue(size_t state);
};
