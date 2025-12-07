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
  
  size_t chooseAction(size_t state);
  
  void update(size_t state, size_t action, float reward, size_t next_state, bool done = false);
  
  // Getters
  float getLearningRate() const { return m_learning_rate; }
  float getDiscountFactor() const { return m_discount_factor; }
  QTable* getQTable() { return m_q_table.get(); }
  
  // Setters
  void setLearningRate(float lr) { m_learning_rate = lr; }
  void setDiscountFactor(float gamma) { m_discount_factor = gamma; }
  void setPolicy(std::unique_ptr<Policy> policy) { m_policy = std::move(policy); }

private:
  std::unique_ptr<QTable> m_q_table;
  std::unique_ptr<Policy> m_policy;
  float m_learning_rate;
  float m_discount_factor;
  size_t m_actions_num;
  
  float getMaxQValue(size_t state);
};
