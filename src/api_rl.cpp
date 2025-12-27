#include "api_rl.h"
#include "rl/q-table.h"
#include "rl/q-learning-agent.h"
#include "rl/policy.h"
#include <memory>
#include <cstring>

// Q-Table Wrapper

struct RL_QTable {
  std::unique_ptr<QTable> impl;
  
  RL_QTable(size_t states, size_t actions) 
    : impl(std::make_unique<QTable>(states, actions)) {}
};

RL_QTable* rl_qtable_create(size_t states_num, size_t actions_num) {
  return new RL_QTable(states_num, actions_num);
}

void rl_qtable_free(RL_QTable* qtable) {
  delete qtable;
}

float rl_qtable_get(RL_QTable* qtable, size_t state, size_t action) {
  return qtable->impl->get(state, action);
}

void rl_qtable_set(RL_QTable* qtable, size_t state, size_t action, float value) {
  qtable->impl->set(state, action, value);
}

size_t rl_qtable_get_states(const RL_QTable* qtable) {
  return qtable->impl->get_states_num();
}

size_t rl_qtable_get_actions(const RL_QTable* qtable) {
  return qtable->impl->get_actions_num();
}

// Policy Wrapper

struct RL_Policy {
  std::unique_ptr<Policy> impl;
};

RL_Policy* rl_policy_create(RL_PolicyType type, float epsilon) {
  RL_Policy* policy = new RL_Policy();
  
  if (type == RL_POLICY_GREEDY) {
    policy->impl = std::make_unique<GreedyPolicy>();
  } else {
    policy->impl = std::make_unique<EpsilonGreedyPolicy>(epsilon);
  }
  
  return policy;
}

void rl_policy_free(RL_Policy* policy) {
  delete policy;
}

size_t rl_policy_select_action(RL_Policy* policy, const float* q_values, size_t num_actions) {
  return policy->impl->select_action(q_values, num_actions);
}

void rl_policy_set_epsilon(RL_Policy* policy, float epsilon) {
  if (auto* eps_policy = dynamic_cast<EpsilonGreedyPolicy*>(policy->impl.get()))
    eps_policy->set_epsilon(epsilon);
}

float rl_policy_get_epsilon(RL_Policy* policy) {
  if (auto* eps_policy = dynamic_cast<EpsilonGreedyPolicy*>(policy->impl.get()))
    return eps_policy->get_epsilon();

  return 0.0f;
}

// Q-Learning Agent Wrapper

struct RL_Agent {
  std::unique_ptr<QLearningAgent> impl;
};

RL_Agent* rl_agent_create(size_t states_num, size_t actions_num,
                          float learning_rate, float discount_factor) {
  RL_Agent* agent = new RL_Agent();
  agent->impl = std::make_unique<QLearningAgent>(states_num, actions_num, 
                                                  learning_rate, discount_factor);
  return agent;
}

void rl_agent_free(RL_Agent* agent) {
  delete agent;
}

size_t rl_agent_choose_action(RL_Agent* agent, size_t state) {
  return agent->impl->choose_action(state);
}

void rl_agent_update(RL_Agent* agent, size_t state, size_t action, 
                     float reward, size_t next_state, int done) {
  agent->impl->update(state, action, reward, next_state, done != 0);
}

void rl_agent_set_policy(RL_Agent* agent, RL_PolicyType policy_type, float epsilon) {
  std::unique_ptr<Policy> policy;
  
  if (policy_type == RL_POLICY_GREEDY)
    policy = std::make_unique<GreedyPolicy>();
  else
    policy = std::make_unique<EpsilonGreedyPolicy>(epsilon);
  
  agent->impl->set_policy(std::move(policy));
}

float rl_agent_get_learning_rate(RL_Agent* agent) {
  return agent->impl->get_learning_rate();
}

void rl_agent_set_learning_rate(RL_Agent* agent, float lr) {
  agent->impl->set_learning_rate(lr);
}

float rl_agent_get_discount_factor(RL_Agent* agent) {
  return agent->impl->get_discount_factor();
}

void rl_agent_set_discount_factor(RL_Agent* agent, float gamma) {
  agent->impl->set_discount_factor(gamma);
}

RL_QTable* rl_agent_get_qtable(RL_Agent* agent) {
  QTable* original = agent->impl->get_qtable();
  RL_QTable* qtable = new RL_QTable(
    original->get_states_num(),
    original->get_actions_num()
  );
  
  const auto& data = original->get_data();
  for (size_t i = 0; i < data.size(); ++i) {
    size_t state = i / original->get_actions_num();
    size_t action = i % original->get_actions_num();
    qtable->impl->set(state, action, data[i]);
  }
  
  return qtable;
}
