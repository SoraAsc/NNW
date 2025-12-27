#pragma once

#include <cstddef>

extern "C" {
  // Q-Table API  
  typedef struct RL_QTable RL_QTable;
  
  // Q-Table operations
  RL_QTable* rl_qtable_create(size_t states_num, size_t actions_num);
  void rl_qtable_free(RL_QTable* qtable);
  float rl_qtable_get(RL_QTable* qtable, size_t state, size_t action);
  void rl_qtable_set(RL_QTable* qtable, size_t state, size_t action, float value);
  size_t rl_qtable_get_states(const RL_QTable* qtable);
  size_t rl_qtable_get_actions(const RL_QTable* qtable);

  // Policy API  
  typedef struct RL_Policy RL_Policy;
  typedef enum { RL_POLICY_GREEDY = 0, RL_POLICY_EPSILON_GREEDY = 1 } RL_PolicyType;
  
  RL_Policy* rl_policy_create(RL_PolicyType type, float epsilon);
  void rl_policy_free(RL_Policy* policy);
  size_t rl_policy_select_action(RL_Policy* policy, const float* q_values, size_t num_actions);
  void rl_policy_set_epsilon(RL_Policy* policy, float epsilon);
  float rl_policy_get_epsilon(RL_Policy* policy);

  // Q-Learning Agent API  
  typedef struct RL_Agent RL_Agent;
  
  RL_Agent* rl_agent_create(size_t states_num, size_t actions_num, 
                            float learning_rate, float discount_factor);
  void rl_agent_free(RL_Agent* agent);
  
  // Choose action and update
  size_t rl_agent_choose_action(RL_Agent* agent, size_t state);
  void rl_agent_update(RL_Agent* agent, size_t state, size_t action, 
                       float reward, size_t next_state, int done);
  
  // Policy management
  void rl_agent_set_policy(RL_Agent* agent, RL_PolicyType policy_type, float epsilon);
  
  // Getters/Setters
  float rl_agent_get_learning_rate(RL_Agent* agent);
  void rl_agent_set_learning_rate(RL_Agent* agent, float lr);
  float rl_agent_get_discount_factor(RL_Agent* agent);
  void rl_agent_set_discount_factor(RL_Agent* agent, float gamma);
  
  // Access internal Q-table
  RL_QTable* rl_agent_get_qtable(RL_Agent* agent);
}
