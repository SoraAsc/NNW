#pragma once

#include <cstddef>

extern "C" {
  // Q-Table API  
  typedef struct RL_QTable RL_QTable;
  
  // Q-Table operations
  RL_QTable* rl_create_qtable(size_t states_num, size_t actions_num);
  void rl_free_qtable(RL_QTable* qtable);
  float rl_get_qtable(RL_QTable* qtable, size_t state, size_t action);
  void rl_set_qtable(RL_QTable* qtable, size_t state, size_t action, float value);
  size_t rl_get_qtable_states(const RL_QTable* qtable);
  size_t rl_get_qtable_actions(const RL_QTable* qtable);

  typedef enum { RL_POLICY_GREEDY = 0, RL_POLICY_EPSILON_GREEDY = 1 } RL_PolicyType;

  // Q-Learning Agent API  
  typedef struct RL_Agent RL_Agent;
  
  RL_Agent* rl_create_agent(size_t states_num, size_t actions_num, 
                            float learning_rate, float discount_factor);
  void rl_free_agent(RL_Agent* agent);
  
  // Choose action and update
  size_t rl_choose_agent_action(RL_Agent* agent, size_t state);
  void rl_update_agent(RL_Agent* agent, size_t state, size_t action, 
                       float reward, size_t next_state, int done);
  
  // Policy management
  void rl_set_agent_policy(RL_Agent* agent, RL_PolicyType policy_type, float epsilon);

  typedef enum { EPS_DECAY_LINEAR = 0, EPS_DECAY_EXPONENTIAL = 1 } EpsilonDecayType;

  // Epsilon decay / scheduler per-agent
  bool rl_set_agent_epsilon_decay(RL_Agent* agent, double start, double min, double rate, EpsilonDecayType type, int per_step);
  void rl_update_agent_epsilon_step(RL_Agent* agent);
  void rl_update_agent_epsilon_episode(RL_Agent* agent);
  double rl_get_agent_epsilon(RL_Agent* agent);
  void rl_reset_agent_epsilon(RL_Agent* agent);
  
  // Getters/Setters
  float rl_get_agent_learning_rate(RL_Agent* agent);
  void rl_set_agent_learning_rate(RL_Agent* agent, float lr);
  float rl_get_agent_discount_factor(RL_Agent* agent);
  void rl_set_agent_discount_factor(RL_Agent* agent, float gamma);
  
  // Access internal Q-table
  RL_QTable* rl_get_agent_qtable(RL_Agent* agent);
}
