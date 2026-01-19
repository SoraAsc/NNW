#include "api_rl.h"
#include "rl/q-table.h"
#include "rl/q-learning-agent.h"
#include "rl/policy.h"
#include <memory>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <random>

// Helper to clamp values
template<typename T>
static T clamp_val(T v, T lo, T hi) { return std::min(hi, std::max(lo, v)); }

// Q-Table Wrapper

struct RL_QTable {
  std::unique_ptr<QTable> impl;
  
  RL_QTable(size_t states, size_t actions) 
    : impl(std::make_unique<QTable>(states, actions)) {}
};

RL_QTable* rl_create_qtable(size_t states_num, size_t actions_num) {
  return new RL_QTable(states_num, actions_num);
}

void rl_free_qtable(RL_QTable* qtable) {
  delete qtable;
}

float rl_get_qtable(RL_QTable* qtable, size_t state, size_t action) {
  return qtable->impl->get(state, action);
}

void rl_set_qtable(RL_QTable* qtable, size_t state, size_t action, float value) {
  qtable->impl->set(state, action, value);
}

size_t rl_get_qtable_states(const RL_QTable* qtable) {
  return qtable->impl->get_states_num();
}

size_t rl_get_qtable_actions(const RL_QTable* qtable) {
  return qtable->impl->get_actions_num();
}

// Q-Learning Agent Wrapper

struct RL_Agent {
  std::unique_ptr<QLearningAgent> impl;

  // Epsilon scheduler state (per-agent)
  double eps_start = 0.1;
  double eps_min = 0.0;
  double eps_rate = 0.0;
  int eps_type = EPS_DECAY_LINEAR;
  bool eps_per_step = true;
  uint64_t eps_steps = 0;
  uint64_t eps_episodes = 0;
  double current_eps = 0.1;
};

RL_Agent* rl_create_agent(size_t states_num, size_t actions_num,
                          float learning_rate, float discount_factor) {
  RL_Agent* agent = new RL_Agent();
  agent->impl = std::make_unique<QLearningAgent>(states_num, actions_num, 
                                                  learning_rate, discount_factor);
  return agent;
}

void rl_free_agent(RL_Agent* agent) {
  delete agent;
}

size_t rl_choose_agent_action(RL_Agent* agent, size_t state) {
  return agent->impl->choose_action(state);
}

void rl_update_agent(RL_Agent* agent, size_t state, size_t action, 
                     float reward, size_t next_state, int done) {
  agent->impl->update(state, action, reward, next_state, done != 0);
}

void rl_set_agent_policy(RL_Agent* agent, RL_PolicyType policy_type, float epsilon) {
  std::unique_ptr<Policy> policy;
  
  if (policy_type == RL_POLICY_GREEDY)
    policy = std::make_unique<GreedyPolicy>();
  else
    policy = std::make_unique<EpsilonGreedyPolicy>(epsilon);
  
  agent->impl->set_policy(std::move(policy));
  // update wrapper epsilon state
  agent->current_eps = clamp_val<double>(epsilon, 0.0, 1.0);
}

bool rl_set_agent_epsilon_decay(RL_Agent* agent, double start, double min, double rate, EpsilonDecayType type, int per_step) {
  if (!agent) return false;
  start = clamp_val<double>(start, 0.0, 1.0);
  min = clamp_val<double>(min, 0.0, 1.0);
  if (min > start) std::swap(min, start);
  if (rate < 0.0) rate = 0.0;

  agent->eps_start = start;
  agent->eps_min = min;
  agent->eps_rate = rate;
  agent->eps_type = static_cast<int>(type);
  agent->eps_per_step = (per_step != 0);
  agent->eps_steps = 0;
  agent->eps_episodes = 0;
  agent->current_eps = start;
  // push to policy if available
  agent->impl->set_epsilon(static_cast<float>(agent->current_eps));
  return true;
}

static double compute_epsilon(double start, double min, double rate, int type, uint64_t t) {
  if (type == EPS_DECAY_LINEAR) {
    double v = start - rate * static_cast<double>(t);
    if (v < min) v = min;
    return v;
  } else {
    double v = min + (start - min) * std::exp(-rate * static_cast<double>(t));
    if (v < min) v = min;
    return v;
  }
}

void rl_update_agent_epsilon_step(RL_Agent* agent) {
  if (!agent) return;
  if (!agent->eps_per_step) return;
  agent->eps_steps++;
  double eps = compute_epsilon(agent->eps_start, agent->eps_min, agent->eps_rate, agent->eps_type, agent->eps_steps);
  agent->current_eps = clamp_val<double>(eps, 0.0, 1.0);
  agent->impl->set_epsilon(static_cast<float>(agent->current_eps));
}

void rl_update_agent_epsilon_episode(RL_Agent* agent) {
  if (!agent) return;
  if (agent->eps_per_step) return;
  agent->eps_episodes++;
  double eps = compute_epsilon(agent->eps_start, agent->eps_min, agent->eps_rate, agent->eps_type, agent->eps_episodes);
  agent->current_eps = clamp_val<double>(eps, 0.0, 1.0);
  agent->impl->set_epsilon(static_cast<float>(agent->current_eps));
}

double rl_get_agent_epsilon(RL_Agent* agent) {
  if (!agent) return 0.0;
  return agent->current_eps;
}

void rl_reset_agent_epsilon(RL_Agent* agent) {
  if (!agent) return;
  agent->eps_steps = 0;
  agent->eps_episodes = 0;
  agent->current_eps = agent->eps_start;
  agent->impl->set_epsilon(static_cast<float>(agent->current_eps));
}

void rl_set_agent_training(RL_Agent* agent, int training) {
  if (!agent) return;
  agent->impl->set_training(training != 0);
}

int rl_get_agent_training(RL_Agent* agent) {
  if (!agent) return 0;
  return agent->impl->is_training() ? 1 : 0;
}

float rl_get_agent_learning_rate(RL_Agent* agent) {
  return agent->impl->get_learning_rate();
}

void rl_set_agent_learning_rate(RL_Agent* agent, float lr) {
  agent->impl->set_learning_rate(lr);
}

float rl_get_agent_discount_factor(RL_Agent* agent) {
  return agent->impl->get_discount_factor();
}

void rl_set_agent_discount_factor(RL_Agent* agent, float gamma) {
  agent->impl->set_discount_factor(gamma);
}

RL_QTable* rl_get_agent_qtable(RL_Agent* agent) {
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
