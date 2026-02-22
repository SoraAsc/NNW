#include "api_rl.h"
#include "rl/q-table.h"
#include "rl/q-learning-agent.h"
#include "rl/policy.h"
#include <memory>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <random>
#include <fstream>
#include <string>

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

void rl_set_agent_reward_clip(RL_Agent* agent, int enabled, float min_val, float max_val) {
  if (!agent) return;
  agent->impl->set_reward_clip(enabled != 0, min_val, max_val);
}

void rl_set_agent_reward_normalization(RL_Agent* agent, int enabled, float scale) {
  if (!agent) return;
  agent->impl->set_reward_normalization(enabled != 0, scale);
}

double rl_get_agent_average_reward(RL_Agent* agent) {
  if (!agent) return 0.0;
  return agent->impl->get_average_reward();
}

double rl_get_agent_last_reward(RL_Agent* agent) {
  if (!agent) return 0.0;
  return agent->impl->get_last_episode_reward();
}

size_t rl_get_agent_episode_count(RL_Agent* agent) {
  if (!agent) return 0;
  return agent->impl->get_episode_count();
}

size_t rl_get_agent_last_episode_length(RL_Agent* agent) {
  if (!agent) return 0;
  return agent->impl->get_last_episode_length();
}

double rl_get_agent_average_episode_length(RL_Agent* agent) {
  if (!agent) return 0.0;
  return agent->impl->get_average_episode_length();
}

void rl_notify_agent_episode_end(RL_Agent* agent) {
  if (!agent) return;
  agent->impl->notify_episode_end();
}

// Q-Table save/load
bool rl_qtable_save(RL_QTable* qtable, const char* path) {
  if (!qtable || !path) return false;
  return qtable->impl->save(std::string(path));
}

RL_QTable* rl_qtable_load(const char* path) {
  if (!path) return nullptr;
  // create dummy qtable and let load resize it
  RL_QTable* q = new RL_QTable(1, 1);
  if (!q->impl->load(std::string(path))) {
    delete q;
    return nullptr;
  }
  return q;
}

// Agent save/load (includes q-table and basic agent params)
bool rl_save_agent(RL_Agent* agent, const char* path) {
  if (!agent || !path) return false;
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) return false;

  const char magic[] = "RLAGENTV1";
  const size_t magic_len = std::strlen(magic);
  ofs.write(magic, magic_len);

  QTable* t = agent->impl->get_qtable();
  uint64_t s = static_cast<uint64_t>(t->get_states_num());
  uint64_t a = static_cast<uint64_t>(t->get_actions_num());
  ofs.write(reinterpret_cast<const char*>(&s), sizeof(s));
  ofs.write(reinterpret_cast<const char*>(&a), sizeof(a));

  float lr = agent->impl->get_learning_rate();
  float gamma = agent->impl->get_discount_factor();
  ofs.write(reinterpret_cast<const char*>(&lr), sizeof(lr));
  ofs.write(reinterpret_cast<const char*>(&gamma), sizeof(gamma));

  int training = agent->impl->is_training() ? 1 : 0;
  ofs.write(reinterpret_cast<const char*>(&training), sizeof(training));

  // save epsilon scheduler state from wrapper
  ofs.write(reinterpret_cast<const char*>(&agent->eps_start), sizeof(agent->eps_start));
  ofs.write(reinterpret_cast<const char*>(&agent->eps_min), sizeof(agent->eps_min));
  ofs.write(reinterpret_cast<const char*>(&agent->eps_rate), sizeof(agent->eps_rate));
  ofs.write(reinterpret_cast<const char*>(&agent->eps_type), sizeof(agent->eps_type));
  ofs.write(reinterpret_cast<const char*>(&agent->eps_per_step), sizeof(agent->eps_per_step));
  ofs.write(reinterpret_cast<const char*>(&agent->eps_steps), sizeof(agent->eps_steps));
  ofs.write(reinterpret_cast<const char*>(&agent->eps_episodes), sizeof(agent->eps_episodes));
  ofs.write(reinterpret_cast<const char*>(&agent->current_eps), sizeof(agent->current_eps));

  // save policy type
  int policy_type = agent->impl->get_policy_type();
  ofs.write(reinterpret_cast<const char*>(&policy_type), sizeof(policy_type));

  // save reward clip/normalization
  bool clip_enabled = agent->impl->get_reward_clip_enabled();
  float clip_min = agent->impl->get_reward_clip_min();
  float clip_max = agent->impl->get_reward_clip_max();
  bool norm_enabled = agent->impl->get_reward_normalize_enabled();
  float norm_scale = agent->impl->get_reward_normalize_scale();
  ofs.write(reinterpret_cast<const char*>(&clip_enabled), sizeof(clip_enabled));
  ofs.write(reinterpret_cast<const char*>(&clip_min), sizeof(clip_min));
  ofs.write(reinterpret_cast<const char*>(&clip_max), sizeof(clip_max));
  ofs.write(reinterpret_cast<const char*>(&norm_enabled), sizeof(norm_enabled));
  ofs.write(reinterpret_cast<const char*>(&norm_scale), sizeof(norm_scale));

  const auto& data = t->get_data();
  if (!data.empty()) ofs.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));

  return ofs.good();
}

RL_Agent* rl_load_agent(const char* path) {
  if (!path) return nullptr;
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) return nullptr;

  char magic[9] = {0};
  const char expected[] = "RLAGENTV1";
  const size_t expected_len = std::strlen(expected);
  std::string read_magic(expected_len, '\0');
  ifs.read(&read_magic[0], expected_len);
  if (read_magic != expected) return nullptr;

  uint64_t s = 0, a = 0;
  ifs.read(reinterpret_cast<char*>(&s), sizeof(s));
  ifs.read(reinterpret_cast<char*>(&a), sizeof(a));
  if (s == 0 || a == 0) return nullptr;

  float lr = 0.0f, gamma = 0.0f;
  ifs.read(reinterpret_cast<char*>(&lr), sizeof(lr));
  ifs.read(reinterpret_cast<char*>(&gamma), sizeof(gamma));

  int training = 1;
  ifs.read(reinterpret_cast<char*>(&training), sizeof(training));

  RL_Agent* agent = rl_create_agent(static_cast<size_t>(s), static_cast<size_t>(a), lr, gamma);
  if (!agent) return nullptr;

  agent->impl->set_training(training != 0);

  // read epsilon scheduler fields into wrapper
  ifs.read(reinterpret_cast<char*>(&agent->eps_start), sizeof(agent->eps_start));
  ifs.read(reinterpret_cast<char*>(&agent->eps_min), sizeof(agent->eps_min));
  ifs.read(reinterpret_cast<char*>(&agent->eps_rate), sizeof(agent->eps_rate));
  ifs.read(reinterpret_cast<char*>(&agent->eps_type), sizeof(agent->eps_type));
  ifs.read(reinterpret_cast<char*>(&agent->eps_per_step), sizeof(agent->eps_per_step));
  ifs.read(reinterpret_cast<char*>(&agent->eps_steps), sizeof(agent->eps_steps));
  ifs.read(reinterpret_cast<char*>(&agent->eps_episodes), sizeof(agent->eps_episodes));
  ifs.read(reinterpret_cast<char*>(&agent->current_eps), sizeof(agent->current_eps));

  // read policy type
  int policy_type = RL_POLICY_EPSILON_GREEDY;
  ifs.read(reinterpret_cast<char*>(&policy_type), sizeof(policy_type));

  // read reward clip/normalization
  bool clip_enabled = false;
  float clip_min = 0.0f, clip_max = 0.0f;
  bool norm_enabled = false;
  float norm_scale = 1.0f;
  ifs.read(reinterpret_cast<char*>(&clip_enabled), sizeof(clip_enabled));
  ifs.read(reinterpret_cast<char*>(&clip_min), sizeof(clip_min));
  ifs.read(reinterpret_cast<char*>(&clip_max), sizeof(clip_max));
  ifs.read(reinterpret_cast<char*>(&norm_enabled), sizeof(norm_enabled));
  ifs.read(reinterpret_cast<char*>(&norm_scale), sizeof(norm_scale));

  // apply policy and epsilon
  rl_set_agent_policy(agent, static_cast<RL_PolicyType>(policy_type), static_cast<float>(agent->current_eps));
  rl_set_agent_epsilon_decay(agent, agent->eps_start, agent->eps_min, agent->eps_rate, static_cast<EpsilonDecayType>(agent->eps_type), agent->eps_per_step ? 1 : 0);

  // apply reward clip/normalization
  rl_set_agent_reward_clip(agent, agent ? (clip_enabled ? 1 : 0) : 0, clip_min, clip_max);
  rl_set_agent_reward_normalization(agent, agent ? (norm_enabled ? 1 : 0) : 0, norm_scale);

  // read qtable data
  QTable* t = agent->impl->get_qtable();
  size_t total = t->get_states_num() * t->get_actions_num();
  std::vector<float> data(total);
  if (total > 0) {
    ifs.read(reinterpret_cast<char*>(data.data()), total * sizeof(float));
    if (!ifs) { rl_free_agent(agent); return nullptr; }
    for (size_t i = 0; i < total; ++i) {
      size_t state = i / t->get_actions_num();
      size_t action = i % t->get_actions_num();
      t->set(state, action, data[i]);
    }
  }

  return agent;
}
