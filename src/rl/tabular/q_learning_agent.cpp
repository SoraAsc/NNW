#include "rl/tabular/q_learning_agent.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include "rl/tabular/policy/epsilon_greedy_policy.h"
#include "rl/tabular/policy/greedy_policy.h"
#include <cstdio>

QLearningAgent::QLearningAgent(size_t states_num,
                               size_t actions_num,
                               float learning_rate,
                               float discount_factor)
  : m_learning_rate(learning_rate),
    m_discount_factor(discount_factor),
    m_training(true) {
  
  m_q_table = std::make_unique<QTable>(states_num, actions_num);
  m_policy = std::make_unique<EpsilonGreedyPolicy>(0.1f);

  // Defaults for reward handling
  m_reward_clip_enabled = true;
  m_reward_clip_min = -1000.0f;
  m_reward_clip_max = 1000.0f;
  m_reward_normalize_enabled = false;
  m_reward_normalize_scale = 1.0f;
}

size_t QLearningAgent::choose_action(size_t state) const {
  const float* q_values = m_q_table->get_row(state);
  assert(q_values != nullptr && "QLearningAgent::choose_action() - invalid state");
  if (!q_values) return 0;

  const size_t action_count = m_q_table->get_actions_num();
  return m_policy->select_action(q_values, action_count, m_training);
}

void QLearningAgent::update(size_t state, size_t action, float reward, 
                            size_t next_state, bool done) {
  if (!m_training) return;

  if (m_reward_clip_enabled) {
    if (reward < m_reward_clip_min) reward = m_reward_clip_min;
    if (reward > m_reward_clip_max) reward = m_reward_clip_max;
  }

  if (m_reward_normalize_enabled && m_reward_normalize_scale != 0.0f)
    reward = reward / m_reward_normalize_scale;

  notify_step_reward(reward);

  // Q-Learning: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
  float current_q = m_q_table->get(state, action);
  float max_next_q = done ? 0.0f : get_max_qvalue(next_state);
  float new_q = current_q + m_learning_rate * (reward + m_discount_factor * max_next_q - current_q);

  m_q_table->set(state, action, new_q);

  if (done) notify_episode_end();
}

float QLearningAgent::get_max_qvalue(size_t state) const {
  const float* q_values = m_q_table->get_row(state);
  if (!q_values) {
    std::fprintf(stderr, "QLearningAgent::get_max_qvalue() - invalid state %zu\n", state);
    return 0.0f;
  }

  const size_t action_count = m_q_table->get_actions_num();
  float max_q = q_values[0];
  for (size_t a = 1; a < action_count; ++a) {
    max_q = std::max(max_q, q_values[a]);
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

int QLearningAgent::get_policy_type() const {
  if (!m_policy) return 0; // RL_POLICY_GREEDY
  if (dynamic_cast<const EpsilonGreedyPolicy*>(m_policy.get())) return 1; // RL_POLICY_EPSILON_GREEDY
  return 0;
}

// ---- Reward clipping / normalization ----
void QLearningAgent::set_reward_clip(bool enabled, float min_val, float max_val) {
  m_reward_clip_enabled = enabled;
  m_reward_clip_min = min_val;
  m_reward_clip_max = max_val;
}

void QLearningAgent::set_reward_normalization(bool enabled, float scale) {
  m_reward_normalize_enabled = enabled;
  m_reward_normalize_scale = (scale == 0.0f) ? 1.0f : scale;
}

void QLearningAgent::notify_step_reward(float reward) {
  m_telemetry.cumulative_reward += reward;
  ++m_telemetry.episode_steps;
  m_telemetry.in_episode = true;
}

void QLearningAgent::notify_episode_end() {
  if (!m_telemetry.in_episode) return; // already finalized or never started

  double last = m_telemetry.cumulative_reward;
  m_telemetry.episodes++;
  m_telemetry.total_reward_all_episodes += last;
  m_telemetry.average_reward = m_telemetry.total_reward_all_episodes / m_telemetry.episodes;
  // record episode length telemetry
  m_telemetry.last_episode_reward = last;
  m_telemetry.last_episode_length = m_telemetry.episode_steps;
  m_telemetry.total_steps_all_episodes += static_cast<double>(m_telemetry.last_episode_length);
  m_telemetry.average_episode_length = m_telemetry.total_steps_all_episodes / m_telemetry.episodes;

  // reset episode counters (do not auto-log; user may call getters)
  m_telemetry.cumulative_reward = 0.0;
  m_telemetry.episode_steps = 0;
  m_telemetry.in_episode = false;
}

double QLearningAgent::get_average_reward() const { return m_telemetry.average_reward; }
double QLearningAgent::get_last_episode_reward() const { return m_telemetry.last_episode_reward; }
size_t QLearningAgent::get_episode_count() const { return m_telemetry.episodes; }

size_t QLearningAgent::get_last_episode_length() const { return m_telemetry.last_episode_length; }
double QLearningAgent::get_average_episode_length() const { return m_telemetry.average_episode_length; }
