#include "rl/q-learning-agent.h"
#include <algorithm>
#include <cmath>
#include "rl/policy.h"
#include <cstdio>

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

  // Defaults for reward handling
  m_reward_clip_enabled = true;
  m_reward_clip_min = -1000.0f;
  m_reward_clip_max = 1000.0f;
  m_reward_normalize_enabled = false;
  m_reward_normalize_scale = 1.0f;
}

size_t QLearningAgent::choose_action(size_t state) {
  std::vector<float> q_values(m_actions_num);
  for (size_t a = 0; a < m_actions_num; ++a) q_values[a] = m_q_table->get(state, a);

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

  notify_step_reward(reward);

  if (m_reward_clip_enabled) {
    if (reward < m_reward_clip_min) reward = m_reward_clip_min;
    if (reward > m_reward_clip_max) reward = m_reward_clip_max;
  }

  if (m_reward_normalize_enabled && m_reward_normalize_scale != 0.0f)
    reward = reward / m_reward_normalize_scale;

  // Q-Learning: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
  float current_q = m_q_table->get(state, action);
  float max_next_q = done ? 0.0f : get_max_qvalue(next_state);
  float new_q = current_q + m_learning_rate * (reward + m_discount_factor * max_next_q - current_q);

  m_q_table->set(state, action, new_q);

  if (done) notify_episode_end();
}

float QLearningAgent::get_max_qvalue(size_t state) {
  float max_q = -std::numeric_limits<float>::infinity();

  if (!m_q_table->is_valid_index(state, 0)) {
    std::fprintf(stderr, "QLearningAgent::get_max_qvalue() - invalid state %zu\n", state);
    return 0.0f;
  }

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