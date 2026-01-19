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

  // Training / evaluation mode: when in evaluation (training=false), update() is a no-op
  void set_training(bool training) { m_training = training; }
  bool is_training() const { return m_training; }
  
  // Getters
  float get_learning_rate() const { return m_learning_rate; }
  float get_discount_factor() const { return m_discount_factor; }
  QTable* get_qtable() { return m_q_table.get(); }

  // Reward clipping / normalization
  void set_reward_clip(bool enabled, float min_val, float max_val);
  void set_reward_normalization(bool enabled, float scale);
  // Telemetry / logging
  void notify_episode_end();
  double get_average_reward() const;
  double get_last_episode_reward() const;
  size_t get_episode_count() const;
  // Episode length telemetry
  size_t get_last_episode_length() const;
  double get_average_episode_length() const;
  
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
  bool m_training = true;
  // Reward handling
  bool m_reward_clip_enabled = false;
  float m_reward_clip_min = -1000.0f;
  float m_reward_clip_max = 1000.0f;
  bool m_reward_normalize_enabled = false;
  float m_reward_normalize_scale = 1.0f;

  struct Telemetry {
    double cumulative_reward = 0.0;
    size_t episode_steps = 0;
    size_t episodes = 0;
    double total_reward_all_episodes = 0.0;
    double average_reward = 0.0;
    double last_episode_reward = 0.0;
    bool in_episode = false;
    // episode length
    size_t last_episode_length = 0;
    double total_steps_all_episodes = 0.0;
    double average_episode_length = 0.0;
  } m_telemetry;
  
  float get_max_qvalue(size_t state);
  void notify_step_reward(float reward);
};
