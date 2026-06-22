#pragma once

#include <cstddef>

struct TabularRlTelemetry {
  double cumulative_reward = 0.0;
  std::size_t episode_steps = 0;
  std::size_t episodes = 0;
  double total_reward_all_episodes = 0.0;
  double average_reward = 0.0;
  double last_episode_reward = 0.0;
  bool in_episode = false;
  std::size_t last_episode_length = 0;
  double total_steps_all_episodes = 0.0;
  double average_episode_length = 0.0;
};
