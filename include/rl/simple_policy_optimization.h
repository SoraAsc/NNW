#pragma once

#include "nn/model.h"
#include "rl/environment.h"
#include <functional>
#include <random>
#include <vector>

namespace rl {

struct SimplePolicyOptimizationConfig {
  float learning_rate = 0.01f;
  float discount_factor = 0.99f;
  size_t max_episodes = 200; // 0 = unlimited
  size_t max_steps_per_episode = 200;
  bool normalize_returns = true;
  bool use_reward_to_go = true;
  bool verbose = false;
};

class SimplePolicyOptimization {
public:
  using EpisodeCallback = std::function<void(size_t episode, float reward)>;

  SimplePolicyOptimization(Model& policy_model, const SimplePolicyOptimizationConfig& config = {});

  // Train the policy for a number of episodes on the provided environment.
  void train(Environment& env, EpisodeCallback callback = nullptr);

  // Train for a single episode, useful for external control.
  void train_episode(Environment& env, EpisodeCallback callback = nullptr);

  // Choose an action from the current policy. When training=false,
  // the policy uses greedy selection over action probabilities.
  size_t choose_action(size_t state, size_t state_size, bool training = true);

private:
  Tensor make_one_hot(size_t state_index, size_t state_size) const;
  std::vector<float> compute_discounted_returns(const std::vector<float>& rewards) const;
  void normalize(std::vector<float>& values) const;
  std::vector<float> action_probabilities(const Tensor& logits) const;

  Model& m_model;
  SimplePolicyOptimizationConfig m_config;
  std::mt19937 m_rng;
};

}
