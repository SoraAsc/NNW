#include "rl/simple_policy_optimization.h"
#include "tensor.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>

namespace rl {

SimplePolicyOptimization::SimplePolicyOptimization(Model& policy_model, const SimplePolicyOptimizationConfig& config)
  : m_model(policy_model), m_config(config), m_rng(std::random_device{}()) {}

void SimplePolicyOptimization::train(Environment& env, EpisodeCallback callback) {
  size_t episode = 0;
  while (m_config.max_episodes == 0 || episode < m_config.max_episodes) {
    train_episode(env, callback);
    episode++;
  }
}

void SimplePolicyOptimization::train_episode(Environment& env, EpisodeCallback callback) {
  size_t state_count = env.get_states_num();
  size_t action_count = env.get_actions_num();

  std::vector<size_t> states;
  std::vector<size_t> actions;
  std::vector<float> rewards;

  size_t state = env.reset();
  float episode_reward = 0.0f;
  size_t step = 0;

  while (!env.is_done() && step < m_config.max_steps_per_episode) {
    states.push_back(state);
    size_t action = choose_action(state, state_count, true);
    actions.push_back(action);

    env.step(action);

    size_t next_state = env.get_state();
    float reward = env.get_reward();
    rewards.push_back(reward);
    episode_reward += reward;

    state = next_state;
    step++;
  }

  // Compute discounted returns
  std::vector<float> returns = compute_discounted_returns(rewards);
  if (m_config.normalize_returns) normalize(returns);

  // Update policy
  for (size_t t = 0; t < states.size(); ++t) {
    Tensor state_input = make_one_hot(states[t], state_count);
    Tensor logits = m_model.forward(state_input);
    std::vector<float> probs = action_probabilities(logits);

    Tensor grad({1, action_count});
    float advantage = returns[t];
    for (size_t a = 0; a < action_count; ++a) {
      grad.data()[a] = probs[a];
    }
    grad.data()[actions[t]] -= 1.0f;
    for (size_t a = 0; a < action_count; ++a) {
      grad.data()[a] *= advantage;
    }

    m_model.clear_state();
    m_model.backward(grad);
    m_model.update(m_config.learning_rate);
  }

  if (m_config.verbose) {
    std::printf("[SPO] Episode reward=%.3f steps=%zu\n", episode_reward, step);
  }
  if (callback) {
    callback(0, episode_reward);
  }
}

size_t SimplePolicyOptimization::choose_action(size_t state, size_t state_size, bool training) {
  if (m_model.layers().empty())
    throw std::runtime_error("Policy model must contain at least one layer");

  Tensor state_input = make_one_hot(state, state_size);
  Tensor logits = m_model.forward(state_input);
  std::vector<float> probs = action_probabilities(logits);

  if (!training) {
    return static_cast<size_t>(std::distance(probs.begin(), std::max_element(probs.begin(), probs.end())));
  }

  std::discrete_distribution<size_t> distribution(probs.begin(), probs.end());
  return distribution(m_rng);
}

Tensor SimplePolicyOptimization::make_one_hot(size_t state_index, size_t state_size) const {
  if (state_size == 0)
    throw std::invalid_argument("State size must be greater than zero");
  if (state_index >= state_size)
    throw std::out_of_range("State index out of range");

  Tensor input({1, state_size});
  input.zero();
  input.data()[state_index] = 1.0f;
  return input;
}

std::vector<float> SimplePolicyOptimization::compute_discounted_returns(const std::vector<float>& rewards) const {
  std::vector<float> returns(rewards.size());
  float running = 0.0f;
  for (int i = static_cast<int>(rewards.size()) - 1; i >= 0; --i) {
    running = rewards[i] + m_config.discount_factor * running;
    returns[i] = running;
  }
  return returns;
}

void SimplePolicyOptimization::normalize(std::vector<float>& values) const {
  if (values.empty()) return;
  float mean = std::accumulate(values.begin(), values.end(), 0.0f) / static_cast<float>(values.size());
  float variance = 0.0f;
  for (float v : values) variance += (v - mean) * (v - mean);
  variance /= static_cast<float>(values.size());
  float stddev = std::sqrt(variance) + 1e-8f;
  for (float& v : values) v = (v - mean) / stddev;
}

std::vector<float> SimplePolicyOptimization::action_probabilities(const Tensor& logits) const {
  size_t n = logits.numel();
  if (n == 0) return {};

  std::vector<float> exps(n);
  const float* data = logits.data();
  float max_logit = data[0];
  for (size_t i = 1; i < n; ++i) max_logit = std::max(max_logit, data[i]);

  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    exps[i] = std::exp(data[i] - max_logit);
    sum += exps[i];
  }
  if (sum <= 0.0f) sum = 1e-8f;
  for (size_t i = 0; i < n; ++i) exps[i] /= sum;
  return exps;
}

}