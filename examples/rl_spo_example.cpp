#include <iostream>
#include "rl/environment.h"
#include "rl/simple_policy_optimization.h"
#include "nn/layers/dense_layer.h"

class MyGridWorld : public Environment {
public:
  MyGridWorld(size_t width = 5, size_t height = 5)
    : m_width(width), m_height(height), m_done(false), m_last_reward(0.0f) {
    reset();
  }

  bool step(size_t action) override {
    size_t x, y;
    state_to_pos(m_current_state, x, y);

    switch (action) {
      case 0: y = (y > 0) ? y - 1 : y; break;
      case 1: y = (y < m_height - 1) ? y + 1 : y; break;
      case 2: x = (x > 0) ? x - 1 : x; break;
      case 3: x = (x < m_width - 1) ? x + 1 : x; break;
      default: break;
    }

    m_current_state = pos_to_state(x, y);
    if (m_current_state == m_goal_state) {
      m_last_reward = 10.0f;
      m_done = true;
    } else m_last_reward = -0.1f;

    return m_done;
  }

  size_t reset() override {
    m_current_state = 0;
    m_goal_state = pos_to_state(m_width - 1, m_height - 1);
    m_done = false;
    m_last_reward = 0.0f;
    return m_current_state;
  }

  size_t get_state() const override { return m_current_state; }
  float get_reward() const override { return m_last_reward; }
  size_t get_states_num() const override { return m_width * m_height; }
  size_t get_actions_num() const override { return 4; }
  bool is_done() const override { return m_done; }

private:
  size_t pos_to_state(size_t x, size_t y) const { return y * m_width + x; }
  void state_to_pos(size_t state, size_t& x, size_t& y) const {
    y = state / m_width;
    x = state % m_width;
  }

  size_t m_width, m_height;
  size_t m_current_state, m_goal_state;
  float m_last_reward;
  bool m_done;
};

int main() {
  MyGridWorld env(5, 5);

  Model policy_model;
  policy_model.add_layer(new DenseLayer(env.get_states_num(), 32, ActivationType::RELU));
  policy_model.add_layer(new DenseLayer(32, env.get_actions_num(), ActivationType::NONE));

  rl::SimplePolicyOptimizationConfig config;
  config.learning_rate = 0.05f;
  config.discount_factor = 0.99f;
  config.max_episodes = 250;
  config.max_steps_per_episode = 100;
  config.normalize_returns = true;
  config.verbose = true;

  rl::SimplePolicyOptimization spo(policy_model, config);
  spo.train(env);

  std::cout << "\n=== Test after SPO training ===\n";
  for (int episode = 0; episode < 5; ++episode) {
    size_t state = env.reset();
    float total_reward = 0.0f;
    int steps = 0;
    while (!env.is_done() && steps < 100) {
      size_t action = spo.choose_action(state, env.get_states_num(), false);
      env.step(action);
      state = env.get_state();
      total_reward += env.get_reward();
      ++steps;
    }
    std::cout << "Test " << episode + 1 << " reward=" << total_reward << " steps=" << steps << "\n";
  }

  return 0;
}
