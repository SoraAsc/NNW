#include <iostream>
#include <vector>
#include "rl/environment.h"
#include "api_rl.h"

class MyGridWorld : public Environment {
public:
  MyGridWorld(size_t width = 5, size_t height = 5)
    : m_width(width), m_height(height), m_done(false), m_last_reward(0.0f) {
    reset();
  }
  
  bool step(size_t action) override {
    size_t x, y;
    stateToPos(m_current_state, x, y);
    
    // Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    switch (action) {
      case 0: y = (y > 0) ? y - 1 : y; break;
      case 1: y = (y < m_height - 1) ? y + 1 : y; break;
      case 2: x = (x > 0) ? x - 1 : x; break;
      case 3: x = (x < m_width - 1) ? x + 1 : x; break;
    }
    
    m_current_state = posToState(x, y);
    
    if (m_current_state == m_goal_state) {
      m_last_reward = 10.0f;
      m_done = true;
    } else m_last_reward = -0.1f;
    
    return m_done;
  }
  
  size_t reset() override {
    m_current_state = 0;
    m_goal_state = posToState(m_width - 1, m_height - 1);
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
  size_t m_width, m_height;
  size_t m_current_state, m_goal_state;
  float m_last_reward;
  bool m_done;
  
  size_t posToState(size_t x, size_t y) const { return y * m_width + x; }
  void stateToPos(size_t state, size_t& x, size_t& y) const {
    y = state / m_width;
    x = state % m_width;
  }
};

int main() {
  std::cout << "=== Q-Learning ===" << std::endl;
  
  MyGridWorld env(5, 5);
  
  RL_Agent* agent = rl_create_agent(
    env.get_states_num(),
    env.get_actions_num(),
    0.1f,   // learning rate
    0.99f   // discount factor
  );
  
  rl_set_agent_policy(agent, RL_POLICY_EPSILON_GREEDY, 1.0f);
  rl_set_agent_epsilon_decay(agent, 1.0, 0.01, 0.0005, EPS_DECAY_EXPONENTIAL, 1);

  rl_set_agent_reward_clip(agent, 1, -10.0f, 10.0f);
  rl_set_agent_reward_normalization(agent, 0, 1.0f);
  
  int num_episodes = 100;
  int max_steps = 100;
  
  std::cout << "\nTraining Q-Learning Agent..." << std::endl;
  
  for (int episode = 0; episode < num_episodes; ++episode) {
    size_t state = env.reset();
    int steps = 0;
    float episode_reward = 0.0f;
    
    while (!env.is_done() && steps < max_steps) {
      size_t action = rl_choose_agent_action(agent, state);
      
      env.step(action);
      
      size_t next_state = env.get_state();
      float reward = env.get_reward();
      int done = env.is_done() ? 1 : 0;
      
      rl_update_agent(agent, state, action, reward, next_state, done);
      // update epsilon scheduler per step
      rl_update_agent_epsilon_step(agent);
      
      episode_reward += reward;
      state = next_state;
      steps++;
    }

    // Ensure episode finalized even if episode ended due to max_steps (not env.is_done())
    if (!env.is_done()) rl_notify_agent_episode_end(agent);
    
    if ((episode + 1) % 10 == 0) {
      std::cout << "Episode " << (episode + 1) << "/" << num_episodes 
                << " | Reward: " << episode_reward << " | Steps: " << steps
                << " | Epsilon: " << rl_get_agent_epsilon(agent);

      double avg = rl_get_agent_average_reward(agent);
      double last = rl_get_agent_last_reward(agent);
      size_t episodes = rl_get_agent_episode_count(agent);
      size_t last_len = rl_get_agent_last_episode_length(agent);
      double avg_len = rl_get_agent_average_episode_length(agent);
      std::cout << " | Telemetry: episodes=" << episodes << " last_reward=" << last << " avg_reward=" << avg
                << " last_len=" << last_len << " avg_len=" << avg_len << std::endl;
    }
  }
  
  std::cout << "\nTesting..." << std::endl;
  rl_set_agent_training(agent, 0);
  
  for (int test = 0; test < 5; ++test) {
    size_t state = env.reset();
    int steps = 0;
    float total_reward = 0.0f;
    
    while (!env.is_done() && steps < max_steps) {
      size_t action = rl_choose_agent_action(agent, state);
      env.step(action);
      state = env.get_state();
      total_reward += env.get_reward();
      steps++;
    }
    
    std::cout << "Test " << (test + 1) << " | Reward: " << total_reward 
              << " | Steps: " << steps << std::endl;
  }
  
  std::cout << "\nQ-table..." << std::endl;
  RL_QTable* qtable = rl_get_agent_qtable(agent);
  std::cout << "Example: Q(0, 3) = " << rl_get_qtable(qtable, 0, 3) << std::endl;
  
  // Save agent to disk
  const char* path = "agent_saved.bin";
  if (rl_save_agent(agent, path))
    std::cout << "Agent saved to " << path << std::endl;
  else
    std::cout << "Failed to save agent to " << path << std::endl;

  // Free original agent
  rl_free_agent(agent);
  rl_free_qtable(qtable);

  // Load agent back
  RL_Agent* loaded = rl_load_agent(path);
  if (loaded) {
    std::cout << "Loaded agent from " << path << std::endl;
    RL_QTable* q2 = rl_get_agent_qtable(loaded);
    std::cout << "Example: Q(0,3) after load = " << rl_get_qtable(q2, 0, 3) << std::endl;
    rl_free_qtable(q2);
    rl_free_agent(loaded);
  } else {
    std::cout << "Failed to load agent from " << path << std::endl;
  }
  
  return 0;
}
