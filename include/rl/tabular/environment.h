#pragma once
#include <cstddef>

class Environment {
public:
  virtual ~Environment() = default;
  
  // Execute one action ands returns if the episode finished
  virtual bool step(size_t action) = 0;
  
  // Reset the enviornment and returns the initial state
  virtual size_t reset() = 0;
  
  // Returns the current state
  virtual size_t get_state() const = 0;
  
  // Returns the reward of the last step  
  virtual float get_reward() const = 0;
  
  // Returns the number of states
  virtual size_t get_states_num() const = 0;
  
  // Returns the number of actions
  virtual size_t get_actions_num() const = 0;
  
  // Returns if the episode is finished
  virtual bool is_done() const = 0;
};
