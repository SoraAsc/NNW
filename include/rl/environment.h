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
  virtual size_t getState() const = 0;
  
  // Returns the reward of the last step  
  virtual float getReward() const = 0;
  
  // Returns the number of states
  virtual size_t getNumStates() const = 0;
  
  // Returns the number of actions
  virtual size_t getNumActions() const = 0;
  
  // Returns if the episode is finished
  virtual bool isDone() const = 0;
};
