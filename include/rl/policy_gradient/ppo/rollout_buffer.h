#pragma once

#include "nn/tensor.h"
#include <vector>
#include <cstddef>

struct TransitionBatch
{
  Tensor states;      // [Batch_Size, ...state_shape]
  Tensor actions;     // [Batch_Size, ...action_shape]
  Tensor log_probs;   // [Batch_Size, 1] or [Batch_Size]
  Tensor returns;     // [Batch_Size, 1] or [Batch_Size]
  Tensor advantages;  // [Batch_Size, 1] or [Batch_Size]
  Tensor values;      // [Batch_Size, 1] or [Batch_Size]
};

class RolloutBuffer {
private:
  Tensor states;
  Tensor actions;
  Tensor log_probs;
  Tensor rewards;
  Tensor is_terminals;
  Tensor values;
  Tensor returns;
  Tensor advantages;

  const size_t max_environments;
  const size_t max_steps;
  size_t num_environments;
  size_t current_step = 0; // Indicates which step of the collection process we are on (ranges from 0 to max_steps - 1)

  std::vector<size_t> state_shape;
  std::vector<size_t> action_shape;

public:
  RolloutBuffer(size_t max_envs, size_t steps, const std::vector<size_t>& state_shape, 
                  const std::vector<size_t>& action_shape);
  ~RolloutBuffer() = default;

  void set_num_environments(size_t num_envs);

  // The input tensors have the format [Num_Environments, Data_Dimension]
  void insert(const Tensor& states, const Tensor& actions, const Tensor& log_probs,
                const Tensor& rewards, const Tensor& is_terminals, const Tensor& values);

  void compute_returns_and_advantages(const Tensor& next_value, const Tensor& next_is_terminal, float gamma, float gae_lambda);

  std::vector<TransitionBatch> get_shuffled_minibatches(size_t batch_size) const;

  void clear();

  [[nodiscard]] size_t get_num_steps() const { return current_step; }
  [[nodiscard]] size_t get_max_steps() const { return max_steps; }
  [[nodiscard]] size_t get_num_envs() const { return num_environments; }
  [[nodiscard]] size_t get_max_envs() const { return max_environments; }
  [[nodiscard]] bool is_full() const { return current_step >= max_steps; }
  [[nodiscard]] size_t size() const { return current_step * num_environments; }
};
