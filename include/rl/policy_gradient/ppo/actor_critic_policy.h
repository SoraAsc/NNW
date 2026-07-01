#pragma once

#include "nn/tensor.h"
#include <cstddef>
#include <vector>

class Model;

enum class ActionSpaceType {
  Discrete,
  Continuous,
  MultiDiscrete,
  MultiBinary
};

struct StepOutput {
  Tensor actions;   // [B, ...action_dim] or [B] for scalar actions
  Tensor log_probs; // [B]
  Tensor values;    // [B]
};

struct EvaluationOutput {
  Tensor log_probs; // [B]
  Tensor entropy;   // [B]
  Tensor values;    // [B]

  // Raw actor network output before distribution sampling/reduction.
  // Shape: [B, num_actions] for Discrete/MultiBinary,
  //        [B, action_dim]  for Continuous,
  //        [B, sum(nvec)]   for MultiDiscrete.
  // Needed by PPOAgent to compute the correct gradient w.r.t. actor logits.
  Tensor actor_output;

  // Actions that were evaluated, stored as indices/values (same as batch.actions).
  // Needed alongside actor_output to compute the log-softmax gradient.
  Tensor actions_taken;
};

class ActorCriticPolicy {
public:
  ActorCriticPolicy(Model& actor, Model& critic, ActionSpaceType space_type);
  ~ActorCriticPolicy() = default;

  // Call once after construction, before the first act()/evaluate().

  // Continuous: initialises the global log_std parameter.
  // init_value = 0.0 → std = 1.0 (recommended starting point).
  void init_log_std(size_t action_dim, float init_value = 0.0f);

  // MultiDiscrete: sets the number of categories per sub-action.
  // e.g. three sub-actions with 3, 4, 2 choices → nvec = {3, 4, 2}
  // actor_net must output [B, sum(nvec)] logits.
  void set_discrete_sizes(const std::vector<size_t>& nvec);

  // Core interface
  [[nodiscard]] StepOutput act(const Tensor& states) const;
  [[nodiscard]] EvaluationOutput evaluate(const Tensor& states, const Tensor& actions) const;

  // Accessors
  [[nodiscard]] Model& get_actor() { return actor_net; }
  [[nodiscard]] const Model& get_actor()  const { return actor_net; }

  [[nodiscard]] Model& get_critic() { return critic_net; }
  [[nodiscard]] const Model& get_critic() const { return critic_net; }

  [[nodiscard]] ActionSpaceType get_action_space() const { return action_space; }

  // log_std is a learnable parameter optimised alongside the actor weights.
  // The optimizer needs direct access to its data pointer and gradient.
  [[nodiscard]] Tensor& get_log_std() { return log_std; }
  [[nodiscard]] const Tensor& get_log_std() const { return log_std; }

private:
  Model& actor_net;
  Model& critic_net;
  const ActionSpaceType action_space;

  Tensor log_std; // Continuous only
  std::vector<size_t> discrete_sizes; // MultiDiscrete only
};