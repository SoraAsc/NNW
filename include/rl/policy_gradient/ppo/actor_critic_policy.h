#pragma once

#include "nn/tensor.h"
#include <cstddef>

class Model;

enum class ActionSpaceType {
  Discrete,
  Continuous,
  MultiDiscrete,
  MultiBinary
};

struct StepOutput {
  Tensor actions;   // [Num_Environments, ...action_dim]
  Tensor log_probs; // [Num_Environments, 1]
  Tensor values;    // [Num_Environments, 1]
};

struct EvaluationOutput {
  Tensor log_probs; // [Batch_Size, 1]
  Tensor entropy;   // [Batch_Size, 1]
  Tensor values;    // [Batch_Size, 1]
};

class ActorCriticPolicy {
private:
  Model& actor_net;
  Model& critic_net;
  const ActionSpaceType action_space;

public:
  ActorCriticPolicy(Model& actor, Model& critic, ActionSpaceType space_type);
  ~ActorCriticPolicy() = default;

  [[nodiscard]] StepOutput act(const Tensor& states) const;
  [[nodiscard]] EvaluationOutput evaluate(const Tensor& states, const Tensor& actions) const;

  [[nodiscard]] Model& get_actor() { return actor_net; }
  [[nodiscard]] const Model& get_actor() const { return actor_net; }

  [[nodiscard]] Model& get_critic() { return critic_net; }
  [[nodiscard]] const Model& get_critic() const { return critic_net; }

  [[nodiscard]] ActionSpaceType get_action_space() const { return action_space; }
};