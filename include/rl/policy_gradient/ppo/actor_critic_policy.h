#pragma once

#include "nn/tensor.h"
#include "rl/policy_gradient/ppo/action_distribution.h"

#include <cstddef>
#include <memory>

class Model;

struct StepOutput {
  Tensor actions;
  Tensor log_probs;
  Tensor values;
};

struct EvaluationOutput {
  Tensor log_probs;
  Tensor entropy;
  Tensor values;
  Tensor actor_output;
  Tensor actions_taken;
};

class ActorCriticPolicy {
public:
  ActorCriticPolicy(
      Model& actor,
      Model& critic,
      std::unique_ptr<ActionDistribution> distribution,
      size_t actor_output_dim);
  ~ActorCriticPolicy() = default;

  [[nodiscard]] StepOutput act(const Tensor& states, bool deterministic = false) const;
  [[nodiscard]] EvaluationOutput evaluate(const Tensor& states, const Tensor& actions) const;
  [[nodiscard]] Tensor actor_backward(
      const Tensor& actor_output,
      const Tensor& actions,
      const Tensor& grad_log_prob,
      float entropy_coef) const;

  [[nodiscard]] size_t action_dim() const;
  [[nodiscard]] Model& get_actor() { return actor_net; }
  [[nodiscard]] Model& get_critic() { return critic_net; }
  [[nodiscard]] const Model& get_actor() const { return actor_net; }
  [[nodiscard]] const Model& get_critic() const { return critic_net; }
  [[nodiscard]] const ActionDistribution& get_distribution() const { return *distribution; }

private:
  Model& actor_net;
  Model& critic_net;
  std::unique_ptr<ActionDistribution> distribution;
  size_t actor_output_size;
};
