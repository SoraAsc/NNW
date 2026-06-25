#pragma once

#include "actor_critic_policy.h"
#include "rollout_buffer.h"

class Optimizer;

class PPOAgent {
private:
  ActorCriticPolicy& policy;
  RolloutBuffer rollout_buffer;

  Optimizer& actor_optimizer;
  Optimizer& critic_optimizer;

  float gamma;
  float gae_lambda;
  float clip_range;
  float value_loss_coef;
  float entropy_coef;
  float max_grad_norm;

  size_t epochs;
  size_t minibatch_size;

public:
  PPOAgent(
    ActorCriticPolicy& policy,
    Optimizer& actor_optimizer,
    Optimizer& critic_optimizer,
    size_t max_envs,
    size_t rollout_steps,
    const std::vector<size_t>& state_shape,
    const std::vector<size_t>& action_shape,
    float gamma,
    float gae_lambda,
    float clip_range,
    float value_loss_coef,
    float entropy_coef,
    float max_grad_norm,
    size_t epochs,
    size_t minibatch_size);

  ~PPOAgent() = default;

  [[nodiscard]] StepOutput act(const Tensor& states) const;

  void store_transition(
    const Tensor& states,
    const Tensor& actions,
    const Tensor& log_probs,
    const Tensor& rewards,
    const Tensor& is_terminals,
    const Tensor& values);

  void train(const Tensor& next_value,
             const Tensor& next_is_terminal);

  [[nodiscard]] RolloutBuffer& get_rollout_buffer() { return rollout_buffer; }
  [[nodiscard]] const RolloutBuffer& get_rollout_buffer() const { return rollout_buffer; }

  [[nodiscard]] ActorCriticPolicy& get_policy() { return policy; }
  [[nodiscard]] const ActorCriticPolicy& get_policy() const { return policy; }
};