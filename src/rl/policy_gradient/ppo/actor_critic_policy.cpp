#include "rl/policy_gradient/ppo/actor_critic_policy.h"

#include "nn/model.h"

#include <stdexcept>
#include <string>

namespace {

Tensor flatten_values(const Tensor& values, size_t batch, const char* operation) {
  const auto& shape = values.shape();
  bool valid = (shape.size() == 1 && shape[0] == batch)
      || (shape.size() == 2 && shape[0] == batch && shape[1] == 1);
  if (!valid)
    throw std::runtime_error(
        std::string(operation) + ": critic must output [batch] or [batch, 1]");
  return values.reshape({batch});
}

} // namespace

ActorCriticPolicy::ActorCriticPolicy(
    Model& actor,
    Model& critic,
    std::unique_ptr<ActionDistribution> action_distribution,
    size_t actor_output_dim)
    : actor_net(actor), critic_net(critic), distribution(std::move(action_distribution)),
      actor_output_size(actor_output_dim) {
  if (!distribution) throw std::invalid_argument("ActorCriticPolicy requires an action distribution");
  (void)distribution->action_dim(actor_output_size);
}

StepOutput ActorCriticPolicy::act(const Tensor& states, bool deterministic) const {
  size_t batch = states.shape()[0];
  Tensor actor_output = actor_net.forward(states);
  Tensor actions = distribution->sample(actor_output, deterministic);
  DistributionEvaluation evaluation = distribution->evaluate(actor_output, actions);
  return {
      std::move(actions),
      std::move(evaluation.log_probs),
      flatten_values(critic_net.forward(states), batch, "act")};
}

EvaluationOutput ActorCriticPolicy::evaluate(const Tensor& states, const Tensor& actions) const {
  size_t batch = states.shape()[0];
  Tensor actor_output = actor_net.forward(states);
  DistributionEvaluation evaluation = distribution->evaluate(actor_output, actions);

  EvaluationOutput output;
  output.log_probs = std::move(evaluation.log_probs);
  output.entropy = std::move(evaluation.entropy);
  output.values = flatten_values(critic_net.forward(states), batch, "evaluate");
  output.actor_output = actor_output;
  output.actions_taken = actions;
  return output;
}

Tensor ActorCriticPolicy::actor_backward(
    const Tensor& actor_output,
    const Tensor& actions,
    const Tensor& grad_log_prob,
    float entropy_coef) const {
  return distribution->backward(actor_output, actions, grad_log_prob, entropy_coef);
}

size_t ActorCriticPolicy::action_dim() const {
  return distribution->action_dim(actor_output_size);
}
