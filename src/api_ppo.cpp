#include "api_ppo.h"

#include "nn/model.h"
#include "nn/tensor.h"
#include "nn/training/optimizer/adamw_optimizer.h"
#include "nn/training/optimizer/optimizer.h"
#include "nn/training/optimizer/sgd_optimizer.h"
#include "rl/policy_gradient/ppo/actor_critic_policy.h"
#include "rl/policy_gradient/ppo/ppo_agent.h"

#include <memory>
#include <vector>

struct RL_PPOAgent {
  NN_Model* actor_model = nullptr;
  NN_Model* critic_model = nullptr;
  std::unique_ptr<ActorCriticPolicy> policy;
  std::unique_ptr<Optimizer> actor_optimizer;
  std::unique_ptr<Optimizer> critic_optimizer;
  std::unique_ptr<PPOAgent> impl;

  size_t state_dim = 0;
  size_t num_actions = 0;
};

namespace {

Tensor make_state_tensor(const float* states, size_t batch_size, size_t state_dim) {
  std::vector<float> values(states, states + batch_size * state_dim);
  return Tensor(values, {batch_size, state_dim});
}

Tensor make_vector_tensor(const float* values, size_t count) {
  std::vector<float> data(values, values + count);
  return Tensor(data, {count});
}

}  // namespace

RL_PPOAgent* rl_ppo_create_agent(
    NN_Model* actor_model,
    NN_Model* critic_model,
    RL_ActionSpaceType action_space,
    RL_OptimizerType optimizer_type,
    size_t num_envs,
    size_t rollout_steps,
    float learning_rate,
    float gamma,
    float gae_lambda,
    float clip_range,
    float value_loss_coef,
    float entropy_coef,
    float max_grad_norm,
    size_t epochs,
    size_t minibatch_size) {
  if (!actor_model || !critic_model) return nullptr;
  

  if (num_envs == 0 || rollout_steps == 0 || epochs == 0 || minibatch_size == 0) return nullptr;

  auto* agent = new RL_PPOAgent();
  agent->actor_model = actor_model;
  agent->critic_model = critic_model;
  agent->state_dim = rl_model_get_input_dim(actor_model);
  agent->num_actions = rl_model_get_output_dim(actor_model);

  Model* actor_impl = static_cast<Model*>(nn_model_get_internal(actor_model));
  Model* critic_impl = static_cast<Model*>(nn_model_get_internal(critic_model));
  if (!actor_impl || !critic_impl) {
    delete agent;
    return nullptr;
  }

  switch (optimizer_type) {
    case RL_OPT_SGD:
      agent->actor_optimizer = std::make_unique<SGD>(*actor_impl, learning_rate);
      agent->critic_optimizer = std::make_unique<SGD>(*critic_impl, learning_rate);
      break;
    default:
    case RL_OPT_ADAMW:
      agent->actor_optimizer = std::make_unique<AdamW>(*actor_impl, learning_rate);
      agent->critic_optimizer = std::make_unique<AdamW>(*critic_impl, learning_rate);
      break;
  }

  ActionSpaceType space = ActionSpaceType::Discrete;
  switch (action_space) {
    case RL_ACTION_CONTINUOUS: space = ActionSpaceType::Continuous; break;
    case RL_ACTION_MULTIDISCRETE: space = ActionSpaceType::MultiDiscrete; break;
    case RL_ACTION_MULTIBINARY: space = ActionSpaceType::MultiBinary; break;
    default: case RL_ACTION_DISCRETE: space = ActionSpaceType::Discrete; break;
  }

  agent->policy = std::make_unique<ActorCriticPolicy>(*actor_impl, *critic_impl, space);

  std::vector<size_t> state_shape{agent->state_dim};
  std::vector<size_t> action_shape{};

  agent->impl = std::make_unique<PPOAgent>(
      *agent->policy,
      *agent->actor_optimizer,
      *agent->critic_optimizer,
      num_envs,
      rollout_steps,
      state_shape,
      action_shape,
      gamma,
      gae_lambda,
      clip_range,
      value_loss_coef,
      entropy_coef,
      max_grad_norm,
      epochs,
      minibatch_size);

  return agent;
}

void rl_ppo_free_agent(RL_PPOAgent* agent) {
  delete agent;
}

void rl_ppo_collect_step(
    RL_PPOAgent* agent,
    const float* states,
    size_t batch_size,
    float* out_actions,
    float* out_log_probs,
    float* out_values) {
  if (!agent || !states || batch_size == 0) return;
  

  Tensor states_t = make_state_tensor(states, batch_size, agent->state_dim);
  StepOutput out = agent->impl->collect_step(states_t);

  const float* act_ptr = out.actions.data();
  const float* lp_ptr = out.log_probs.data();
  const float* val_ptr = out.values.data();

  if (out_actions)
    for (size_t i = 0; i < batch_size; ++i) out_actions[i] = act_ptr[i];

  if (out_log_probs) 
    for (size_t i = 0; i < batch_size; ++i) out_log_probs[i] = lp_ptr[i];

  if (out_values)
    for (size_t i = 0; i < batch_size; ++i) out_values[i] = val_ptr[i];
}

void rl_ppo_store_transition(
    RL_PPOAgent* agent,
    const float* states,
    size_t batch_size,
    const float* actions,
    const float* log_probs,
    const float* rewards,
    const float* terminals,
    const float* values) {
  if (!agent || !states || !actions || !log_probs || !rewards || !terminals || !values || batch_size == 0) {
    return;
  }

  Tensor states_t = make_state_tensor(states, batch_size, agent->state_dim);
  Tensor actions_t = make_vector_tensor(actions, batch_size);
  Tensor log_probs_t = make_vector_tensor(log_probs, batch_size);
  Tensor rewards_t = make_vector_tensor(rewards, batch_size);
  Tensor terminals_t = make_vector_tensor(terminals, batch_size);
  Tensor values_t = make_vector_tensor(values, batch_size);

  agent->impl->store_transition(states_t, actions_t, log_probs_t, rewards_t, terminals_t, values_t);
}

void rl_ppo_train(RL_PPOAgent* agent, const float* next_value, size_t batch_size, const float* next_terminal) 
{
  if (!agent || !next_value || !next_terminal || batch_size == 0) return;
  
  Tensor next_value_t = make_vector_tensor(next_value, batch_size);
  Tensor next_terminal_t = make_vector_tensor(next_terminal, batch_size);
  agent->impl->train(next_value_t, next_terminal_t);
}
