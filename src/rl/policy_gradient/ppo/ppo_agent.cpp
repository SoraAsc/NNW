#include "rl/policy_gradient/ppo/ppo_agent.h"
#include "rl/policy_gradient/ppo/actor_critic_policy.h"
#include "rl/policy_gradient/ppo/rollout_buffer.h"
#include "nn/model.h"
#include "nn/tensor.h"

#include <stdexcept>
#include <cmath>
#include <algorithm>

#include "nn/training/optimizer/optimizer.h"
#include "nn/layers/layer.h"
#include <cmath>

static void clip_grad_norm(Model& model, float max_norm)
{
  if (max_norm <= 0.0f) return;

  // Compute global L2 norm across all gradient tensors
  float global_norm_sq = 0.0f;
  for (Layer* layer : model.layers())
    for (auto& [param, grad] : layer->get_parameters())
    {
      if (!grad) continue;
      const float* g = grad->data();
      size_t n = grad->numel();
      for (size_t i = 0; i < n; ++i) global_norm_sq += g[i] * g[i];
    }

  float global_norm = std::sqrt(global_norm_sq);
  if (global_norm <= max_norm) return;   // already within bound

  float scale = max_norm / (global_norm + 1e-6f);
  for (Layer* layer : model.layers())
    for (auto& [param, grad] : layer->get_parameters())
    {
      if (!grad) continue;
      float* g = grad->data();
      size_t n = grad->numel();
      for (size_t i = 0; i < n; ++i) g[i] *= scale;
    }
}

// Internal loss helpers
namespace {

// −mean( min(ratio*adv, clamp(ratio,1±ε)*adv) )
// Returns a scalar.
float actor_loss(const Tensor& new_log_probs, const Tensor& old_log_probs, const Tensor& advantages, float clip_range)
{
  size_t N = new_log_probs.numel();
  if (N == 0) return 0.0f;

  const float* nlp = new_log_probs.data();
  const float* olp = old_log_probs.data();
  const float* adv = advantages.data();

  float sum = 0.0f;
  for (size_t i = 0; i < N; ++i)
  {
    float ratio = std::exp(nlp[i] - olp[i]);
    float surr1 = ratio * adv[i];
    float surr2 = std::min(std::max(ratio, 1.0f - clip_range), 1.0f + clip_range) * adv[i];
    sum += std::min(surr1, surr2);
  }
  return -(sum / static_cast<float>(N));
}

// mean( (returns − values)² )
float value_loss(const Tensor& returns, const Tensor& values)
{
  size_t N = returns.numel();
  if (N == 0) return 0.0f;

  const float* ret = returns.data();
  const float* val = values.data();

  float sum = 0.0f;
  for (size_t i = 0; i < N; ++i)
  {
    float d = ret[i] - val[i];
    sum += d * d;
  }
  return sum / static_cast<float>(N);
}

// −mean(entropy)
float entropy_loss(const Tensor& entropy)
{
  size_t N = entropy.numel();
  if (N == 0) return 0.0f;

  const float* ptr = entropy.data();
  float sum = 0.0f;
  for (size_t i = 0; i < N; ++i) sum += ptr[i];
  return -(sum / static_cast<float>(N));
}

//  Backward pass helpers
Tensor grad_actor_loss(const Tensor& new_log_probs,
                        const Tensor& old_log_probs,
                        const Tensor& advantages,
                        float clip_range)
{
  size_t N = new_log_probs.numel();
  const float* nlp = new_log_probs.data();
  const float* olp = old_log_probs.data();
  const float* adv = advantages.data();

  std::vector<float> grad(N);
  float inv_N = 1.0f / static_cast<float>(N);

  for (size_t i = 0; i < N; ++i)
  {
    float ratio = std::exp(nlp[i] - olp[i]);
    float surr1 = ratio * adv[i];
    float ratio_clip = std::min(std::max(ratio, 1.0f - clip_range), 1.0f + clip_range);
    float surr2 = ratio_clip * adv[i];

    float g_actor = 0.0f;
    bool clipped  = (ratio < 1.0f - clip_range || ratio > 1.0f + clip_range);
    if (!clipped || surr1 <= surr2) g_actor = -ratio * adv[i] * inv_N;
    
    grad[i] = g_actor;
  }
  return Tensor(grad, {N});
}

// Gradient of value_loss w.r.t. values. Shape [B].
Tensor grad_value_loss(const Tensor& returns, const Tensor& values, float value_loss_coef)
{
  size_t N = returns.numel();
  const float* ret = returns.data();
  const float* val = values.data();
  float inv_N = 1.0f / static_cast<float>(N);

  std::vector<float> grad(N);
  for (size_t i = 0; i < N; ++i) grad[i] = value_loss_coef * (-2.0f * (ret[i] - val[i])) * inv_N;

  return Tensor(grad, {N});
}

} // namespace

PPOAgent::PPOAgent(
    ActorCriticPolicy& policy_,
    Optimizer& actor_opt,
    Optimizer& critic_opt,
    size_t max_envs,
    size_t rollout_steps,
    const std::vector<size_t>& state_shape,
    const std::vector<size_t>& action_shape,
    float gamma_,
    float gae_lambda_,
    float clip_range_,
    float value_loss_coef_,
    float entropy_coef_,
    float max_grad_norm_,
    size_t epochs_,
    size_t minibatch_size_)
  : policy(policy_),
    rollout_buffer(max_envs, rollout_steps, state_shape, action_shape),
    actor_optimizer(actor_opt),
    critic_optimizer(critic_opt),
    gamma(gamma_),
    gae_lambda(gae_lambda_),
    clip_range(clip_range_),
    value_loss_coef(value_loss_coef_),
    entropy_coef(entropy_coef_),
    max_grad_norm(max_grad_norm_),
    epochs(epochs_),
    minibatch_size(minibatch_size_)
{
  if (rollout_steps == 0) throw std::invalid_argument("PPOAgent: rollout_steps must be > 0");
  if (minibatch_size == 0) throw std::invalid_argument("PPOAgent: minibatch_size must be > 0");
  if (epochs == 0) throw std::invalid_argument("PPOAgent: epochs must be > 0");
}

StepOutput PPOAgent::collect_step(const Tensor& states) const
{
  return policy.act(states, !training);
}

void PPOAgent::store_transition(const Tensor& states,
                                 const Tensor& actions,
                                 const Tensor& log_probs,
                                 const Tensor& rewards,
                                 const Tensor& is_terminals,
                                 const Tensor& values)
{
  if (!training) return;
  rollout_buffer.insert(states, actions, log_probs, rewards, is_terminals, values);
}

void PPOAgent::train(const Tensor& next_value, const Tensor& next_is_terminal)
{
  if (!training) return;
  if (rollout_buffer.size() == 0) throw std::runtime_error("PPOAgent::train called with empty rollout buffer");

  // 1. Compute returns and advantages (GAE)
  rollout_buffer.compute_returns_and_advantages(next_value, next_is_terminal, gamma, gae_lambda);

  // 2. Optimisation loop
  for (size_t epoch = 0; epoch < epochs; ++epoch)
  {
    std::vector<TransitionBatch> batches = rollout_buffer.get_shuffled_minibatches(minibatch_size);

    for (const TransitionBatch& batch : batches)
    {
      size_t B = batch.states.shape()[0];

      // 2a. Forward pass
      // evaluate() runs actor_net and critic_net forward under current params.
      EvaluationOutput eval = policy.evaluate(batch.states, batch.actions);

      // 2b. Compute scalar losses (for logging / early-stop)
      float a_loss = actor_loss(eval.log_probs, batch.log_probs, batch.advantages, clip_range);

      float v_loss = value_loss(batch.returns, eval.values);

      float e_loss = entropy_loss(eval.entropy);

      (void)a_loss; // suppress unused-variable warning (maybe i will use it later...)
      (void)v_loss;
      (void)e_loss;

      // 2c. Backward: critic
      // Loss w.r.t. critic output (values): dL_value / d(values)
      Tensor grad_val = grad_value_loss(batch.returns, eval.values, value_loss_coef);

      // grad_val is [B]; critic output is [B] or [B,1].
      Tensor grad_critic_out = grad_val.reshape({B, 1});

      for (Layer* l : policy.get_critic().layers()) l->zero_grad();
      policy.get_critic().backward(grad_critic_out);
      clip_grad_norm(policy.get_critic(), max_grad_norm);
      critic_optimizer.step();

      Tensor grad_lp = grad_actor_loss(
          eval.log_probs, batch.log_probs, batch.advantages, clip_range);
      Tensor grad_actor_out = policy.actor_backward(
          eval.actor_output, eval.actions_taken, grad_lp, entropy_coef);

      for (Layer* l : policy.get_actor().layers()) l->zero_grad();
      policy.get_actor().backward(grad_actor_out);
      clip_grad_norm(policy.get_actor(), max_grad_norm);
      actor_optimizer.step();
    }
  }

  // 3. Clear buffer for the next rollout
  rollout_buffer.clear();
}

void PPOAgent::set_training(bool enabled)
{
  if (training == enabled) return;
  training = enabled;
  // Stored log-probabilities belong to the previous mode/policy interaction.
  // Never carry a partial training rollout across an evaluation interval.
  rollout_buffer.clear();
}
