#pragma once

#include "api_nn.h"

#include <cstddef>

#define API

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct RL_PPOAgent RL_PPOAgent;

  typedef enum {
    RL_ACTION_DISCRETE = 0,
    RL_ACTION_CONTINUOUS = 1,
    RL_ACTION_MULTIDISCRETE = 2,
    RL_ACTION_MULTIBINARY = 3
  } RL_ActionSpaceType;

  typedef enum {
    RL_OPT_SGD = 0,
    RL_OPT_ADAMW = 1
  } RL_OptimizerType;

  API RL_PPOAgent* rl_ppo_create_agent(
    RL_Model* actor_model,
    RL_Model* critic_model,
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
    size_t minibatch_size);

  API void rl_ppo_free_agent(RL_PPOAgent* agent);

  API void rl_ppo_collect_step(
    RL_PPOAgent* agent,
    const float* states,
    size_t batch_size,
    float* out_actions,
    float* out_log_probs,
    float* out_values);

  API void rl_ppo_store_transition(
    RL_PPOAgent* agent,
    const float* states,
    size_t batch_size,
    const float* actions,
    const float* log_probs,
    const float* rewards,
    const float* terminals,
    const float* values);

  API void rl_ppo_train(
    RL_PPOAgent* agent,
    const float* next_value,
    size_t batch_size,
    const float* next_terminal);

#ifdef __cplusplus
}
#endif
