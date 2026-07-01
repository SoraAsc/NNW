// Minimal PPO example for CartPole-v1.
#include "nn/layers/dense_layer.h"
#include "nn/model.h"
#include "nn/training/optimizer/adamw_optimizer.h"
#include "rl/policy_gradient/ppo/actor_critic_policy.h"
#include "rl/policy_gradient/ppo/ppo_agent.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

struct CartPoleEnv {
  static constexpr float GRAVITY = 9.8f;
  static constexpr float MASSCART = 1.0f;
  static constexpr float MASSPOLE = 0.1f;
  static constexpr float TOTAL_MASS = MASSCART + MASSPOLE;
  static constexpr float HALF_POLE = 0.5f;
  static constexpr float POLEMASS_LEN = MASSPOLE * HALF_POLE;
  static constexpr float FORCE_MAG = 10.0f;
  static constexpr float TAU = 0.02f;

  static constexpr float ANGLE_THRESHOLD = 12.0f * M_PI / 180.0f;
  static constexpr float POS_THRESHOLD = 2.4f;

  float x, x_dot, theta, theta_dot;
  bool done;
  std::mt19937 rng;

  explicit CartPoleEnv(uint32_t seed = 0) : rng(seed) { reset(); }

  std::vector<float> reset() {
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    x = dist(rng);
    x_dot = dist(rng);
    theta = dist(rng);
    theta_dot = dist(rng);
    done = false;
    return state();
  }

  struct StepResult {
    std::vector<float> obs;
    float reward;
    bool done;
  };

  StepResult step(int action) {
    float force = (action == 1) ? FORCE_MAG : -FORCE_MAG;

    float cos_t = std::cos(theta);
    float sin_t = std::sin(theta);
    float temp = (force + POLEMASS_LEN * theta_dot * theta_dot * sin_t) / TOTAL_MASS;
    float theta_acc = (GRAVITY * sin_t - cos_t * temp) /
                      (HALF_POLE * (4.0f / 3.0f - MASSPOLE * cos_t * cos_t / TOTAL_MASS));
    float x_acc = temp - POLEMASS_LEN * theta_acc * cos_t / TOTAL_MASS;

    x += TAU * x_dot;
    x_dot += TAU * x_acc;
    theta += TAU * theta_dot;
    theta_dot += TAU * theta_acc;

    done = std::abs(x) > POS_THRESHOLD || std::abs(theta) > ANGLE_THRESHOLD;
    return {state(), 1.0f, done};
  }

  std::vector<float> state() const { return {x, x_dot, theta, theta_dot}; }
};

Tensor pack_obs(const std::vector<float>& obs, size_t num_envs, size_t state_dim)
{
  return Tensor(obs, {num_envs, state_dim});
}

Tensor pack_scalar(const std::vector<float>& values)
{
  return Tensor(values, {values.size()});
}

int main()
{
  constexpr size_t STATE_DIM = 4;
  constexpr size_t NUM_ACTIONS = 2;
  constexpr size_t NUM_ENVS = 8;
  constexpr size_t ROLLOUT_STEPS = 128;
  constexpr size_t TOTAL_STEPS = 200'000;
  constexpr size_t EPOCHS = 4;
  constexpr size_t MINIBATCH_SIZE = 64;

  constexpr float LR = 3e-4f;
  constexpr float GAMMA = 0.99f;
  constexpr float GAE_LAMBDA = 0.95f;
  constexpr float CLIP_RANGE = 0.2f;
  constexpr float VALUE_COEF = 0.5f;
  constexpr float ENTROPY_COEF = 0.01f;
  constexpr float MAX_GRAD_NORM = 0.5f;

  Model actor_net;
  actor_net.add_layer(new DenseLayer(STATE_DIM, 64, ActivationType::TANH));
  actor_net.add_layer(new DenseLayer(64, 64, ActivationType::TANH));
  actor_net.add_layer(new DenseLayer(64, NUM_ACTIONS, ActivationType::NONE));

  Model critic_net;
  critic_net.add_layer(new DenseLayer(STATE_DIM, 64, ActivationType::TANH));
  critic_net.add_layer(new DenseLayer(64, 64, ActivationType::TANH));
  critic_net.add_layer(new DenseLayer(64, 1, ActivationType::NONE));

  AdamW actor_opt(actor_net, LR);
  AdamW critic_opt(critic_net, LR);

  ActorCriticPolicy policy(actor_net, critic_net, ActionSpaceType::Discrete);

  PPOAgent agent(
    policy,
    actor_opt,
    critic_opt,
    NUM_ENVS,
    ROLLOUT_STEPS,
    {STATE_DIM},
    {},
    GAMMA,
    GAE_LAMBDA,
    CLIP_RANGE,
    VALUE_COEF,
    ENTROPY_COEF,
    MAX_GRAD_NORM,
    EPOCHS,
    MINIBATCH_SIZE
  );
  agent.get_rollout_buffer().set_num_environments(NUM_ENVS);

  std::vector<CartPoleEnv> envs;
  envs.reserve(NUM_ENVS);
  for (size_t i = 0; i < NUM_ENVS; ++i) envs.emplace_back(static_cast<uint32_t>(i * 42));

  std::vector<float> obs(NUM_ENVS * STATE_DIM);
  for (size_t i = 0; i < NUM_ENVS; ++i) {
    auto state = envs[i].reset();
    std::copy(state.begin(), state.end(), obs.begin() + i * STATE_DIM);
  }

  size_t total_collected = 0;
  size_t episode_count = 0;
  std::vector<float> ep_rewards;
  ep_rewards.reserve(100);

  std::vector<float> env_ep_reward(NUM_ENVS, 0.0f);

  std::cout << "Training PPO on CartPole-v1\n"
            << "  envs=" << NUM_ENVS
            << "  rollout=" << ROLLOUT_STEPS
            << "  total_steps=" << TOTAL_STEPS << "\n\n";

  while (total_collected < TOTAL_STEPS) {
    std::vector<float> next_done(NUM_ENVS, 0.0f);

    for (size_t step = 0; step < ROLLOUT_STEPS; ++step) {
      Tensor obs_t = pack_obs(obs, NUM_ENVS, STATE_DIM);
      StepOutput out = agent.collect_step(obs_t);

      const float* act_ptr = out.actions.data();
      const float* lp_ptr = out.log_probs.data();
      const float* val_ptr = out.values.data();

      std::vector<float> rewards(NUM_ENVS);
      std::vector<float> dones(NUM_ENVS);
      std::vector<float> next_obs(NUM_ENVS * STATE_DIM, 0.0f);

      for (size_t e = 0; e < NUM_ENVS; ++e)
      {
        int action = static_cast<int>(act_ptr[e]);
        auto [nobs, rew, done] = envs[e].step(action);

        rewards[e] = rew;
        dones[e] = done ? 1.0f : 0.0f;

        env_ep_reward[e] += rew;

        if (done)
        {
          ep_rewards.push_back(env_ep_reward[e]);
          if (ep_rewards.size() > 100) ep_rewards.erase(ep_rewards.begin());

          env_ep_reward[e] = 0.0f;
          ++episode_count;

          auto reset_obs = envs[e].reset();

          std::copy(reset_obs.begin(), reset_obs.end(), next_obs.begin() + e * STATE_DIM);
        } else std::copy(nobs.begin(), nobs.end(), next_obs.begin() + e * STATE_DIM);
      }

      agent.store_transition(
        obs_t,
        out.actions,
        out.log_probs,
        pack_scalar(rewards),
        pack_scalar(dones),
        out.values
      );

      obs.swap(next_obs);
      total_collected += NUM_ENVS;
    }

    Tensor last_obs_t = pack_obs(obs, NUM_ENVS, STATE_DIM);
    StepOutput boot = agent.collect_step(last_obs_t);

    agent.train(boot.values, pack_scalar(next_done));

    if (total_collected % 10'000 < NUM_ENVS * ROLLOUT_STEPS) {
      float mean_rew = 0.0f;
      if (!ep_rewards.empty()) {
        for (float r : ep_rewards) mean_rew += r;
        mean_rew /= static_cast<float>(ep_rewards.size());
      }

      std::cout << "steps=" << total_collected
                << "  episodes=" << episode_count
                << "  mean_ep_reward(last100)=" << mean_rew << "\n";

      if (mean_rew >= 195.0f && ep_rewards.size() >= 100) {
        std::cout << "\nCartPole solved in " << total_collected << " steps!\n";
        break;
      }
    }
  }

  std::cout << "\nTraining complete.\n";
  return 0;
}