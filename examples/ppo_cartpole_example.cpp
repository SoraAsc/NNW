// Minimal PPO example for CartPole-v1.
#include "api_ppo.h"

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

  NN_Model* actor = nn_create_model(STATE_DIM);
  nn_add_dense(actor, 64, NN_ACT_TANH);
  nn_add_dense(actor, 64, NN_ACT_TANH);
  nn_add_dense(actor, NUM_ACTIONS, NN_ACT_LINEAR);

  NN_Model* critic = nn_create_model(STATE_DIM);
  nn_add_dense(critic, 64, NN_ACT_TANH);
  nn_add_dense(critic, 64, NN_ACT_TANH);
  nn_add_dense(critic, 1, NN_ACT_LINEAR);

  RL_PPOAgent* agent = rl_ppo_create_agent(
    actor,
    critic,
    RL_ACTION_DISCRETE,
    nullptr,
    0,
    RL_OPT_ADAMW,
    NUM_ENVS,
    ROLLOUT_STEPS,
    LR,
    GAMMA,
    GAE_LAMBDA,
    CLIP_RANGE,
    VALUE_COEF,
    ENTROPY_COEF,
    MAX_GRAD_NORM,
    EPOCHS,
    MINIBATCH_SIZE
  );

  if (!agent) {
    nn_free_model(actor);
    nn_free_model(critic);
    std::cerr << "Failed to create PPO agent\n";
    return 1;
  }

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
      std::vector<float> actions(NUM_ENVS, 0.0f);
      std::vector<float> log_probs(NUM_ENVS, 0.0f);
      std::vector<float> values(NUM_ENVS, 0.0f);
      rl_ppo_collect_step(agent, obs.data(), NUM_ENVS, actions.data(), log_probs.data(), values.data());

      std::vector<float> rewards(NUM_ENVS);
      std::vector<float> dones(NUM_ENVS);
      std::vector<float> next_obs(NUM_ENVS * STATE_DIM, 0.0f);

      for (size_t e = 0; e < NUM_ENVS; ++e)
      {
        int action = static_cast<int>(actions[e]);
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

      rl_ppo_store_transition(
        agent,
        obs.data(),
        NUM_ENVS,
        actions.data(),
        log_probs.data(),
        rewards.data(),
        dones.data(),
        values.data());

      obs.swap(next_obs);
      total_collected += NUM_ENVS;
    }

    std::vector<float> boot_values(NUM_ENVS, 0.0f);
    std::vector<float> boot_actions(NUM_ENVS, 0.0f);
    std::vector<float> boot_log_probs(NUM_ENVS, 0.0f);
    rl_ppo_collect_step(agent, obs.data(), NUM_ENVS, boot_actions.data(), boot_log_probs.data(), boot_values.data());

    rl_ppo_train(agent, boot_values.data(), NUM_ENVS, next_done.data());

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

  rl_ppo_free_agent(agent);
  nn_free_model(actor);
  nn_free_model(critic);

  std::cout << "\nTraining complete.\n";
  return 0;
}
