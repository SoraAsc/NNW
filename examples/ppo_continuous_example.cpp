#include <iostream>
#include <vector>

#include "api_nn.h"
#include "api_ppo.h"

int main() {
  constexpr size_t state_dim = 9;
  constexpr size_t action_dim = 2;
  constexpr size_t rollout_steps = 4;

  NN_Model* actor = nn_create_model(state_dim);
  nn_add_dense(actor, 32, NN_ACT_TANH);
  nn_add_dense(actor, 32, NN_ACT_TANH);
  nn_add_dense(actor, action_dim, NN_ACT_TANH);

  NN_Model* critic = nn_create_model(state_dim);
  nn_add_dense(critic, 32, NN_ACT_TANH);
  nn_add_dense(critic, 32, NN_ACT_TANH);
  nn_add_dense(critic, 1, NN_ACT_LINEAR);

  RL_PPOAgent* agent = rl_ppo_create_agent(
      actor,
      critic,
      RL_ACTION_CONTINUOUS,
      nullptr,
      0,
      RL_OPT_ADAMW,
      1,
      rollout_steps,
      3e-4f,
      0.99f,
      0.95f,
      0.2f,
      0.5f,
      0.01f,
      0.5f,
      4,
      1);

  if (!agent) {
    std::cout << "Failed to create continuous PPO agent\n";
    nn_free_model(actor);
    nn_free_model(critic);
    return 1;
  }

  std::vector<float> state(state_dim, 0.0f);
  std::vector<float> reward(1, 0.0f);
  std::vector<float> done(1, 0.0f);

  for (size_t step = 0; step < rollout_steps; ++step) {
    std::vector<float> actions(action_dim);
    std::vector<float> log_probs(1);
    std::vector<float> values(1);

    rl_ppo_collect_step(agent, state.data(), 1, actions.data(), log_probs.data(), values.data());
    rl_ppo_store_transition(agent, state.data(), 1, actions.data(), log_probs.data(), reward.data(), done.data(), values.data());

    std::cout << "step " << step << " action=[" << actions[0] << ", " << actions[1] << "]"
              << " logp=" << log_probs[0] << " value=" << values[0] << "\n";
  }

  std::vector<float> bootstrap_value(1, 0.0f);
  std::vector<float> bootstrap_done(1, 1.0f);
  rl_ppo_train(agent, bootstrap_value.data(), 1, bootstrap_done.data());

  rl_ppo_free_agent(agent);
  nn_free_model(actor);
  nn_free_model(critic);
  return 0;
}
