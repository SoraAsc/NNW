#include "rl/policy_gradient/ppo/rollout_buffer.h"
#include "nn/tensor.h"

#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <random>
#include <cmath>

namespace {

std::vector<size_t> prepend(size_t d0, size_t d1, const std::vector<size_t>& tail) {
  std::vector<size_t> s = {d0, d1};
  s.insert(s.end(), tail.begin(), tail.end());
  return s;
}

// Normalise the live region [0, steps*envs) of a [max_steps, max_envs] buffer.
void normalise_advantages(Tensor& buf, size_t steps, size_t envs) {
  size_t N = steps * envs;
  if (N == 0) return;   // nothing to normalise
  float* ptr = buf.data();

  float sum = 0.0f;
  for (size_t i = 0; i < N; ++i) sum += ptr[i];
  float m = sum / static_cast<float>(N);

  float sq = 0.0f;
  for (size_t i = 0; i < N; ++i) { float d = ptr[i] - m; sq += d * d; }
  float s = std::sqrt(sq / static_cast<float>(N)) + 1e-8f;

  for (size_t i = 0; i < N; ++i) ptr[i] = (ptr[i] - m) / s;
}

}

RolloutBuffer::RolloutBuffer(size_t max_envs,
                             size_t steps,
                             const std::vector<size_t>& state_shape_,
                             const std::vector<size_t>& action_shape_)
  : max_environments(max_envs),
    max_steps(steps),
    num_environments(max_envs),
    state_shape(state_shape_),
    action_shape(action_shape_)
{
  states = Tensor::zeros(prepend(max_steps, max_envs, state_shape_));
  actions = Tensor::zeros(prepend(max_steps, max_envs, action_shape_));
  log_probs = Tensor::zeros({max_steps, max_envs});
  rewards = Tensor::zeros({max_steps, max_envs});
  is_terminals = Tensor::zeros({max_steps, max_envs});
  values = Tensor::zeros({max_steps, max_envs});
  returns = Tensor::zeros({max_steps, max_envs});
  advantages = Tensor::zeros({max_steps, max_envs});
}

void RolloutBuffer::set_num_environments(size_t num_envs) {
  if (num_envs > max_environments) throw std::invalid_argument("set_num_environments: exceeds max_environments");
  num_environments = num_envs;
}

void RolloutBuffer::insert(const Tensor& new_states,
                           const Tensor& new_actions,
                           const Tensor& new_log_probs,
                           const Tensor& new_rewards,
                           const Tensor& new_is_terminals,
                           const Tensor& new_values)
{
  if (is_full()) throw std::runtime_error("RolloutBuffer::insert — buffer full, call clear()");

  states.set_row(current_step, new_states);
  actions.set_row(current_step, new_actions);
  log_probs.set_row(current_step, new_log_probs);
  rewards.set_row(current_step, new_rewards);
  is_terminals.set_row(current_step, new_is_terminals);
  values.set_row(current_step, new_values);

  ++current_step;
}


void RolloutBuffer::compute_returns_and_advantages(
    const Tensor& next_value,
    const Tensor& next_is_terminal,
    float gamma,
    float gae_lambda)
{
  if (next_value.numel() < num_environments || next_is_terminal.numel() < num_environments)
    throw std::invalid_argument("compute_returns_and_advantages: bootstrap tensors too small");

  const float* next_done = next_is_terminal.data();

  // Per-environment carry state
  std::vector<float> last_adv(num_environments, 0.0f);
  std::vector<float> last_val(next_value.data(), next_value.data() + num_environments);

  for (int t = static_cast<int>(current_step) - 1; t >= 0; --t) {
    Tensor V_row = values.get_row(t);
    Tensor r_row = rewards.get_row(t);
    Tensor done_row = is_terminals.get_row(t);

    const float* V = V_row.data();
    const float* r = r_row.data();
    const float* done = done_row.data();

    std::vector<float> adv_row(num_environments);
    std::vector<float> ret_row(num_environments);

    for (size_t e = 0; e < num_environments; ++e)
    {
      bool is_last = (t == static_cast<int>(current_step) - 1);

      float non_terminal = is_last ? 1.0f - next_done[e] : 1.0f - done[e];
      float delta = r[e] + gamma * non_terminal * last_val[e] - V[e];

      float adv = delta + gamma * gae_lambda * non_terminal * last_adv[e];

      adv_row[e] = adv;
      ret_row[e] = adv + V[e];

      last_adv[e] = adv;
      last_val[e] = V[e];
    }

    advantages.set_row(t, Tensor(adv_row, {num_environments}));
    returns.set_row(t, Tensor(ret_row, {num_environments}));
  }

  normalise_advantages(advantages, current_step, num_environments);
}


std::vector<TransitionBatch> RolloutBuffer::get_shuffled_minibatches(size_t batch_size) const {
  const size_t total = current_step * num_environments;
  if (total == 0) return {};

  // Shuffled flat indices: flat_idx → step = idx/E, env = idx%E
  std::vector<size_t> indices(total);
  std::iota(indices.begin(), indices.end(), 0);
  {
    std::mt19937 rng(std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), rng);
  }

  auto extract = [&](const Tensor& buf, size_t flat_idx) -> Tensor
  {
    size_t step = flat_idx / num_environments;
    size_t env = flat_idx % num_environments;
    return buf.get_row(step).get_row(env);
  };

  // Pre-measure element sizes from the first index
  size_t state_elem_n = extract(states,  indices[0]).numel();
  size_t action_elem_n = extract(actions, indices[0]).numel();

  std::vector<TransitionBatch> batches;
  batches.reserve((total + batch_size - 1) / batch_size);

  for (size_t start = 0; start < total; start += batch_size)
  {
    size_t end = std::min(start + batch_size, total);
    size_t bs = end - start;

    std::vector<float> b_states(bs * state_elem_n);
    std::vector<float> b_actions(bs * action_elem_n);
    std::vector<float> b_lp(bs), b_ret(bs), b_adv(bs), b_val(bs);

    for (size_t i = 0; i < bs; ++i)
    {
      size_t idx = indices[start + i];

      Tensor se = extract(states,  idx);
      std::copy(se.data(), se.data() + state_elem_n, b_states.begin() + i * state_elem_n);

      Tensor ae = extract(actions, idx);
      std::copy(ae.data(), ae.data() + action_elem_n, b_actions.begin() + i * action_elem_n);

      b_lp [i] = extract(log_probs, idx).data()[0];
      b_ret[i] = extract(returns, idx).data()[0];
      b_adv[i] = extract(advantages, idx).data()[0];
      b_val[i] = extract(values, idx).data()[0];
    }

    auto batch_state_shape = state_shape;
    batch_state_shape.insert(batch_state_shape.begin(), bs);

    auto batch_action_shape = action_shape;
    batch_action_shape.insert(batch_action_shape.begin(), bs);

    TransitionBatch batch;
    batch.states = Tensor(b_states, batch_state_shape);
    batch.actions = Tensor(b_actions, batch_action_shape);
    batch.log_probs = Tensor(b_lp, {bs});
    batch.returns = Tensor(b_ret, {bs});
    batch.advantages = Tensor(b_adv, {bs});
    batch.values = Tensor(b_val, {bs});

    batches.push_back(std::move(batch));
  }

  return batches;
}

void RolloutBuffer::clear()
{
  current_step = 0;
  states.zero(); actions.zero(); log_probs.zero();
  rewards.zero(); is_terminals.zero(); values.zero();
  returns.zero(); advantages.zero();
}
