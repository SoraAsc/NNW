// Supports four action spaces:
//
//   Discrete      actor_net output: [B, num_actions]   (logits)
//                 sample: argmax of Categorical (Gumbel trick)
//                 log_prob: log softmax[chosen action]
//                 entropy: -sum(p * log p)
//
//   Continuous    actor_net output: [B, action_dim]    (mean)
//                 log_std: learned global parameter, shape [action_dim]
//                 sample: mean + std * N(0,1)
//                 log_prob: sum of Normal log-probs over dims
//                 entropy: sum(log_std) + const * action_dim
//
//   MultiDiscrete actor_net output: [B, sum(nvec)]     (concatenated logits)
//                 Each sub-action is independent Categorical.
//                 nvec stored as constructor parameter (see header note).
//
//   MultiBinary   actor_net output: [B, num_actions]   (logits)
//                 Each dim is independent Bernoulli.
//                 log_prob: sum of Bernoulli log-probs over dims
//
// For MultiDiscrete, the caller must set the discrete_sizes vector via
// set_discrete_sizes() before use.
#include "rl/policy_gradient/ppo/actor_critic_policy.h"
#include "nn/model.h"
#include "nn/tensor.h"

#include <cmath>
#include <random>
#include <stdexcept>
#include <numeric>
#include <algorithm>

// Thread-local RNG  (one generator per thread, seeded once)
static std::mt19937& get_rng()
{
  static thread_local std::mt19937 rng(std::random_device{}());
  return rng;
}

// Math helpers
namespace {

constexpr float LOG_2PI = 1.8378770664f; // log(2π)
constexpr float LOG_STD_MIN = -20.0f;
constexpr float LOG_STD_MAX = 2.0f;

// Sample from U(0,1) using the thread-local RNG.
float uniform01()
{
  static thread_local std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  return dist(get_rng());
}

// Sample from N(0,1).
float standard_normal()
{
  static thread_local std::normal_distribution<float> dist(0.0f, 1.0f);
  return dist(get_rng());
}

// Categorical helpers

std::vector<int> categorical_sample(const Tensor& logits)
{
  size_t B = logits.shape()[0];
  size_t C = logits.shape()[1];
  const float* ptr = logits.data();

  std::vector<int> actions(B);
  for (size_t b = 0; b < B; ++b)
  {
    const float* row = ptr + b * C;
    int best = 0;
    float best_val = -1e30f;
    for (size_t c = 0; c < C; ++c)
    {
      // Gumbel noise: -log(-log(U))
      float u = std::max(uniform01(), 1e-8f);
      float g = -std::log(-std::log(u));
      float v = row[c] + g;
      if (v > best_val) { best_val = v; best = static_cast<int>(c); }
    }
    actions[b] = best;
  }
  return actions;
}

Tensor categorical_log_prob(const Tensor& logits, const std::vector<int>& actions)
{
  size_t B = logits.shape()[0];
  size_t C = logits.shape()[1];
  const float* ptr = logits.data();

  // Numerically stable log-softmax: log_p[c] = logit[c] - log(sum_exp)
  std::vector<float> lp(B);
  for (size_t b = 0; b < B; ++b)
  {
    const float* row = ptr + b * C;
    float max_l = *std::max_element(row, row + C);
    float sum_exp = 0.0f;
    for (size_t c = 0; c < C; ++c) sum_exp += std::exp(row[c] - max_l);
    float log_sum = max_l + std::log(sum_exp);
    lp[b] = row[actions[b]] - log_sum;
  }
  return Tensor(lp, {B});
}

// Entropy of Categorical: -sum(p * log_p) per row. Returns [B].
Tensor categorical_entropy(const Tensor& logits)
{
  size_t B = logits.shape()[0];
  size_t C = logits.shape()[1];
  const float* ptr = logits.data();

  std::vector<float> ent(B);
  for (size_t b = 0; b < B; ++b)
  {
    const float* row = ptr + b * C;
    float max_l = *std::max_element(row, row + C);
    float sum_exp = 0.0f;
    for (size_t c = 0; c < C; ++c) sum_exp += std::exp(row[c] - max_l);
    float log_sum = max_l + std::log(sum_exp);

    float h = 0.0f;
    for (size_t c = 0; c < C; ++c)
    {
      float log_p = row[c] - log_sum;
      h -= std::exp(log_p) * log_p;
    }
    ent[b] = h;
  }
  return Tensor(ent, {B});
}

// log_prob from tensor of actions [B] (float, will be cast to int).
Tensor categorical_log_prob_from_tensor(const Tensor& logits, const Tensor& actions_t)
{
  size_t B = actions_t.numel();
  const float* aptr = actions_t.data();
  std::vector<int> acts(B);
  for (size_t i = 0; i < B; ++i) acts[i] = static_cast<int>(aptr[i]);
  return categorical_log_prob(logits, acts);
}

// Normal (Gaussian) helpers

Tensor normal_log_prob(const Tensor& mean, const Tensor& log_std_vec, const Tensor& actions)
{
  size_t B = mean.shape()[0];
  size_t D = mean.shape()[1];

  const float* mu = mean.data();
  const float* ls = log_std_vec.data();   // [D]
  const float* act = actions.data();

  std::vector<float> lp(B, 0.0f);
  for (size_t b = 0; b < B; ++b)
    for (size_t d = 0; d < D; ++d)
    {
      float std_d = std::exp(ls[d]);
      float z = (act[b * D + d] - mu[b * D + d]) / std_d;
      // log N(x; μ, σ) = -0.5*(z² + log(2π)) - log(σ)
      lp[b] += -0.5f * (z * z + LOG_2PI) - ls[d];
    }
  return Tensor(lp, {B});
}

Tensor normal_entropy(const Tensor& log_std_vec, size_t B)
{
  size_t D = log_std_vec.numel();
  const float* ls = log_std_vec.data();

  float h_per_sample = 0.0f;
  for (size_t d = 0; d < D; ++d) h_per_sample += 0.5f * (1.0f + LOG_2PI) + ls[d];

  return Tensor(std::vector<float>(B, h_per_sample), {B});
}

// Bernoulli helpers

Tensor bernoulli_sample(const Tensor& logits)
{
  size_t B = logits.shape()[0];
  size_t D = logits.shape()[1];
  const float* ptr = logits.data();

  std::vector<float> out(B * D);
  for (size_t i = 0; i < B * D; ++i)
  {
    float p = 1.0f / (1.0f + std::exp(-ptr[i]));  // sigmoid
    out[i] = (uniform01() < p) ? 1.0f : 0.0f;
  }
  return Tensor(out, {B, D});
}

Tensor bernoulli_log_prob(const Tensor& logits, const Tensor& actions)
{
  size_t B = logits.shape()[0];
  size_t D = logits.shape()[1];
  const float* lg = logits.data();
  const float* act = actions.data();

  std::vector<float> lp(B, 0.0f);
  for (size_t b = 0; b < B; ++b)
    for (size_t d = 0; d < D; ++d)
    {
      float l = lg[b * D + d];
      float x = act[b * D + d];
      // Stable softplus: avoids exp(l) overflow for large positive l.
      //   l >= 0: softplus(l) = l + log(1+exp(-l))  (exp of negative is safe)
      //   l <  0: softplus(l) = log(1+exp(l))        (exp of negative is safe)
      float sp = (l >= 0.0f) ? l + std::log(1.0f + std::exp(-l)) : std::log(1.0f + std::exp(l));
      lp[b] += x * l - sp;
    }
  return Tensor(lp, {B});
}

// Entropy of Bernoulli: -p*log(p) - (1-p)*log(1-p), summed over dims. Returns [B].
Tensor bernoulli_entropy(const Tensor& logits)
{
  size_t B = logits.shape()[0];
  size_t D = logits.shape()[1];
  const float* ptr = logits.data();
  constexpr float EPS = 1e-8f;

  std::vector<float> ent(B, 0.0f);
  for (size_t b = 0; b < B; ++b)
    for (size_t d = 0; d < D; ++d)
    {
      float p = 1.0f / (1.0f + std::exp(-ptr[b * D + d]));
      float q = 1.0f - p;
      ent[b] += -(p * std::log(p + EPS) + q * std::log(q + EPS));
    }
  return Tensor(ent, {B});
}

} // namespace

// Constructor
ActorCriticPolicy::ActorCriticPolicy(Model& actor, Model& critic, ActionSpaceType space_type)
  : actor_net(actor), critic_net(critic), action_space(space_type)
{
  // log_std is initialised to 0 (std=1) for Continuous only.
  // For MultiDiscrete, discrete_sizes must be set via set_discrete_sizes().
}

// set_discrete_sizes  (required for MultiDiscrete)
// nvec: number of categories per sub-action, e.g. {3, 4, 2}
void ActorCriticPolicy::set_discrete_sizes(const std::vector<size_t>& nvec)
{
  if (action_space != ActionSpaceType::MultiDiscrete)
    throw std::logic_error("set_discrete_sizes: only valid for MultiDiscrete action space");
  discrete_sizes = nvec;
}

// init_log_std  (required for Continuous — call once after construction)
// action_dim: dimensionality of the continuous action
// init_value: starting log_std (0.0 → std=1.0 is a sensible default)
void ActorCriticPolicy::init_log_std(size_t action_dim, float init_value)
{
  if (action_space != ActionSpaceType::Continuous)
    throw std::logic_error("init_log_std: only valid for Continuous action space");

  std::vector<float> init_data(action_dim, init_value);
  log_std = Tensor(init_data, {action_dim});
}

// act
// states: [B, state_dim]
// Returns StepOutput with shapes [B, ...action], [B], [B]
StepOutput ActorCriticPolicy::act(const Tensor& states) const
{
  size_t B = states.shape()[0];

  Tensor actor_out = actor_net.forward(states);   // shape depends on space
  Tensor value = critic_net.forward(states);  // expected [B, 1] or [B]

  // Critic must return shape [B] or [B, 1] — anything else is a config error.
  {
    const auto& vs = value.shape();
    bool ok = (vs.size() == 1 && vs[0] == B) || (vs.size() == 2 && vs[0] == B && vs[1] == 1);
    if (!ok)
      throw std::runtime_error(
        "act: critic_net must output shape [B] or [B,1]. "
        "Got numel=" + std::to_string(value.numel()) +
        " with " + std::to_string(vs.size()) + " dim(s). "
        "Check that critic output layer has out_features=1."
      );
  }
  Tensor values_flat = value.reshape({B});

  StepOutput out;
  out.values = values_flat;

  switch (action_space)
  {
    // Discrete
    // actor_out: [B, num_actions] (logits)
    case ActionSpaceType::Discrete:
    {
      if (actor_out.shape().size() != 2 || actor_out.shape()[0] != B)
        throw std::runtime_error(
          "act (Discrete): actor_net must output shape [B, num_actions], "
          "got " + std::to_string(actor_out.shape().size()) + " dim(s)."
        );
      std::vector<int> acts = categorical_sample(actor_out);
      out.log_probs = categorical_log_prob(actor_out, acts);

      // Store actions as float tensor [B]
      std::vector<float> act_f(B);
      for (size_t b = 0; b < B; ++b) act_f[b] = static_cast<float>(acts[b]);
      out.actions = Tensor(act_f, {B});
      break;
    }

    // Continuous
    // actor_out: [B, action_dim] (mean)
    // log_std: [action_dim]   (global parameter)
    case ActionSpaceType::Continuous:
    {
      if (log_std.numel() == 0)
        throw std::runtime_error("act: call init_log_std() before using Continuous policy");

      if (actor_out.shape().size() != 2 || actor_out.shape()[0] != B)
        throw std::runtime_error(
          "act (Continuous): actor_net must output shape [B, action_dim], "
          "got " + std::to_string(actor_out.shape().size()) + " dim(s)."
        );
      size_t D = actor_out.shape()[1];
      const float* mu_ptr = actor_out.data();
      const float* ls_ptr = log_std.data();

      // Clamp log_std for stability
      std::vector<float> ls_clamped(D);
      for (size_t d = 0; d < D; ++d) ls_clamped[d] = std::min(std::max(ls_ptr[d], LOG_STD_MIN), LOG_STD_MAX);
      Tensor ls_t(ls_clamped, {D});

      // Sample: action = mean + std * N(0,1)
      std::vector<float> act_data(B * D);
      for (size_t b = 0; b < B; ++b)
        for (size_t d = 0; d < D; ++d)
          act_data[b * D + d] = mu_ptr[b * D + d] + std::exp(ls_clamped[d]) * standard_normal();

      out.actions   = Tensor(act_data, {B, D});
      out.log_probs = normal_log_prob(actor_out, ls_t, out.actions);
      break;
    }

    // MultiDiscrete
    // actor_out: [B, sum(nvec)] — concatenated logits for each sub-action
    case ActionSpaceType::MultiDiscrete:
    {
      if (discrete_sizes.empty())
        throw std::runtime_error("act: call set_discrete_sizes() before using MultiDiscrete policy");

      // Validate that actor_net output width matches sum(nvec).
      size_t expected_C = 0;
      for (size_t n : discrete_sizes) expected_C += n;
      if (actor_out.shape().size() < 2 || actor_out.shape()[1] != expected_C)
        throw std::runtime_error(
          "act (MultiDiscrete): actor_net output width does not match sum(nvec). "
          "Expected " + std::to_string(expected_C) +
          ", got " + std::to_string(actor_out.shape().size() < 2 ? 0 : actor_out.shape()[1])
        );

      size_t num_sub = discrete_sizes.size();
      std::vector<float> act_data(B * num_sub);
      std::vector<float> lp_data(B, 0.0f);

      size_t offset = 0;
      for (size_t s = 0; s < num_sub; ++s)
      {
        size_t C = discrete_sizes[s];

        // Slice logits for this sub-action: rows [B], cols [offset, offset+C)
        std::vector<float> sub_logits(B * C);
        const float* src = actor_out.data();
        size_t total_C = actor_out.shape()[1];
        for (size_t b = 0; b < B; ++b)
          for (size_t c = 0; c < C; ++c)
            sub_logits[b * C + c] = src[b * total_C + offset + c];

        Tensor sub_t(sub_logits, {B, C});
        std::vector<int> sub_acts = categorical_sample(sub_t);
        Tensor sub_lp = categorical_log_prob(sub_t, sub_acts);

        const float* lp_ptr = sub_lp.data();
        for (size_t b = 0; b < B; ++b)
        {
          act_data[b * num_sub + s] = static_cast<float>(sub_acts[b]);
          lp_data[b] += lp_ptr[b];
        }
        offset += C;
      }

      out.actions = Tensor(act_data, {B, num_sub});
      out.log_probs = Tensor(lp_data,  {B});
      break;
    }

    // MultiBinary
    // actor_out: [B, num_actions] (logits for each independent Bernoulli)
    case ActionSpaceType::MultiBinary:
    {
      if (actor_out.shape().size() != 2 || actor_out.shape()[0] != B)
        throw std::runtime_error(
          "act (MultiBinary): actor_net must output shape [B, num_actions], "
          "got " + std::to_string(actor_out.shape().size()) + " dim(s)."
        );
      out.actions = bernoulli_sample(actor_out);
      out.log_probs = bernoulli_log_prob(actor_out, out.actions);
      break;
    }
  }

  return out;
}

// evaluate
// Called during training with a minibatch of stored (state, action) pairs.
// Returns log_probs, entropy, and values for the current policy parameters.
EvaluationOutput ActorCriticPolicy::evaluate(const Tensor& states, const Tensor& actions) const
{
  size_t B = states.shape()[0];

  Tensor actor_out = actor_net.forward(states);
  Tensor value = critic_net.forward(states);
  // Store raw actor output for gradient computation in PPOAgent.
  // Must be assigned before the switch modifies nothing (actor_out is read-only below).

  // Critic must return shape [B] or [B, 1] — anything else is a config error.
  {
    const auto& vs = value.shape();
    bool ok = (vs.size() == 1 && vs[0] == B) || (vs.size() == 2 && vs[0] == B && vs[1] == 1);
    if (!ok)
      throw std::runtime_error(
        "evaluate: critic_net must output shape [B] or [B,1]. "
        "Got numel=" + std::to_string(value.numel()) +
        " with " + std::to_string(vs.size()) + " dim(s). "
        "Check that critic output layer has out_features=1."
      );
  }
  Tensor values_flat = value.reshape({B});

  EvaluationOutput out;
  out.values = values_flat;
  out.actor_output = actor_out; // [B, out_dim] — raw logits/means

  switch (action_space)
  {
    case ActionSpaceType::Discrete:
    {
      out.log_probs = categorical_log_prob_from_tensor(actor_out, actions);
      out.entropy = categorical_entropy(actor_out);
      break;
    }

    case ActionSpaceType::Continuous:
    {
      if (log_std.numel() == 0) throw std::runtime_error("evaluate: call init_log_std() first");

      size_t D = actor_out.shape()[1];
      const float* ls_ptr = log_std.data();
      std::vector<float> ls_clamped(D);
      for (size_t d = 0; d < D; ++d) ls_clamped[d] = std::min(std::max(ls_ptr[d], LOG_STD_MIN), LOG_STD_MAX);
      Tensor ls_t(ls_clamped, {D});

      out.log_probs = normal_log_prob(actor_out, ls_t, actions);
      out.entropy = normal_entropy(ls_t, B);
      break;
    }

    case ActionSpaceType::MultiDiscrete:
    {
      if (discrete_sizes.empty()) throw std::runtime_error("evaluate: call set_discrete_sizes() first");

      // Validate actor_net output width matches sum(nvec).
      size_t expected_C = 0;
      for (size_t n : discrete_sizes) expected_C += n;
      if (actor_out.shape().size() < 2 || actor_out.shape()[1] != expected_C)
        throw std::runtime_error(
          "evaluate (MultiDiscrete): actor_net output width does not match sum(nvec). "
          "Expected " + std::to_string(expected_C) +
          ", got " + std::to_string(actor_out.shape().size() < 2 ? 0 : actor_out.shape()[1])
        );

      size_t num_sub = discrete_sizes.size();
      size_t total_C = actor_out.shape()[1];
      const float* src = actor_out.data();
      const float* act_ptr = actions.data();

      std::vector<float> lp_data(B, 0.0f);
      std::vector<float> ent_data(B, 0.0f);

      size_t offset = 0;
      for (size_t s = 0; s < num_sub; ++s)
      {
        size_t C = discrete_sizes[s];

        std::vector<float> sub_logits(B * C);
        for (size_t b = 0; b < B; ++b)
          for (size_t c = 0; c < C; ++c)
            sub_logits[b * C + c] = src[b * total_C + offset + c];

        Tensor sub_t(sub_logits, {B, C});

        // Extract the action for this sub-dimension from actions [B, num_sub]
        std::vector<int> sub_acts(B);
        for (size_t b = 0; b < B; ++b) sub_acts[b] = static_cast<int>(act_ptr[b * num_sub + s]);

        Tensor sub_lp = categorical_log_prob(sub_t, sub_acts);
        Tensor sub_ent = categorical_entropy(sub_t);

        const float* lp_ptr = sub_lp.data();
        const float* ent_ptr = sub_ent.data();
        for (size_t b = 0; b < B; ++b)
        {
          lp_data [b] += lp_ptr [b];
          ent_data[b] += ent_ptr[b];
        }
        offset += C;
      }

      out.log_probs = Tensor(lp_data, {B});
      out.entropy = Tensor(ent_data, {B});
      break;
    }

    case ActionSpaceType::MultiBinary:
    {
      out.log_probs = bernoulli_log_prob(actor_out, actions);
      out.entropy = bernoulli_entropy(actor_out);
      break;
    }
  }
  out.actions_taken = actions;
  return out;
}