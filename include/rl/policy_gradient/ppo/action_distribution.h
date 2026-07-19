#pragma once

#include "nn/tensor.h"

#include <cstddef>
#include <memory>
#include <vector>

struct DistributionEvaluation {
  Tensor log_probs; // [batch]
  Tensor entropy;   // [batch]
};

// Defines all action-space-specific PPO behaviour in one place. Custom C++
// distributions can implement this interface without changing PPOAgent.
class ActionDistribution {
public:
  virtual ~ActionDistribution() = default;

  [[nodiscard]] virtual size_t action_dim(size_t actor_output_dim) const = 0;
  [[nodiscard]] virtual Tensor sample(
      const Tensor& actor_output, bool deterministic = false) const = 0;
  [[nodiscard]] virtual DistributionEvaluation evaluate(
      const Tensor& actor_output, const Tensor& actions) const = 0;

  // Returns d(actor loss)/d(actor output). grad_log_prob is the PPO surrogate
  // gradient with respect to each sample's scalar joint log probability.
  [[nodiscard]] virtual Tensor backward(
      const Tensor& actor_output,
      const Tensor& actions,
      const Tensor& grad_log_prob,
      float entropy_coef) const = 0;
};

[[nodiscard]] std::unique_ptr<ActionDistribution> make_categorical_distribution();
[[nodiscard]] std::unique_ptr<ActionDistribution> make_squashed_gaussian_distribution(
    size_t action_dim, float initial_log_std = -1.0f);
[[nodiscard]] std::unique_ptr<ActionDistribution> make_multi_categorical_distribution(
    std::vector<size_t> category_sizes);
[[nodiscard]] std::unique_ptr<ActionDistribution> make_bernoulli_distribution();
