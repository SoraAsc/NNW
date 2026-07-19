#include "rl/policy_gradient/ppo/action_distribution.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace {

constexpr float LOG_2PI = 1.8378770664f;
constexpr float EPS = 1e-6f;

std::mt19937& rng() {
  static thread_local std::mt19937 engine(std::random_device{}());
  return engine;
}

float uniform01() {
  static thread_local std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  return dist(rng());
}

float standard_normal() {
  static thread_local std::normal_distribution<float> dist(0.0f, 1.0f);
  return dist(rng());
}

void require_matrix(const Tensor& tensor, const char* name) {
  if (tensor.shape().size() != 2)
    throw std::invalid_argument(std::string(name) + " must have shape [batch, features]");
}

std::vector<float> softmax_row(const float* row, size_t count) {
  float maximum = *std::max_element(row, row + count);
  std::vector<float> probabilities(count);
  float sum = 0.0f;
  for (size_t i = 0; i < count; ++i) {
    probabilities[i] = std::exp(row[i] - maximum);
    sum += probabilities[i];
  }
  for (float& probability : probabilities) probability /= sum;
  return probabilities;
}

size_t categorical_sample(const float* logits, size_t count) {
  size_t best = 0;
  float best_value = -INFINITY;
  for (size_t i = 0; i < count; ++i) {
    float u = std::max(uniform01(), 1e-8f);
    float value = logits[i] - std::log(-std::log(u));
    if (value > best_value) { best = i; best_value = value; }
  }
  return best;
}

class CategoricalDistribution final : public ActionDistribution {
public:
  size_t action_dim(size_t) const override { return 1; }

  Tensor sample(const Tensor& output, bool deterministic) const override {
    require_matrix(output, "categorical actor output");
    size_t B = output.shape()[0], C = output.shape()[1];
    std::vector<float> actions(B);
    for (size_t b = 0; b < B; ++b) {
      const float* row = output.data() + b * C;
      actions[b] = static_cast<float>(deterministic
          ? static_cast<size_t>(std::max_element(row, row + C) - row)
          : categorical_sample(row, C));
    }
    return Tensor(actions, {B, 1});
  }

  DistributionEvaluation evaluate(const Tensor& output, const Tensor& actions) const override {
    size_t B = output.shape()[0], C = output.shape()[1];
    std::vector<float> log_probs(B), entropy(B);
    for (size_t b = 0; b < B; ++b) {
      auto p = softmax_row(output.data() + b * C, C);
      int action = static_cast<int>(actions.data()[b]);
      if (action < 0 || static_cast<size_t>(action) >= C) throw std::out_of_range("categorical action");
      log_probs[b] = std::log(p[action] + 1e-8f);
      for (size_t c = 0; c < C; ++c) entropy[b] -= p[c] * std::log(p[c] + 1e-8f);
    }
    return {Tensor(log_probs, {B}), Tensor(entropy, {B})};
  }

  Tensor backward(const Tensor& output, const Tensor& actions, const Tensor& grad_lp, float entropy_coef) const override {
    size_t B = output.shape()[0], C = output.shape()[1];
    std::vector<float> gradient(B * C);
    for (size_t b = 0; b < B; ++b) {
      auto p = softmax_row(output.data() + b * C, C);
      int action = static_cast<int>(actions.data()[b]);
      float h = 0.0f;
      for (size_t c = 0; c < C; ++c) h -= p[c] * std::log(p[c] + 1e-8f);
      for (size_t c = 0; c < C; ++c) {
        float indicator = static_cast<int>(c) == action ? 1.0f : 0.0f;
        float entropy_grad = entropy_coef / static_cast<float>(B) * p[c] * (std::log(p[c] + 1e-8f) + h);
        gradient[b * C + c] = grad_lp.data()[b] * (indicator - p[c]) + entropy_grad;
      }
    }
    return Tensor(gradient, {B, C});
  }
};

class SquashedGaussianDistribution final : public ActionDistribution {
public:
  SquashedGaussianDistribution(size_t dim, float initial_log_std)
      : log_std(std::vector<float>(dim, initial_log_std), {dim}) {}

  size_t action_dim(size_t actor_output_dim) const override {
    if (actor_output_dim != log_std.numel()) throw std::invalid_argument("Gaussian output dimension mismatch");
    return log_std.numel();
  }

  Tensor sample(const Tensor& output, bool deterministic) const override {
    require_matrix(output, "Gaussian actor output");
    size_t B = output.shape()[0], D = output.shape()[1];
    action_dim(D);
    std::vector<float> actions(B * D);
    for (size_t b = 0; b < B; ++b)
      for (size_t d = 0; d < D; ++d)
        actions[b * D + d] = std::tanh(output.data()[b * D + d]
            + (deterministic ? 0.0f : std::exp(log_std.data()[d]) * standard_normal()));
    return Tensor(actions, {B, D});
  }

  DistributionEvaluation evaluate(const Tensor& output, const Tensor& actions) const override {
    size_t B = output.shape()[0], D = output.shape()[1];
    std::vector<float> log_probs(B), entropy(B);
    for (size_t b = 0; b < B; ++b)
      for (size_t d = 0; d < D; ++d) {
        float action = std::clamp(actions.data()[b * D + d], -1.0f + EPS, 1.0f - EPS);
        float raw = std::atanh(action);
        float stddev = std::exp(log_std.data()[d]);
        float z = (raw - output.data()[b * D + d]) / stddev;
        log_probs[b] += -0.5f * (z * z + LOG_2PI) - log_std.data()[d] - std::log(1.0f - action * action + EPS);
        entropy[b] += 0.5f * (1.0f + LOG_2PI) + log_std.data()[d];
      }
    return {Tensor(log_probs, {B}), Tensor(entropy, {B})};
  }

  Tensor backward(const Tensor& output, const Tensor& actions, const Tensor& grad_lp, float) const override {
    size_t B = output.shape()[0], D = output.shape()[1];
    std::vector<float> gradient(B * D);
    for (size_t b = 0; b < B; ++b)
      for (size_t d = 0; d < D; ++d) {
        size_t i = b * D + d;
        float action = std::clamp(actions.data()[i], -1.0f + EPS, 1.0f - EPS);
        float variance = std::exp(2.0f * log_std.data()[d]);
        gradient[i] = grad_lp.data()[b] * (std::atanh(action) - output.data()[i]) / variance;
      }
    return Tensor(gradient, {B, D});
  }

private:
  Tensor log_std;
};

class BernoulliDistribution final : public ActionDistribution {
public:
  size_t action_dim(size_t actor_output_dim) const override { return actor_output_dim; }

  Tensor sample(const Tensor& output, bool deterministic) const override {
    size_t B = output.shape()[0], D = output.shape()[1];
    std::vector<float> actions(B * D);
    for (size_t i = 0; i < actions.size(); ++i) {
      float p = 1.0f / (1.0f + std::exp(-output.data()[i]));
      actions[i] = deterministic ? (p >= 0.5f ? 1.0f : 0.0f) : (uniform01() < p ? 1.0f : 0.0f);
    }
    return Tensor(actions, {B, D});
  }

  DistributionEvaluation evaluate(const Tensor& output, const Tensor& actions) const override {
    size_t B = output.shape()[0], D = output.shape()[1];
    std::vector<float> log_probs(B), entropy(B);
    for (size_t b = 0; b < B; ++b)
      for (size_t d = 0; d < D; ++d) {
        size_t i = b * D + d;
        float p = 1.0f / (1.0f + std::exp(-output.data()[i]));
        float a = actions.data()[i];
        log_probs[b] += a * std::log(p + 1e-8f) + (1.0f - a) * std::log(1.0f - p + 1e-8f);
        entropy[b] -= p * std::log(p + 1e-8f) + (1.0f - p) * std::log(1.0f - p + 1e-8f);
      }
    return {Tensor(log_probs, {B}), Tensor(entropy, {B})};
  }

  Tensor backward(const Tensor& output, const Tensor& actions, const Tensor& grad_lp, float entropy_coef) const override {
    size_t B = output.shape()[0], D = output.shape()[1];
    std::vector<float> gradient(B * D);
    for (size_t b = 0; b < B; ++b)
      for (size_t d = 0; d < D; ++d) {
        size_t i = b * D + d;
        float p = 1.0f / (1.0f + std::exp(-output.data()[i]));
        float entropy_derivative = p * (1.0f - p) * std::log((1.0f - p + 1e-8f) / (p + 1e-8f));
        gradient[i] = grad_lp.data()[b] * (actions.data()[i] - p)
            - entropy_coef / static_cast<float>(B) * entropy_derivative;
      }
    return Tensor(gradient, {B, D});
  }
};

class MultiCategoricalDistribution final : public ActionDistribution {
public:
  explicit MultiCategoricalDistribution(std::vector<size_t> sizes) : sizes(std::move(sizes)) {
    if (this->sizes.empty() || std::any_of(this->sizes.begin(), this->sizes.end(), [](size_t n) { return n < 2; }))
      throw std::invalid_argument("MultiCategorical requires category sizes >= 2");
  }

  size_t action_dim(size_t actor_output_dim) const override {
    size_t expected = std::accumulate(sizes.begin(), sizes.end(), size_t{0});
    if (actor_output_dim != expected) throw std::invalid_argument("MultiCategorical output must equal sum(category_sizes)");
    return sizes.size();
  }

  Tensor sample(const Tensor& output, bool deterministic) const override {
    size_t B = output.shape()[0], C = output.shape()[1], A = action_dim(C);
    std::vector<float> actions(B * A);
    size_t offset = 0;
    for (size_t a = 0; a < A; ++a) {
      for (size_t b = 0; b < B; ++b) {
        const float* row = output.data() + b * C + offset;
        actions[b * A + a] = static_cast<float>(deterministic
            ? static_cast<size_t>(std::max_element(row, row + sizes[a]) - row)
            : categorical_sample(row, sizes[a]));
      }
      offset += sizes[a];
    }
    return Tensor(actions, {B, A});
  }

  DistributionEvaluation evaluate(const Tensor& output, const Tensor& actions) const override {
    size_t B = output.shape()[0], C = output.shape()[1], A = action_dim(C);
    std::vector<float> log_probs(B), entropy(B);
    size_t offset = 0;
    for (size_t a = 0; a < A; ++a) {
      for (size_t b = 0; b < B; ++b) {
        auto p = softmax_row(output.data() + b * C + offset, sizes[a]);
        int selected = static_cast<int>(actions.data()[b * A + a]);
        if (selected < 0 || static_cast<size_t>(selected) >= sizes[a]) throw std::out_of_range("multi-categorical action");
        log_probs[b] += std::log(p[selected] + 1e-8f);
        for (float probability : p) entropy[b] -= probability * std::log(probability + 1e-8f);
      }
      offset += sizes[a];
    }
    return {Tensor(log_probs, {B}), Tensor(entropy, {B})};
  }

  Tensor backward(const Tensor& output, const Tensor& actions, const Tensor& grad_lp, float entropy_coef) const override {
    size_t B = output.shape()[0], C = output.shape()[1], A = action_dim(C);
    std::vector<float> gradient(B * C);
    size_t offset = 0;
    for (size_t a = 0; a < A; ++a) {
      for (size_t b = 0; b < B; ++b) {
        auto p = softmax_row(output.data() + b * C + offset, sizes[a]);
        int selected = static_cast<int>(actions.data()[b * A + a]);
        float h = 0.0f;
        for (float probability : p) h -= probability * std::log(probability + 1e-8f);
        for (size_t c = 0; c < sizes[a]; ++c) {
          float indicator = static_cast<int>(c) == selected ? 1.0f : 0.0f;
          float entropy_grad = entropy_coef / static_cast<float>(B) * p[c] * (std::log(p[c] + 1e-8f) + h);
          gradient[b * C + offset + c] = grad_lp.data()[b] * (indicator - p[c]) + entropy_grad;
        }
      }
      offset += sizes[a];
    }
    return Tensor(gradient, {B, C});
  }

private:
  std::vector<size_t> sizes;
};

} // namespace

std::unique_ptr<ActionDistribution> make_categorical_distribution() {
  return std::make_unique<CategoricalDistribution>();
}

std::unique_ptr<ActionDistribution> make_squashed_gaussian_distribution(size_t dim, float initial_log_std) {
  return std::make_unique<SquashedGaussianDistribution>(dim, initial_log_std);
}

std::unique_ptr<ActionDistribution> make_multi_categorical_distribution(std::vector<size_t> sizes) {
  return std::make_unique<MultiCategoricalDistribution>(std::move(sizes));
}

std::unique_ptr<ActionDistribution> make_bernoulli_distribution() {
  return std::make_unique<BernoulliDistribution>();
}
