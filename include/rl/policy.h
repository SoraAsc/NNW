#pragma once
#include <cstddef>
#include <random>

// For now, i'll leave them all in the same file.
class Policy {
public:
  virtual ~Policy() = default;
  
  virtual size_t selectAction(const float* q_values, size_t num_actions) = 0;
};

class GreedyPolicy : public Policy {
public:
  GreedyPolicy() = default;
  
  size_t selectAction(const float* q_values, size_t num_actions) override;
};

class EpsilonGreedyPolicy : public Policy {
public:
  explicit EpsilonGreedyPolicy(float epsilon = 0.1f);
  
  size_t selectAction(const float* q_values, size_t num_actions) override;
  
  void setEpsilon(float epsilon) { m_epsilon = epsilon; }
  float getEpsilon() const { return m_epsilon; }

private:
  float m_epsilon;
  std::mt19937 m_rng{std::random_device{}()};
};
