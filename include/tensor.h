#pragma once
#include <vector>
#include <cstddef>

class Tensor {
public:
  Tensor(const std::vector<size_t>& shape);
  ~Tensor();

  float* data();
  const float* data() const; // Read-only access to data

  const std::vector<size_t>& shape() const;

  // Basic Operations
  static Tensor add(const Tensor& a, const Tensor& b);
  static Tensor mul_scalar(const Tensor& a, float scalar);
  static Tensor matmul(const Tensor& a, const Tensor& b);
  // static Tensor relu(const Tensor& a);
  // static Tensor sigmoid(const Tensor& a);
  // static Tensor softmax(const Tensor& a);
  // static Tensor transpose(const Tensor& a);
private:
  std::vector<size_t> m_shape;
  std::vector<float> m_data;
};