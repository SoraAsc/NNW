#pragma once
#include <vector>
#include <cstddef>

class Tensor {
public:
  Tensor() = default;
  Tensor(const std::vector<size_t>& shape);
  Tensor(const std::vector<float>& data, const std::vector<size_t>& shape);
  ~Tensor();

  float* data();
  const float* data() const; // Read-only access to data
  const std::vector<size_t>& shape() const;
  const size_t numel() const;

  void zero();
  void fill(float value);

  // Helper functions for common tensor operations
  static Tensor add_rowwise(const Tensor& a, const Tensor& rowVec); // Add row vector to each row of matrix a
  static Tensor reduce_sum_rows(const Tensor& a); // Sum over rows, result is 1D
  // Basic Operations
  static Tensor add(const Tensor& a, const Tensor& b);
  static Tensor mul_scalar(const Tensor& a, float scalar);
  static Tensor matmul(const Tensor& a, const Tensor& b);
  static Tensor transpose(const Tensor& a);

private:
  std::vector<size_t> m_shape;
  std::vector<float> m_data;

  size_t compute_numel() const;
};