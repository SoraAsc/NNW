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
  size_t numel() const;

  void zero();
  void fill(float value);
  
  // In-place operations for efficiency (especially useful for optimizers)
  void add_scalar_inplace(float scalar);
  void sub_scalar_inplace(float scalar);

  // Row access on the leading dimension (works for any rank ≥ 1)
  // A "row" here means the sub-tensor at index `row` along axis 0.
  // e.g. tensor shape [T, E, D] → get_row(t) returns shape [E, D]
  Tensor get_row(size_t row) const;
  void set_row(size_t row, const Tensor& src);

  // Reshape — total elements must be unchanged.
  // Pass (size_t)-1 for at most one dimension to infer it automatically.
  Tensor reshape(const std::vector<size_t>& new_shape) const;

  // Gather rows along axis 0 by index list.
  // Returns tensor whose leading dim == indices.size().
  Tensor gather(const std::vector<size_t>& indices) const;

  // Helper functions for common tensor operations
  static Tensor add_rowwise(const Tensor& a, const Tensor& rowVec); // Add row vector to each row of matrix a
  static Tensor reduce_sum_rows(const Tensor& a); // Sum over rows, result is 1D
  // Basic Operations
  static Tensor add(const Tensor& a, const Tensor& b);
  static Tensor sub(const Tensor& a, const Tensor& b);
  static Tensor mul(const Tensor& a, const Tensor& b);
  static float mean(const Tensor& a);
  static Tensor mul_scalar(const Tensor& a, float scalar);
  static Tensor matmul(const Tensor& a, const Tensor& b);
  static Tensor transpose(const Tensor& a);
  static float sum(const Tensor& a);   // sum of all elements
  static Tensor exp(const Tensor& a);
  static Tensor log(const Tensor& a);
  static Tensor square(const Tensor& a);
  static Tensor clamp(const Tensor& a, float min_val, float max_val);
  static Tensor minimum(const Tensor& a, const Tensor& b);  // element-wise min
  static Tensor maximum(const Tensor& a, const Tensor& b);  // element-wise max (symmetric, added for completeness)

  static Tensor zeros(const std::vector<size_t>& shape); // all-zero tensor
  static Tensor from_scalar(float value); // shape {1}
  static float std_val(const Tensor& a);  // population std-dev of all elements

  // Distribution
  static Tensor softmax(const Tensor& a);
  static Tensor sum_last_dim(const Tensor& a);

private:
  std::vector<size_t> m_shape;
  std::vector<float> m_data;

  size_t compute_numel() const;
};