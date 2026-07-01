#include "nn/tensor.h"
#include <stdexcept>
#include <cmath>

Tensor::Tensor(const std::vector<size_t>& shape): m_shape(shape) {
  if(shape.empty()) throw std::invalid_argument("Shape cannot be empty");
  m_data.resize(compute_numel(), 0.0f); // Initialize with zeros
}

Tensor::Tensor(const std::vector<float>& data, const std::vector<size_t>& shape): m_shape(shape), m_data(data) {
  if (compute_numel() != m_data.size())
    throw std::invalid_argument("Data size does not match shape");
}

Tensor::~Tensor() {} // m_data and m_shape are automatically cleaned up because they are std::vector

float* Tensor::data() {
  return m_data.data();
}

const float* Tensor::data() const {
  return m_data.data();
}

void Tensor::zero() {
  std::fill(m_data.begin(), m_data.end(), 0.0f);
}

void Tensor::fill(float value) {
  std::fill(m_data.begin(), m_data.end(), value);
}

void Tensor::add_scalar_inplace(float scalar) {
  for (float& val : m_data) val += scalar;
}

void Tensor::sub_scalar_inplace(float scalar) {
  for (float& val : m_data) val -= scalar;
}

// Returns the sub-tensor at index `row` along axis 0.
// Shape [d0, d1, ..., dk] → result shape [d1, ..., dk]
Tensor Tensor::get_row(size_t row) const {
  const auto& s = m_shape;
  if (s.empty()) throw std::runtime_error("get_row: tensor has no shape");
  if (row >= s[0]) throw std::out_of_range("get_row: row index out of bounds");
 
  // Row stride = product of all dims except the first
  size_t row_stride = numel() / s[0];
 
  // Build child shape  (drop leading dim)
  std::vector<size_t> child_shape(s.begin() + 1, s.end());
  if (child_shape.empty()) child_shape = {1}; // scalar row → shape {1}
 
  std::vector<float> child_data(
    m_data.begin() + row * row_stride,
    m_data.begin() + row * row_stride + row_stride);
 
  return Tensor(child_data, child_shape);
}

void Tensor::set_row(size_t row, const Tensor& src) {
  const auto& s = m_shape;
  if (s.empty()) throw std::runtime_error("set_row: tensor has no shape");
  if (row >= s[0]) throw std::out_of_range("set_row: row index out of bounds");
 
  size_t row_stride = numel() / s[0];
 
  if (src.numel() != row_stride) 
    throw std::invalid_argument("set_row: source numel does not match row stride");

  // Check that the sub-shapes are compatible, not just the total count.
  // Expected child shape is m_shape[1:] (or {1} for a 1-D parent).
  if (s.size() > 1) {
    const auto& src_s = src.shape();
    // src shape must equal m_shape[1:]
    bool shape_ok = (src_s.size() == s.size() - 1);
    if (shape_ok)
      for (size_t i = 0; i < src_s.size(); ++i)
        if (src_s[i] != s[i + 1]) { shape_ok = false; break; }

    if (!shape_ok)
      throw std::invalid_argument(
          "set_row: source shape does not match expected row shape "
          "(numel matched but layout differs)");
  }
 
 
  std::copy(src.m_data.begin(), src.m_data.end(), m_data.begin() + row * row_stride);
}

// Allows exactly one -1 (as size_t max) to mean "infer this dimension".
Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const
{
  constexpr size_t INFER = static_cast<size_t>(-1);
 
  size_t infer_idx  = new_shape.size(); // sentinel: no inferred dim
  size_t known_prod = 1;
 
  for (size_t i = 0; i < new_shape.size(); ++i)
  {
    if (new_shape[i] == INFER)
    {
      if (infer_idx != new_shape.size()) throw std::invalid_argument("reshape: only one dimension may be -1");
      infer_idx = i;
    }
    else known_prod *= new_shape[i];
  }
 
  std::vector<size_t> resolved = new_shape;
  if (infer_idx != new_shape.size())
  {
    if (numel() % known_prod != 0)
      throw std::invalid_argument("reshape: total elements not divisible for inferred dimension");
    resolved[infer_idx] = numel() / known_prod;
  }
 
  // Validate total matches
  size_t total = 1;
  for (size_t d : resolved) total *= d;
  if (total != numel()) throw std::invalid_argument("reshape: new shape has different numel");
 
  return Tensor(m_data, resolved);
}

size_t Tensor::compute_numel() const {
  size_t total = 1;
  for (size_t dim : m_shape) total *= dim;
  return total;
}

const std::vector<size_t>& Tensor::shape() const {
  return m_shape;
}

size_t Tensor::numel() const {
  return m_data.size();
}

// Helper functions for common tensor operations

Tensor Tensor::add_rowwise(const Tensor& a, const Tensor& rowVec) {
  if (a.m_shape.size() != 2 || rowVec.m_shape.size() != 1)
    throw std::invalid_argument("Input tensor must be 2D and row vector must be 1D");
  
  if (a.m_shape[1] != rowVec.m_shape[0])
    throw std::invalid_argument("Row vector size must match number of columns in matrix");
  
  Tensor result(a.m_shape);
  size_t rows = a.m_shape[0];
  size_t cols = a.m_shape[1];
  const float* rv_data = rowVec.m_data.data();
  
  // Optimized: row-wise operation with better cache locality
  for (size_t i = 0; i < rows; ++i) {
    const float* a_row = a.m_data.data() + i * cols;
    float* result_row = result.m_data.data() + i * cols;
    for (size_t j = 0; j < cols; ++j)
      result_row[j] = a_row[j] + rv_data[j];
  }
  
  return result;
}

// Gathers rows along axis 0 by an index list.
// Input shape [N, ...rest] → output shape [indices.size(), ...rest]
Tensor Tensor::gather(const std::vector<size_t>& indices) const {
  const auto& s = m_shape;
  if (s.empty()) throw std::runtime_error("gather: tensor has no shape");
 
  size_t row_stride = numel() / s[0];
 
  std::vector<size_t> out_shape = s;
  out_shape[0] = indices.size();
 
  std::vector<float> out_data(indices.size() * row_stride);
 
  for (size_t i = 0; i < indices.size(); ++i)
  {
    size_t src_row = indices[i];
    if (src_row >= s[0]) throw std::out_of_range("gather: index out of bounds");
 
    std::copy(
      m_data.begin() + src_row * row_stride,
      m_data.begin() + src_row * row_stride + row_stride,
      out_data.begin() + i * row_stride);
  }
 
  return Tensor(out_data, out_shape);
}

Tensor Tensor::reduce_sum_rows(const Tensor& a) {
  if (a.m_shape.size() != 2)
    throw std::invalid_argument("Input tensor must be 2D for row-wise sum");
  
  size_t rows = a.m_shape[0];
  size_t cols = a.m_shape[1];
  
  Tensor result({cols}); // Result is 1D with size equal to number of columns
  float* result_data = result.m_data.data();
  const float* a_data = a.m_data.data();
  
  // Optimized: iterate by rows instead of columns for better cache locality
  std::fill(result_data, result_data + cols, 0.0f);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j)
      result_data[j] += a_data[i * cols + j];
  }
  
  return result;
}

// Basic Operations

Tensor Tensor::add(const Tensor& a, const Tensor& b) {
  if (a.m_shape != b.m_shape) throw std::invalid_argument("Shapes do not match for addition");
  
  Tensor result(a.m_shape);
  size_t total_size = a.m_data.size();
  
  for (size_t i = 0; i < total_size; ++i)
    result.m_data[i] = a.m_data[i] + b.m_data[i];
  
  return result;
}

Tensor Tensor::sub(const Tensor& a, const Tensor& b) {
  if (a.m_shape != b.m_shape) throw std::invalid_argument("Shapes do not match for subtraction");
  
  Tensor result(a.m_shape);
  size_t total_size = a.m_data.size();
  
  for (size_t i = 0; i < total_size; ++i)
    result.m_data[i] = a.m_data[i] - b.m_data[i];
  
  return result;
}

Tensor Tensor::mul(const Tensor& a, const Tensor& b) {
  if (a.m_shape != b.m_shape) throw std::invalid_argument("Shapes do not match for multiplication");
  
  Tensor result(a.m_shape);
  size_t total_size = a.m_data.size();
  
  for (size_t i = 0; i < total_size; ++i)
    result.m_data[i] = a.m_data[i] * b.m_data[i];
  
  return result;
}

float Tensor::mean(const Tensor& a) {
  if (a.m_data.empty()) throw std::invalid_argument("Cannot compute mean of empty tensor");
  
  float sum = 0.0f;
  for (float val : a.m_data) sum += val;
  
  return sum / static_cast<float>(a.m_data.size());
}

Tensor Tensor::mul_scalar(const Tensor& a, float scalar) {
  Tensor result(a.m_shape);
  size_t total_size = a.m_data.size();
  
  for (size_t i = 0; i < total_size; ++i)
    result.m_data[i] = a.m_data[i] * scalar;
  
  return result;
}

Tensor Tensor::matmul(const Tensor& a, const Tensor& b) {
  if (a.m_shape.size() != 2 || b.m_shape.size() != 2)
    throw std::invalid_argument("Both tensors must be 2D for matrix multiplication");
  
  if (a.m_shape[1] != b.m_shape[0])
    throw std::invalid_argument("Inner dimensions do not match for matrix multiplication");
  
  size_t m = a.m_shape[0];  // rows of a
  size_t k = a.m_shape[1];  // cols of a / rows of b
  size_t n = b.m_shape[1];  // cols of b
  
  std::vector<size_t> result_shape = {m, n};
  Tensor result(result_shape);
  
  const float* a_data = a.m_data.data();
  const float* b_data = b.m_data.data();
  float* result_data = result.m_data.data();
  
  for (size_t i = 0; i < m; ++i) {
    std::fill(result_data + i * n, result_data + i * n + n, 0.0f);
    
    for (size_t k_idx = 0; k_idx < k; ++k_idx) {
      float a_val = a_data[i * k + k_idx];
      const float* b_row = b_data + k_idx * n;
      float* res_row = result_data + i * n;
      
      // Accumulate with sequential access to result and b
      for (size_t j = 0; j < n; ++j) res_row[j] += a_val * b_row[j];
    }
  }
  
  return result;
}

Tensor Tensor::transpose(const Tensor& a) {
  if (a.m_shape.size() != 2) throw std::invalid_argument("Tensor must be 2D for transpose");
  
  std::vector<size_t> result_shape = {a.m_shape[1], a.m_shape[0]};
  Tensor result(result_shape);
  
  for (size_t i = 0; i < a.m_shape[0]; ++i) {
    for (size_t j = 0; j < a.m_shape[1]; ++j)
      result.m_data[j * result_shape[1] + i] = a.m_data[i * a.m_shape[1] + j];
  }
  return result;
}

float Tensor::sum(const Tensor& a) {
  float acc = 0.0f;
  const float* ptr = a.data();
  size_t n = a.numel();
  for (size_t i = 0; i < n; ++i) acc += ptr[i];
  return acc;
}

Tensor Tensor::exp(const Tensor& a) {
  std::vector<float> out(a.numel());
  const float* ptr = a.data();
  for (size_t i = 0; i < a.numel(); ++i) out[i] = std::exp(ptr[i]);
  return Tensor(out, a.shape());
}

Tensor Tensor::log(const Tensor& a)
{
  constexpr float EPS = 1e-8f;
  std::vector<float> out(a.numel());
  const float* ptr = a.data();
  for (size_t i = 0; i < a.numel(); ++i) out[i] = std::log(std::max(ptr[i], EPS));
  return Tensor(out, a.shape());
}

Tensor Tensor::square(const Tensor& a) {
  std::vector<float> out(a.numel());
  const float* ptr = a.data();
  for (size_t i = 0; i < a.numel(); ++i) out[i] = ptr[i] * ptr[i];
  return Tensor(out, a.shape());
}

Tensor Tensor::clamp(const Tensor& a, float min_val, float max_val) {
  std::vector<float> out(a.numel());
  const float* ptr = a.data();
  for (size_t i = 0; i < a.numel(); ++i) out[i] = std::min(std::max(ptr[i], min_val), max_val);
  return Tensor(out, a.shape());
}

Tensor Tensor::minimum(const Tensor& a, const Tensor& b) {
  if (a.shape() != b.shape()) throw std::invalid_argument("minimum: shape mismatch");
  std::vector<float> out(a.numel());
  const float* pa = a.data();
  const float* pb = b.data();
  for (size_t i = 0; i < a.numel(); ++i) out[i] = std::min(pa[i], pb[i]);
  return Tensor(out, a.shape());
}

Tensor Tensor::maximum(const Tensor& a, const Tensor& b) {
  if (a.shape() != b.shape()) throw std::invalid_argument("maximum: shape mismatch");
  std::vector<float> out(a.numel());
  const float* pa = a.data();
  const float* pb = b.data();
  for (size_t i = 0; i < a.numel(); ++i) out[i] = std::max(pa[i], pb[i]);
  return Tensor(out, a.shape());
}

Tensor Tensor::zeros(const std::vector<size_t>& shape) {
  return Tensor(shape); // constructor already zero-initialises
}

Tensor Tensor::from_scalar(float value) {
  return Tensor(std::vector<float>{value}, {1});
}

float Tensor::std_val(const Tensor& a) {
  if (a.numel() == 0) throw std::invalid_argument("std_val: empty tensor");
  float m = Tensor::mean(a);
  float acc = 0.0f;
  const float* ptr = a.data();
  size_t n = a.numel();
  for (size_t i = 0; i < n; ++i)
  {
    float d = ptr[i] - m;
    acc += d * d;
  }
  return std::sqrt(acc / static_cast<float>(n));
}

Tensor Tensor::softmax(const Tensor& a) {
  if (a.shape().size() != 2) throw std::invalid_argument("softmax: expected 2-D tensor [B, C]");
 
  size_t B = a.shape()[0];
  size_t C = a.shape()[1];
  std::vector<float> out(B * C);
  const float* ptr = a.data();
 
  for (size_t b = 0; b < B; ++b)
  {
    const float* row = ptr + b * C;
    float*       dst = out.data() + b * C;
 
    // Find row max for numerical stability
    float row_max = row[0];
    for (size_t c = 1; c < C; ++c)
      if (row[c] > row_max) row_max = row[c];
 
    float sum_exp = 0.0f;
    for (size_t c = 0; c < C; ++c)
    {
      dst[c] = std::exp(row[c] - row_max);
      sum_exp += dst[c];
    }
    for (size_t c = 0; c < C; ++c) dst[c] /= sum_exp;
  }
 
  return Tensor(out, a.shape());
}

Tensor Tensor::sum_last_dim(const Tensor& a) {
  if (a.shape().size() != 2) throw std::invalid_argument("sum_last_dim: expected 2-D tensor [B, C]");
 
  size_t B = a.shape()[0];
  size_t C = a.shape()[1];
  std::vector<float> out(B, 0.0f);
  const float* ptr = a.data();
 
  for (size_t b = 0; b < B; ++b)
    for (size_t c = 0; c < C; ++c)
      out[b] += ptr[b * C + c];
 
  return Tensor(out, {B});
}