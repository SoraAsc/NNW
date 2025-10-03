#include "tensor.h"
#include <stdexcept>

Tensor::Tensor(const std::vector<size_t>& shape): m_shape(shape) {
  if(shape.empty()) throw std::invalid_argument("Shape cannot be empty");
  
  size_t total_size = 1;
  for (size_t dim : shape) total_size *= dim;
  
  m_data.resize(total_size, 0.0f); // Initialize with zeros
}

Tensor::~Tensor() {} // m_data and m_shape are automatically cleaned up because they are std::vector

float* Tensor::data() {
  return m_data.data();
}

const float* Tensor::data() const {
  return m_data.data();
}

const std::vector<size_t>& Tensor::shape() const {
  return m_shape;
}

const size_t Tensor::numel() const {
  return m_data.size();
}

Tensor Tensor::add(const Tensor& a, const Tensor& b) {
  if (a.m_shape != b.m_shape) throw std::invalid_argument("Shapes do not match for addition");
  
  Tensor result(a.m_shape);
  size_t total_size = a.m_data.size();
  
  for (size_t i = 0; i < total_size; ++i)
    result.m_data[i] = a.m_data[i] + b.m_data[i];
  
  return result;
}

Tensor Tensor::add_scalar(const Tensor& a, float scalar) {
  Tensor result(a.m_shape);
  size_t total_size = a.m_data.size();
  
  for (size_t i = 0; i < total_size; ++i)
    result.m_data[i] = a.m_data[i] + scalar;
  
  return result;
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
  
  std::vector<size_t> result_shape = {a.m_shape[0], b.m_shape[1]};
  Tensor result(result_shape);
  
  for (size_t i = 0; i < a.m_shape[0]; ++i) {
    for (size_t j = 0; j < b.m_shape[1]; ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < a.m_shape[1]; ++k) 
        sum += a.m_data[i * a.m_shape[1] + k] * b.m_data[k * b.m_shape[1] + j];

      result.m_data[i * result_shape[1] + j] = sum;
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