#include "api.h"
#include "tensor.h"

Tensor* tensor_create(const size_t* shape, size_t dim) {
  std::vector<size_t> vec_shape(shape, shape + dim);
  return new Tensor(vec_shape);
}

void tensor_free(Tensor* tensor) {
  delete tensor;
}

float* tensor_data(Tensor* tensor) {
  return tensor->data();
}

const float* tensor_data_const(const Tensor* tensor) {
  return tensor->data();
}

const size_t* tensor_shape(const Tensor* tensor) {
  return tensor->shape().data();
}

Tensor* tensor_add(const Tensor* a, const Tensor* b) {
  return new Tensor(Tensor::add(*a, *b));
}

Tensor* tensor_mul_scalar(const Tensor* a, float scalar) {
  return new Tensor(Tensor::mul_scalar(*a, scalar));
}

Tensor* tensor_matmul(const Tensor* a, const Tensor* b) {
  return new Tensor(Tensor::matmul(*a, *b));
}
