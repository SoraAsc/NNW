#pragma once

#ifdef _WIN32
  #ifdef BUILDING_DLL
    #define API __declspec(dllexport)
  #else
    #define API __declspec(dllimport)
  #endif
#else
  #define API
#endif

#include "tensor.h"

extern "C" {
  API Tensor* tensor_create(const size_t* shape, size_t dim);
  API void tensor_free(Tensor* tensor);
  API float* tensor_data(Tensor* tensor);
  API const float* tensor_data_const(const Tensor* tensor); // Read-only access to data
  API const size_t* tensor_shape(const Tensor* tensor);

  // Basic Operations
  API Tensor* tensor_add(const Tensor* a, const Tensor* b);
  API Tensor* tensor_mul_scalar(const Tensor* a, float scalar);
  API Tensor* tensor_matmul(const Tensor* a, const Tensor* b);
  // API Tensor* tensor_relu(const Tensor* a);
  // API Tensor* tensor_sigmoid(const Tensor* a);
  // API Tensor* tensor_softmax(const Tensor* a);
  // API Tensor* tensor_tanh(const Tensor* a);

}