// #pragma once

// I need to make this a proper cross-platform shared library interface later

// #ifdef _WIN32
//   #ifdef BUILDING_DLL
//     #define API __declspec(dllexport)
//   #else
//     #define API __declspec(dllimport)
//   #endif
// #else
//   #define API
// #endif

// // #include "tensor.h"
// #include "nn/layers/dense_layer.h"

// extern "C" {
//   API DenseLayer* dense_layer_create(size_t in_features, size_t out_features);
//   API void dense_layer_free(DenseLayer* layer);
//   API Tensor* dense_layer_forward(DenseLayer* layer, const Tensor* input);
//   API Tensor* dense_layer_backward(DenseLayer* layer, const Tensor* grad_output);
//   API void dense_layer_update(DenseLayer* layer, float learning_rate);

//   API const char* dense_info(DenseLayer* layer);
//   API const char* dense_detailed_info(DenseLayer* layer);

// }