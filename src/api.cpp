#include "api.h"
#include "nn/layers/dense_layer.h"
#include <cstring>

DenseLayer* dense_layer_create(size_t in_features, size_t out_features) {
  return new DenseLayer(in_features, out_features);
}

void dense_layer_free(DenseLayer* layer) {
  delete layer;
}

Tensor* dense_layer_forward(DenseLayer* layer, const Tensor* input) {
  return new Tensor(layer->forward(*input));
}

Tensor* dense_layer_backward(DenseLayer* layer, const Tensor* grad_output) {
  return new Tensor(layer->backward(*grad_output));
}

void dense_layer_update(DenseLayer* layer, float learning_rate) {
  layer->update(learning_rate);
}

const char* dense_info(DenseLayer* layer) {
  static thread_local std::string buffer;
  buffer = layer->info();
  return buffer.c_str(); 
}

const char* dense_detailed_info(DenseLayer* layer) {
  static thread_local std::string buffer;
  buffer = layer->detailed_info();
  return buffer.c_str();
}