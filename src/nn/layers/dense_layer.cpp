#include "nn/layers/dense_layer.h"
#include <random>
#include <sstream>
#include <iomanip>

DenseLayer::DenseLayer(const size_t in_features, const size_t out_features) {
  weights = Tensor({out_features, in_features}); // W: [out, in]
  biases = Tensor({out_features}); // b: [out]
  std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<float> dis(-0.1, 0.1);
  for(size_t i = 0; i < weights.numel(); i++) weights.data()[i] = dis(gen);
  for(size_t i = 0; i < biases.numel(); i++) biases.data()[i] = 0.0f;
};

DenseLayer::~DenseLayer() {} // Tensors are automatically cleaned up by their destructors

Tensor DenseLayer::forward(const Tensor& input) {
  // X: [batch, in]
  input_cache = input; // Cache input for backward pass
  // X @ W^T -> [batch, in] @ [in, out] = [batch, out]
  Tensor output = Tensor::matmul(input_cache, Tensor::transpose(weights));
  // + b (row-wise)
  output = Tensor::add_rowwise(output, biases);
  return output;
};

Tensor DenseLayer::backward(const Tensor& grad_output) {
  // grad_output: [batch, out]
  // dW = grad_output^T @ X -> [out, batch] @ [batch, in] = [out, in]
  grad_weights = Tensor::matmul(Tensor::transpose(grad_output), input_cache); // [out_features, in_features]
  // db = sum over rows of batch/grad_output -> [out]
  grad_biases = Tensor::reduce_sum_rows(grad_output); // [out_features]
  // dX = grad_output @ W -> [batch, out] @ [out, in] = [batch, in]
  Tensor grad_input = Tensor::matmul(grad_output, weights); // [batch_size, in_features]
  return grad_input;
};

void DenseLayer::update(float learning_rate) {
  weights = Tensor::add(weights, Tensor::mul_scalar(grad_weights, -learning_rate));
  biases = Tensor::add(biases, Tensor::mul_scalar(grad_biases, -learning_rate));
};

std::string DenseLayer::info() {
  std::ostringstream oss;
  oss << "Dense Layer: ";
  oss << "Output=" << weights.shape()[0] << ", ";
  oss << "Weights=" << weights.numel() << ", ";
  oss << "Biases=" << biases.numel();
  return oss.str();
}

std::string DenseLayer::detailed_info() {
  std::ostringstream oss;
  oss << "Dense Layer:\n";
  oss << "Input shape: ";
  for (auto s : input_cache.shape()) oss << s << " ";
  oss << "\nWeights (" << weights.shape()[0] << "x" << weights.shape()[1] << "):\n";

  for (size_t i = 0; i < weights.numel(); ++i) {
    oss << std::fixed << std::setprecision(3) << weights.data()[i] << " ";
    if ((i+1) % weights.shape()[1] == 0) oss << "\n";
  }

  oss << "Bias (" << biases.numel() << "): ";
  for (size_t i = 0; i < biases.numel(); ++i) oss << biases.data()[i] << " ";
  oss << "\n";

  oss << "Last input cache (" << input_cache.numel() << "): ";
  for (size_t i = 0; i < input_cache.numel(); ++i) oss << input_cache.data()[i] << " ";
  oss << "\n";

  return oss.str();
}