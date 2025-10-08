#include "nn/loss/loss_mse.h"
#include <stdexcept>

float MSELoss::forward(const Tensor& predicted, const Tensor& target) {
  if(predicted.shape() != target.shape())
    throw std::invalid_argument("Predicted and target tensors must have the same shape for MSE loss.");

  Tensor diff = Tensor::sub(predicted, target); // Element-wise difference
  Tensor sq_diff = Tensor::mul(diff, diff); // Element-wise square
  float mse = Tensor::mean(sq_diff); // Mean of squared differences
  return mse;
}

Tensor MSELoss::backward(const Tensor& predicted, const Tensor& target) {
  if(predicted.shape() != target.shape())
    throw std::invalid_argument("Predicted and target tensors must have the same shape for MSE loss.");

  size_t n = predicted.numel();
  Tensor grad = Tensor::mul_scalar(Tensor::sub(predicted, target), 2.0f / static_cast<float>(n)); // dL/dy = 2*(y_pred - y_true)/n
  return grad;
}