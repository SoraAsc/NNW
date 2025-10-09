#include "training/trainer/trainer.h"
#include <stdexcept>
#include <algorithm>
#include <iostream>

Trainer::Trainer(Model* model, Loss* loss, Optimizer* optimizer, const TrainerConfig& config)
  : m_model(model), m_loss(loss), m_optimizer(optimizer), m_config(config), m_rng(std::random_device{}()) {}

Trainer::~Trainer() = default;

Tensor Trainer::stack_batch(const std::vector<Tensor>& samples, size_t start, size_t end) {
  if (start >= end || end > samples.size()) throw std::out_of_range("Invalid batch indices");
  size_t batch_size = end - start;
  const std::vector<size_t>& sample_shape = samples[start].shape();
  std::vector<size_t> batch_shape = sample_shape;
  batch_shape.insert(batch_shape.begin(), batch_size); // Prepend batch size

  Tensor batch(batch_shape);
  float* batch_data = batch.data();
  size_t sample_numel = samples[start].numel();

  for (size_t i = 0; i < batch_size; ++i) {
    if (samples[start + i].shape() != sample_shape) throw std::invalid_argument("Inconsistent sample shapes in batch");
    std::copy(samples[start + i].data(), samples[start + i].data() + sample_numel, batch_data + i * sample_numel);
  }
  return batch;
}

float Trainer::train_batch(const std::vector<Tensor>& batch_inputs, const std::vector<Tensor>& batch_targets) {
  if (batch_inputs.size() != batch_targets.size()) throw std::invalid_argument("Mismatched batch input and target sizes");
  size_t batch_size = batch_inputs.size();
  Tensor inputs = stack_batch(batch_inputs, 0, batch_size);
  Tensor targets = stack_batch(batch_targets, 0, batch_size);

  m_optimizer->zero_grad();
  // Forward pass
  Tensor predictions = m_model->forward(inputs);
  float loss_value = m_loss->forward(predictions, targets);

  // Backward pass
  Tensor loss_grad = m_loss->backward(predictions, targets);
  m_model->backward(loss_grad);

  m_optimizer->step(); // Update parameters

  return loss_value;
}

void Trainer::train(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets) {
  if (inputs.size() != targets.size()) throw std::invalid_argument("Mismatched input and target sizes");
  size_t dataset_size = inputs.size();
  size_t batch_size = m_config.batch_size;
  size_t num_batches = (dataset_size + batch_size - 1) / batch_size;

  std::vector<size_t> indices(dataset_size);
  for (size_t i = 0; i < dataset_size; ++i) indices[i] = i;

  for (size_t epoch = 0; epoch < m_config.epochs; ++epoch) {
    if (m_config.shuffle) std::shuffle(indices.begin(), indices.end(), m_rng);

    float epoch_loss = 0.0f;
    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      size_t start = batch_idx * batch_size;
      size_t end = std::min(start + batch_size, dataset_size);

      std::vector<Tensor> batch_inputs, batch_targets;
      for (size_t i = start; i < end; ++i) {
        batch_inputs.push_back(inputs[indices[i]]);
        batch_targets.push_back(targets[indices[i]]);
      }

      float batch_loss = train_batch(batch_inputs, batch_targets);
      epoch_loss += batch_loss;
    }
    epoch_loss /= num_batches;
    std::cout << "Epoch " << (epoch + 1) << "/" << m_config.epochs << ", Loss: " << epoch_loss << std::endl;
  }
}