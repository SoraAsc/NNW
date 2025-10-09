#include "training/optimizer/optimizer_sgd.h"

SGD::SGD(Model& model, float learning_rate): m_model(model), m_learning_rate(learning_rate) {}

void SGD::step() {
  // Update model parameters using SGD
  m_model.update(m_learning_rate);
}

void SGD::zero_grad() {
  // Clear gradients in all layers
  m_model.clear_state();
}