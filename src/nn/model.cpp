#include "nn/model.h"

Model::~Model() {
  for (Layer* layer : m_layers) delete layer;
}

void Model::add_layer(Layer* layer) {
  m_layers.push_back(layer);
}

Tensor Model::forward(const Tensor& input) {
  Tensor output = input;
  for (Layer* layer : m_layers) output = layer->forward(output);
  return output;
}

Tensor Model::backward(const Tensor& grad_output) {
  Tensor grad = grad_output;
  for(int i = m_layers.size() - 1; i >= 0; --i) {
    grad = m_layers[i]->backward(grad);
  }
  return grad;
}

void Model::update(float learning_rate) {
  for (Layer* layer : m_layers) layer->update(learning_rate);
}

void Model::clear_state() {
  for (Layer* layer : m_layers) layer->zero_grad();
}

std::vector<Layer*>& Model::layers() {
  return m_layers;
}