#include "rl/q-table.h"
#include <stdexcept>

QTable::QTable(const size_t states_num, const size_t actions_num) {
  if(states_num <= 0 || actions_num <= 0) throw std::invalid_argument("Invalid Size");

  this->states_num = states_num;
  this->actions_num = actions_num;
  m_data.resize(states_num * actions_num, 0.0f);
}

float QTable::get(size_t state, size_t action) {
  return m_data[state * actions_num + action];
}

void QTable::set(size_t state, size_t action, float value) {
  m_data[state * actions_num + action] = value;
}