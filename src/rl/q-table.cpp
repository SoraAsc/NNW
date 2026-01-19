#include "rl/q-table.h"
#include <stdexcept>
#include <cstdio>

QTable::QTable(const size_t states_num, const size_t actions_num) {
  if(states_num <= 0 || actions_num <= 0) throw std::invalid_argument("Invalid Size");

  this->states_num = states_num;
  this->actions_num = actions_num;
  m_data.resize(states_num * actions_num, 0.0f);
}

float QTable::get(size_t state, size_t action) {
  if (!is_valid_index(state, action)) {
    std::fprintf(stderr, "QTable::get() - invalid index (state=%zu, action=%zu)\n", state, action);
    return 0.0f;
  }

  return m_data[state * actions_num + action];
}

void QTable::set(size_t state, size_t action, float value) {
  if (!is_valid_index(state, action)) {
    std::fprintf(stderr, "QTable::set() - invalid index (state=%zu, action=%zu) - ignoring\n", state, action);
    return;
  }

  m_data[state * actions_num + action] = value;
}

bool QTable::is_valid_index(size_t state, size_t action) const {
  return state < states_num && action < actions_num;
}