#include "rl/q-table.h"
#include <stdexcept>
#include <cstdio>
#include <fstream>
#include <string>
#include <cstdint>
#include <cstring>

QTable::QTable(const size_t states_num, const size_t actions_num) {
  if(states_num <= 0 || actions_num <= 0) throw std::invalid_argument("Invalid Size");

  this->states_num = states_num;
  this->actions_num = actions_num;
  m_data.resize(states_num * actions_num, 0.0f);
}

bool QTable::save(const std::string& path) const {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) return false;

  // simple header: magic + states + actions
  const char magic[] = "QTABLEV1";
  const size_t magic_len = std::strlen(magic);
  ofs.write(magic, magic_len);

  uint64_t s = static_cast<uint64_t>(states_num);
  uint64_t a = static_cast<uint64_t>(actions_num);
  ofs.write(reinterpret_cast<const char*>(&s), sizeof(s));
  ofs.write(reinterpret_cast<const char*>(&a), sizeof(a));

  if (!m_data.empty())
    ofs.write(reinterpret_cast<const char*>(m_data.data()), m_data.size() * sizeof(float));

  return ofs.good();
}

bool QTable::load(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) return false;

  const char expected[] = "QTABLEV1";
  const size_t expected_len = std::strlen(expected);
  std::string read_magic(expected_len, '\0');
  ifs.read(&read_magic[0], expected_len);
  if (read_magic != expected) return false;

  uint64_t s = 0, a = 0;
  ifs.read(reinterpret_cast<char*>(&s), sizeof(s));
  ifs.read(reinterpret_cast<char*>(&a), sizeof(a));

  if (s == 0 || a == 0) return false;

  size_t new_states = static_cast<size_t>(s);
  size_t new_actions = static_cast<size_t>(a);
  std::vector<float> data(new_states * new_actions);
  ifs.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
  if (!ifs) return false;

  // commit
  states_num = new_states;
  actions_num = new_actions;
  m_data.swap(data);
  return true;
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