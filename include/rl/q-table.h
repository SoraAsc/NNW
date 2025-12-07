#pragma once
#include <vector>

class QTable {
public:
  QTable(const size_t states_num, size_t actions_num);
  ~QTable() = default;

  float get(size_t state, size_t action);
  void set(size_t state, size_t action, float value);
  
  size_t get_states_num() const { return states_num; }
  size_t get_actions_num() const { return actions_num; }
  const std::vector<float>& get_data() const { return m_data; }

private:
  std::vector<float> m_data;
  size_t states_num;
  size_t actions_num;
};