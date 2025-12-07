#pragma once
#include <vector>

class QTable {
public:
  QTable(const size_t states_num, size_t actions_num);
  ~QTable() = default;

  float get(size_t state, size_t action);
  void set(size_t state, size_t action, float value);

private:
  std::vector<float> m_data;
  size_t states_num;
  size_t actions_num;
};