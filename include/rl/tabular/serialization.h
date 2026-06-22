#pragma once

#include "q_table.h"

#include <string>

inline bool save_q_table(const QTable& table, const std::string& path) {
  return table.save(path);
}

inline bool load_q_table(QTable& table, const std::string& path) {
  return table.load(path);
}
