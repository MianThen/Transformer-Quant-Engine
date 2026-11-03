// mc_gate_wall.h
#ifndef MC_GATE_WALL_H
#define MC_GATE_WALL_H

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>

namespace mc_gate {

struct RuleWall {
  const char* rule_id;
  double ms;
};

inline std::string wall_by_rule_json(const std::vector<RuleWall>& walls) {
  static const char* kKeys[6] = {
    "GATE_Full", "B0", "B1", "B2", "B3", "B4"
  };

  double acc[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  for (const auto& w : walls) {
    for (int i = 0; i < 6; ++i) {
      if (std::strcmp(w.rule_id, kKeys[i]) == 0) {
        acc[i] = w.ms;
        break;
      }
    }
  }

  char buf[256];
  std::snprintf(buf, sizeof(buf),
    "{\"GATE_Full\":%.3f,\"B0\":%.3f,\"B1\":%.3f,\"B2\":%.3f,\"B3\":%.3f,\"B4\":%.3f}",
    acc[0], acc[1], acc[2], acc[3], acc[4], acc[5]);

  return std::string(buf);
}

} // namespace mc_gate

#ifdef MC_GATE_WALL_SELFTEST
int main() {
  using namespace mc_gate;

  {
    std::vector<RuleWall> walls = {
      { "B2", 3.456 },
      { "GATE_Full", 12.345 },
      { "B0", 1.234 },
      { "B4", 5.678 },
      { "B1", 2.345 },
      { "B3", 4.567 }
    };
    std::printf("WALL_JSON %s\n", wall_by_rule_json(walls).c_str());
  }

  {
    std::vector<RuleWall> walls = {
      { "B0", 9.999 },
      { "UNKNOWN_RULE", 7.777 }
    };
    std::printf("WALL_JSON_PARTIAL %s\n", wall_by_rule_json(walls).c_str());
  }

  std::printf("WALL_DONE\n");
  return 0;
}
#endif // MC_GATE_WALL_SELFTEST

#endif // MC_GATE_WALL_H
