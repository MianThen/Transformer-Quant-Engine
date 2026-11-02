#ifndef MC_GATE_SEEDREC_H
#define MC_GATE_SEEDREC_H

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "mc_gate_seed.h"
#include "mc_gate_rng.h"
#include "mc_gate_sha256.h"

namespace mc_gate {

struct RunConfig {
  std::string contract_version;
  std::string grid;
  std::uint64_t run_seed;
  double q;
  double alpha;
  int reps_scale;
};

inline std::string config_canonical_json(const RunConfig& c) {
  char buf[512];
  std::snprintf(buf, sizeof(buf),
                "{\"contract_version\":\"%s\",\"grid\":\"%s\",\"run_seed\":%llu,"
                "\"q\":%.6f,\"alpha\":%.6f,\"reps_scale\":%d}",
                c.contract_version.c_str(),
                c.grid.c_str(),
                static_cast<unsigned long long>(c.run_seed),
                c.q,
                c.alpha,
                c.reps_scale);
  return std::string(buf);
}

inline std::string config_hash(const RunConfig& c) {
  std::string canonical = config_canonical_json(c);
  return sha256_hex(canonical.data(), canonical.size());
}

inline std::string stream_fingerprint(const std::string& dataset_key, int rep, int batch,
                                      const char* stream, std::uint64_t run_seed) {
  Pcg64 rng(seed_of(dataset_key, rep, batch, stream, run_seed));
  std::string a = hex16(rng.next());
  std::string b = hex16(rng.next());
  std::string c = hex16(rng.next());
  return a + " " + b + " " + c;
}

}  // namespace mc_gate

#ifdef MC_GATE_SEEDREC_SELFTEST
int main() {
  using namespace mc_gate;

  const std::string key = "v1|null|S=0.0000|N=126|T=4|rho=0.30|gauss|p=50.4|ds=0";
  const std::uint64_t run_seed = 20261027ULL;
  const int rep = 0;
  const int batch = 0;

  RunConfig cfg{"1.4.5", "mc_gate_core_v1", run_seed, 0.05, 0.05, 1};

  std::printf("SEEDREC_BEGIN\n");
  std::printf("FP ret %s\n", stream_fingerprint(key, rep, batch, "ret", run_seed).c_str());
  std::printf("FP assign %s\n", stream_fingerprint(key, rep, batch, "assign", run_seed).c_str());
  std::printf("FP hold %s\n", stream_fingerprint(key, rep, batch, "hold", run_seed).c_str());
  std::printf("FP boot %s\n", stream_fingerprint(key, rep, batch, "boot", run_seed).c_str());
  std::printf("FP aux %s\n", stream_fingerprint(key, rep, batch, "aux", run_seed).c_str());
  std::printf("CANON %s\n", config_canonical_json(cfg).c_str());
  std::printf("CONFIG_HASH %s\n", config_hash(cfg).c_str());

  RunConfig tampered = cfg;
  tampered.run_seed = 20261028ULL;
  std::printf("TAMPER_HASH %s\n", config_hash(tampered).c_str());

  std::printf("SEEDREC_DONE\n");
  return 0;
}
#endif  // MC_GATE_SEEDREC_SELFTEST

#endif  // MC_GATE_SEEDREC_H
