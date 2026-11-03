// AC-08：τ 回收（单格 |z|≤4.5，全局 Σz² ≤ chi2_202(0.99)=251.7）
#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include "mc_gate_grid_cells.inc"
#include "mc_gate_seed.h"
#include "mc_gate_dgp.h"
#include "mc_gate_streams.h"
#include "mc_gate_baselines.h"

static const char* comp_name(int c) {
    switch (c) {
        case 0: return "null";
        case 1: return "one_good";
        case 2: return "mixed";
        case 3: return "mixed_signed";
        case 4: return "all_negative";
    }
    return "null";
}
static double c_kappa(int fam) {           // §3.5
    switch (fam) {
        case 0: return 0.5;                // gauss  kappa=0
        case 1: return 2.0;                // t5     kappa=6
        case 2: return 3.5;                // garch  kappa=12（保守）
        default: return 4.25;              // shortvol kappa=15（保守）
    }
}
int main() {
    using namespace mc_gate;
    constexpr std::uint64_t run_seed = 20261027ULL;
    constexpr int REPS = 200;
    const int total = static_cast<int>(sizeof(kGridCells) / sizeof(kGridCells[0]));

    std::unordered_map<std::string, int> done;
    double sum_z2 = 0.0;
    double max_abs_z = 0.0;
    int panels = 0;
    int bad_cell = 0;

    for (int k = 0; k < total; ++k) {
        const GridCell& c = kGridCells[k];
        const std::string key = dataset_key(comp_name(c.comp), c.S, c.N, c.T, c.rho, c.dist, c.p, c.ds);
        if (done.count(key)) { continue; }   // §5.5：同 panel_ref 只算一次
        done[key] = 1;
        ++panels;

        const Family fam = static_cast<Family>(c.family);
        const Composition comp = static_cast<Composition>(c.comp);
        double sum_sr = 0.0;
        for (int rep = 0; rep < REPS; ++rep) {
            const int kappa = kappa_of(key, rep, 0, run_seed, c.T);
            ReturnMatrix m = generate_with_kappa(fam, comp, c.T, c.N, c.rho, c.S,
                                                 seed_of(key, rep, 0, "ret", run_seed), kappa);
            sum_sr += strategy_sr(m, kappa);
        }
        const double mean_sr = sum_sr / REPS;
        const double var_single = (c.p + c_kappa(c.family) * c.S * c.S) / static_cast<double>(c.N);
        const double se = std::sqrt(var_single / static_cast<double>(REPS));
        const double z = (mean_sr - c.S) / se;
        sum_z2 += z * z;
        if (std::fabs(z) > max_abs_z) max_abs_z = std::fabs(z);
        const int ok = (std::fabs(z) <= 4.5) ? 1 : 0;
        if (!ok) ++bad_cell;
        std::printf("TRUTH cell=%d S=%.4f mean_sr=%.6f se=%.6f z=%.3f ok=%d\n",
                    c.cell_id, c.S, mean_sr, se, z, ok);
    }
    std::printf("TRUTH_PANELS %d sum_z2=%.2f limit=251.7 ok=%d\n",
                panels, sum_z2, (sum_z2 <= 251.7) ? 1 : 0);
    std::printf("TRUTH_MAXZ %.3f limit=4.5 bad_cells=%d ok=%d\n",
                max_abs_z, bad_cell, (bad_cell == 0) ? 1 : 0);
    std::printf("AC08_OK %d\n", (bad_cell == 0 && sum_z2 <= 251.7 && panels > 0) ? 1 : 0);
    std::printf("TRUTH_DONE\n");
    return 0;
}
