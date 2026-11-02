#include "mc_gate_seed.h"
#include "mc_gate_dgp.h"
#include "mc_gate_panel.h"
#include "mc_gate_baselines.h"
#include "mc_gate_adapter.h"
#include "mc_gate_output.h"
#include "mc_gate_replay.h"
#include "mc_gate_grid_cells.inc"

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <span>
#include <cstdlib>

static const char* comp_name(mc_gate::Composition cc) {
    switch (cc) {
        case mc_gate::Composition::Null:        return "null";
        case mc_gate::Composition::OneGood:     return "one_good";
        case mc_gate::Composition::Mixed:       return "mixed";
        case mc_gate::Composition::MixedSigned: return "mixed_signed";
        case mc_gate::Composition::AllNegative: return "all_negative";
    }
    return "null";
}

int main(int argc, char** argv) {
    using namespace mc_gate;

    int limit = 269;
    int cap = 0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--limit") == 0 && i + 1 < argc) {
            limit = std::atoi(argv[i + 1]);
            ++i;
        } else if (std::strcmp(argv[i], "--reps-cap") == 0 && i + 1 < argc) {
            cap = std::atoi(argv[i + 1]);
            ++i;
        }
    }

    const int total = static_cast<int>(sizeof(kGridCells) / sizeof(kGridCells[0]));
    long long ac41_ok = 0;
    if (limit > total) limit = total;
    if (limit < 0) limit = 0;

    std::printf("GRID total=%d limit=%d\n", total, limit);

    FILE* fa = std::fopen("/tmp/qgate/C27/aggregates.jsonl", "wb");
    std::vector<OutputRecord> outs;

    for (int k = 0; k < limit; ++k) {
        const GridCell& c = kGridCells[k];
        int reps = (cap > 0 && c.reps > cap) ? cap : c.reps;
        int sel[6] = {0, 0, 0, 0, 0, 0};
        std::uint64_t fp = 0;

        for (int rep = 0; rep < reps; ++rep) {
            Family fam = static_cast<Family>(c.family);
            Composition cmp = static_cast<Composition>(c.comp);
            std::string key = dataset_key(comp_name(cmp), c.S, c.N, c.T, c.rho, c.dist, c.p, c.ds);

            ReturnMatrix m = generate_panel(fam, cmp, c.T, c.N, c.rho, c.S, key, rep, 0, 20261027ULL);

            std::vector<double> flat(static_cast<std::size_t>(c.T) * static_cast<std::size_t>(c.N));
            for (int i = 0; i < c.T; ++i) {
                for (int t = 0; t < c.N; ++t) {
                    flat[static_cast<std::size_t>(i) * static_cast<std::size_t>(c.N) + static_cast<std::size_t>(t)] = m.at(i, t);
                }
            }

            ResearchWindow w{ std::span<const double>(flat.data(), flat.size()),
                              static_cast<std::size_t>(c.T), static_cast<std::size_t>(c.N), fp };
            WindowStats st = make_window_stats(w, false);
            // AC-41：维度绑定断言（抓朝向转置）——observations==N、trials==T、sr 长度==T
            if (st.T != c.T || st.N != c.N || st.sr.size() != static_cast<std::size_t>(c.T)) {
                std::printf("AC41_FAIL cell=%d stats.T=%d want=%d stats.N=%d want=%d sr=%zu\n",
                            c.cell_id, st.T, c.T, st.N, c.N, st.sr.size());
                return 2;
            }
            ++ac41_ok;

            Decision d = decide_from_stats(st, RuleConfig{ GateMode::Full, 0.05, 0.05 });
            RuleResult r1 = rule_b0(m);
            RuleResult r2 = rule_b1(m, 0.05);
            RuleResult r3 = rule_b2(m);
            RuleResult r4 = rule_b3(m);
            RuleResult r5 = rule_b4_cscv(m, 6, 0.5);

            if (d.selected) sel[0]++;
            if (r1.selected) sel[1]++;
            if (r2.selected) sel[2]++;
            if (r3.selected) sel[3]++;
            if (r4.selected) sel[4]++;
            if (r5.selected) sel[5]++;

            auto push1 = [&](const char* id, bool sel_flag, long long idx, bool ab, std::string rsn) {
                OutputRecord o{};
                o.cell_id = c.cell_id;
                o.rep_id = rep;
                o.rule_id = id;
                o.selected = sel_flag;
                o.index = idx;
                o.abstain = ab;
                o.returns_sha256 = fp;
                o.reason = std::move(rsn);
                outs.push_back(o);
            };

            push1("GATE_Full", d.selected, static_cast<long long>(d.index), d.abstain,
                  d.reason ? std::string(d.reason) : std::string("selected"));
            push1("B0", r1.selected, static_cast<long long>(r1.index), false, "baseline");
            push1("B1", r2.selected, static_cast<long long>(r2.index), false, "baseline");
            push1("B2", r3.selected, static_cast<long long>(r3.index), false, "baseline");
            push1("B3", r4.selected, static_cast<long long>(r4.index), false, "baseline");
            push1("B4", r5.selected, static_cast<long long>(r5.index), false, "baseline");
        }

        const char* rule_names[6] = {"GATE_Full", "B0", "B1", "B2", "B3", "B4"};
        for (int i = 0; i < 6; ++i) {
            std::fprintf(fa, "{\"cell_id\":%d,\"rule\":\"%s\",\"reps\":%d,\"selected\":%d}\n",
                         c.cell_id, rule_names[i], reps, sel[i]);
        }

        std::printf("CELL %d %d %d\n", c.cell_id, reps, reps * 6);
    }

    std::fclose(fa);

    write_decisions_jsonl_gz("/tmp/qgate/C27/decisions.jsonl.gz", outs);

    std::printf("AGG_CELLS %d\n", limit * 6);
    std::printf("RECORDS %zu\n", outs.size());

    std::printf("AC41 ok=%lld\n", ac41_ok);
    std::printf("GRID_DONE\n");

    return 0;
}
