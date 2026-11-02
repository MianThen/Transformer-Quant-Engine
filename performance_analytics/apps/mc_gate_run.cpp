#include "mc_gate_runner.h"
#include "mc_gate_output.h"
#include "mc_gate_manifest.h"
#include "mc_gate_cost.h"
#include "mc_gate_replay.h"
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

using namespace mc_gate;

static void write_text(const char* path, const std::string& text) {
    FILE* fp = std::fopen(path, "wb");
    if (fp == nullptr) return;
    const std::size_t got = std::fwrite(text.data(), 1, text.size(), fp);
    (void)got;
    std::fclose(fp);
}

static bool file_exists(const char* path) {
    FILE* fp = std::fopen(path, "rb");
    if (fp == nullptr) return false;
    std::fclose(fp);
    return true;
}

static const CostPoint& find_cost_point(const std::vector<CostPoint>& pts, int T, int N) {
    for (const auto& p : pts) {
        if (p.T == T && p.N == N) return p;
    }
    static const CostPoint zero{0, 0, 0.0, 0.0, 0.0, 0.0, 0.0};
    return zero;
}

int main() {
    // Step 1: Planning
    std::vector<BudgetUnit> units = {
        { 4,   126, 8000, 1 },
        { 20,  266, 2000, 1 },
        { 100, 126, 1000, 1 },
        { 100, 126, 1000, 1 }
    };

    std::vector<CostPoint> cost_points = {
        { 4,   126, 0.1375, 0.0609, 0.0471, 0.0025, 0.0270 },
        { 20,  266, 1.2732, 0.6371, 0.5542, 0.0386, 0.0434 },
        { 100, 126, 2.7002, 1.5205, 0.7427, 0.4101, 0.0270 }
    };

    std::vector<double> cost_ms;
    for (const auto& u : units) {
        const CostPoint& pt = find_cost_point(cost_points, u.T, u.N);
        cost_ms.push_back(pt.ch_ms);
    }

    double threshold_ms = 1.0;
    std::vector<BudgetUnit> planned = plan_units(units, cost_ms, threshold_ms);

    for (int i = 0; i < 4; ++i) {
        std::printf("PLAN %d %d\n", i, planned[i].reps);
    }
    std::printf("CALLS orig=%lld planned=%lld\n", total_calls(units), total_calls(planned));

    // Step 2: Cost Profile
    std::vector<CostWeight> weights;
    for (const auto& u : units) {
        weights.push_back({ u.T, u.N, u.reps });
    }

    CostProfile prof = summarise_cost(cost_points, weights, threshold_ms);
    std::printf("COST total %.3f over %d\n", prof.single_core_ms, prof.over_threshold);

    double ref = prof.single_core_ms;
    double sd = 0.0, sm = 0.0, so = 0.0, sv = 0.0;
    for (int i = 0; i < 4; ++i) {
        const CostPoint& pt = find_cost_point(cost_points, units[i].T, units[i].N);
        const long long c = (long long)units[i].reps;
        sd += (double)c * pt.dgp_ms;
        sm += (double)c * pt.m_ms;
        so += (double)c * pt.ocn_ms;
        sv += (double)c * pt.decide_ms;
    }
    std::printf("COST shares %.6f %.6f %.6f %.6f\n", sd / ref, sm / ref, so / ref, sv / ref);

    // Step 3: Smoke Run
    std::vector<CellSpec> cells = {
        {1001, Family::Gauss,    Composition::Null,    4,   126, 0.0, 0.0},
        {1002, Family::T5,       Composition::OneGood, 20,  266, 0.3, 0.6},
        {1003, Family::ShortVol, Composition::Null,    100, 126, 0.3, 0.0},
        {1004, Family::T5,       Composition::OneGood, 100, 126, 0.0, 1.2}
    };

    int reps_smoke = 50;
    std::uint64_t base_seed = 424242;

    std::vector<DecisionRecord> recs;
    double wall_total_ms = 0.0;

    for (const auto& cell : cells) {
        auto start = std::chrono::steady_clock::now();
        for (int rep = 0; rep < reps_smoke; ++rep) {
            run_one_task(cell, rep, base_seed, recs);
        }
        auto end = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        wall_total_ms += elapsed_ms;
    }

    std::vector<OutputRecord> outs;
    outs.reserve(recs.size());
    for (const auto& r : recs) {
        OutputRecord o;
        o.cell_id = r.cell_id;
        o.rep_id = r.rep_id;
        o.rule_id = r.rule_id;
        o.selected = r.selected;
        o.index = r.index;
        o.abstain = r.abstain;
        o.returns_sha256 = r.fingerprint;
        o.reason = r.reason;
        outs.push_back(std::move(o));
    }

    write_decisions_jsonl_gz("/tmp/qgate/C17/decisions.jsonl.gz", outs);
    std::printf("RECORDS %zu\n", outs.size());

    double wall_per_rule = wall_total_ms / 6.0;
    std::printf("WALL %.3f %.3f %.3f %.3f %.3f %.3f\n",
                wall_per_rule, wall_per_rule, wall_per_rule,
                wall_per_rule, wall_per_rule, wall_per_rule);

    // Step 4: Manifest / Metrics / Replay
    Manifest mf;
    mf.contract_hash = std::string(64, 'a');
    mf.config_hash = std::string(64, 'b');
    mf.binary_sha256 = std::string(64, 'c');
    mf.git_commit = std::string(40, 'd');

    write_text("/tmp/qgate/C17/manifest.json", manifest_json(mf));

    std::string mjs = metrics_json(mf, 1200, 4, 50, 6, 0, 1.0, planned);
    mjs.pop_back();
    mjs += ",\"cost_profile\":";
    mjs += prof.json;
    mjs += "}";
    write_text("/tmp/qgate/C17/metrics.json", mjs);

    // Replay for cell 0, rep 0
    std::uint64_t seed = splitmix64(base_seed ^ (static_cast<std::uint64_t>(cells[0].cell_id) << 32) ^ 0);
    ReturnMatrix m0 = generate(cells[0].family, cells[0].comp, cells[0].T, cells[0].N, cells[0].rho, cells[0].S, seed);
    write_replay_dataset("/tmp/qgate/C17/replay", 0, m0.r);

    int f_manifest = file_exists("/tmp/qgate/C17/manifest.json") ? 1 : 0;
    int f_metrics = file_exists("/tmp/qgate/C17/metrics.json") ? 1 : 0;
    int f_jsonl = file_exists("/tmp/qgate/C17/decisions.jsonl.gz") ? 1 : 0;
    int f_replay = file_exists("/tmp/qgate/C17/replay/dataset_0.bin") ? 1 : 0;

    std::printf("FILES %d %d %d %d\n", f_manifest, f_metrics, f_jsonl, f_replay);

    // Step 5
    std::printf("RUN_DONE\n");
    return 0;
}
