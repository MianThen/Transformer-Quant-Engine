#include "mc_gate_seed.h"
#include "mc_gate_dgp.h"
#include "mc_gate_panel.h"
#include "mc_gate_streams.h"
#include "mc_gate_seedrec.h"
#include "mc_gate_baselines.h"
#include "mc_gate_adapter.h"
#include "mc_gate_manifest.h"
#include "mc_gate_cost.h"
#include "mc_gate_output.h"
#include "mc_gate_replay.h"

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <span>
#include <chrono>
#include <algorithm>
#include <cmath>

using namespace mc_gate;

int main() {
    // Smoke config
    RunConfig cfg{"1.5.0", "mc_gate_core_v1", 20261027ULL, 0.05, 0.05, 1};

    // Cell definitions
    struct CellDef {
        int T, N;
        Family fam;
        Composition comp;
        double S;
        const char* comp_name;
        long long cell_id;
    };

    CellDef cells[4] = {
        {4, 126, Family::Gauss, Composition::Null, 0.0, "null", 2001},
        {20, 266, Family::T5, Composition::OneGood, 0.6, "one_good", 2002},
        {100, 126, Family::T5, Composition::Null, 0.0, "null", 2003},
        {100, 126, Family::T5, Composition::OneGood, 0.6, "one_good", 2004},
    };

    // Budget units
    std::vector<BudgetUnit> units = {
        {4, 126, 8000, 1},
        {20, 266, 2000, 1},
        {100, 126, 1000, 1},
        {100, 126, 1000, 1},
    };

    // Cost points
    std::vector<CostPoint> cost_pts = {
        {4, 126, 0.1375, 0.0609, 0.0471, 0.0025, 0.0270},
        {20, 266, 1.2732, 0.6371, 0.5542, 0.0386, 0.0434},
        {100, 126, 2.7002, 1.5205, 0.7427, 0.4101, 0.0270},
    };

    // Step 1: Planning and Cost
    std::vector<double> cost_ms(4);
    for (int i = 0; i < 4; ++i) {
        // Find matching cost point
        for (const auto& cp : cost_pts) {
            if (cp.T == units[i].T && cp.N == units[i].N) {
                cost_ms[i] = cp.ch_ms;
                break;
            }
        }
    }

    double threshold_ms = 1.0;
    std::vector<BudgetUnit> planned = plan_units(units, cost_ms, threshold_ms);

    for (int i = 0; i < 4; ++i) {
        std::printf("PLAN %d %d\n", i, planned[i].reps);
    }

    long long orig_calls = total_calls(units);
    long long planned_calls = total_calls(planned);
    std::printf("CALLS orig=%lld planned=%lld\n", orig_calls, planned_calls);

    std::vector<CostWeight> weights(4);
    for (int i = 0; i < 4; ++i) {
        weights[i] = {units[i].T, units[i].N, units[i].reps};
    }

    CostProfile prof = summarise_cost(cost_pts, weights, threshold_ms);
    std::printf("COST total %.3f over %d\n", prof.single_core_ms, prof.over_threshold);

    // Step 2: Execution
    std::vector<OutputRecord> outs;
    double wall[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // AC-35：逐 cell 收集 m 与最大样本 SR，用于双轨门槛
    double m_sum[4] = {0.0, 0.0, 0.0, 0.0};
    long long n_rep[4] = {0, 0, 0, 0};
    std::vector<std::vector<double>> maxsr(4);

    for (int k = 0; k < 4; ++k) {
        const auto& c = cells[k];
        for (int rep = 0; rep < planned[k].reps; ++rep) {
            std::string key = dataset_key(c.comp_name, c.S, c.N, c.T, 0.30, "gauss", 50.4, 0);
            ReturnMatrix m = generate_panel(c.fam, c.comp, c.T, c.N, 0.30, c.S, key, rep, 0, cfg.run_seed);

            // Construct ResearchWindow
            std::vector<double> flat(c.T * c.N);
            for (int i = 0; i < c.T; ++i) {
                for (int t = 0; t < c.N; ++t) {
                    flat[i * c.N + t] = m.at(i, t);
                }
            }
            ResearchWindow w{std::span<const double>(flat.data(), flat.size()), (std::size_t)c.T, (std::size_t)c.N, 0};

            // Rule 0: GATE
            {
                auto t0 = std::chrono::steady_clock::now();
                WindowStats ws = make_window_stats(w, false);
                Decision d = decide_from_stats(ws, RuleConfig{GateMode::Full, 0.05, 0.05});
                auto t1 = std::chrono::steady_clock::now();
                m_sum[k] += ws.m;
                ++n_rep[k];
                if (!ws.sr.empty()) {
                    maxsr[k].push_back(*std::max_element(ws.sr.begin(), ws.sr.end()));
                }
                wall[0] += std::chrono::duration<double, std::milli>(t1 - t0).count();

                OutputRecord rec;
                rec.cell_id = c.cell_id;
                rec.rep_id = rep;
                rec.rule_id = "GATE_Full";
                rec.selected = d.selected;
                rec.index = (long long)d.index;
                rec.abstain = d.abstain;
                rec.returns_sha256 = 0;
                rec.reason = d.reason ? d.reason : "selected";
                outs.push_back(rec);
            }

            // Rule 1: B0
            {
                auto t0 = std::chrono::steady_clock::now();
                RuleResult rr = rule_b0(m);
                auto t1 = std::chrono::steady_clock::now();
                wall[1] += std::chrono::duration<double, std::milli>(t1 - t0).count();

                OutputRecord rec;
                rec.cell_id = c.cell_id;
                rec.rep_id = rep;
                rec.rule_id = "B0";
                rec.selected = rr.selected;
                rec.index = (long long)rr.index;
                rec.abstain = false;
                rec.returns_sha256 = 0;
                rec.reason = "baseline";
                outs.push_back(rec);
            }

            // Rule 2: B1
            {
                auto t0 = std::chrono::steady_clock::now();
                RuleResult rr = rule_b1(m, 0.05);
                auto t1 = std::chrono::steady_clock::now();
                wall[2] += std::chrono::duration<double, std::milli>(t1 - t0).count();

                OutputRecord rec;
                rec.cell_id = c.cell_id;
                rec.rep_id = rep;
                rec.rule_id = "B1";
                rec.selected = rr.selected;
                rec.index = (long long)rr.index;
                rec.abstain = false;
                rec.returns_sha256 = 0;
                rec.reason = "baseline";
                outs.push_back(rec);
            }

            // Rule 3: B2
            {
                auto t0 = std::chrono::steady_clock::now();
                RuleResult rr = rule_b2(m);
                auto t1 = std::chrono::steady_clock::now();
                wall[3] += std::chrono::duration<double, std::milli>(t1 - t0).count();

                OutputRecord rec;
                rec.cell_id = c.cell_id;
                rec.rep_id = rep;
                rec.rule_id = "B2";
                rec.selected = rr.selected;
                rec.index = (long long)rr.index;
                rec.abstain = false;
                rec.returns_sha256 = 0;
                rec.reason = "baseline";
                outs.push_back(rec);
            }

            // Rule 4: B3
            {
                auto t0 = std::chrono::steady_clock::now();
                RuleResult rr = rule_b3(m);
                auto t1 = std::chrono::steady_clock::now();
                wall[4] += std::chrono::duration<double, std::milli>(t1 - t0).count();

                OutputRecord rec;
                rec.cell_id = c.cell_id;
                rec.rep_id = rep;
                rec.rule_id = "B3";
                rec.selected = rr.selected;
                rec.index = (long long)rr.index;
                rec.abstain = false;
                rec.returns_sha256 = 0;
                rec.reason = "baseline";
                outs.push_back(rec);
            }

            // Rule 5: B4
            {
                auto t0 = std::chrono::steady_clock::now();
                RuleResult rr = rule_b4_cscv(m, 6, 0.5);
                auto t1 = std::chrono::steady_clock::now();
                wall[5] += std::chrono::duration<double, std::milli>(t1 - t0).count();

                OutputRecord rec;
                rec.cell_id = c.cell_id;
                rec.rep_id = rep;
                rec.rule_id = "B4";
                rec.selected = rr.selected;
                rec.index = (long long)rr.index;
                rec.abstain = false;
                rec.returns_sha256 = 0;
                rec.reason = "baseline";
                outs.push_back(rec);
            }
        }
    }

    write_decisions_jsonl_gz("/tmp/qgate/C26/decisions.jsonl.gz", outs);
    std::printf("RECORDS %zu\n", outs.size());
    std::printf("WALL %.3f %.3f %.3f %.3f %.3f %.3f\n", wall[0], wall[1], wall[2], wall[3], wall[4], wall[5]);

    // Step 3: Persistence
    Manifest mf;
    mf.contract_hash = std::string(64, 'a');
    mf.binary_sha256 = std::string(64, 'c');
    mf.git_commit = std::string(40, 'd');
    mf.config_hash = config_hash(cfg);

    std::string mjson = manifest_json(mf);
    mjson.pop_back();
    mjson += ",\"run_seed\":" + std::to_string((unsigned long long)cfg.run_seed);
    mjson += ",\"seed_tree\":{";

    // Build seed_tree
    for (int k = 0; k < 4; ++k) {
        const auto& c = cells[k];
        std::string key = dataset_key(c.comp_name, c.S, c.N, c.T, 0.30, "gauss", 50.4, 0);
        if (k > 0) mjson += ",";
        mjson += "\"" + key + "\":{";
        const char* streams[] = {"ret", "assign", "hold", "boot", "aux"};
        for (int s = 0; s < 5; ++s) {
            if (s > 0) mjson += ",";
            std::string fp = stream_fingerprint(key, 0, 0, streams[s], cfg.run_seed);
            mjson += "\"" + std::string(streams[s]) + "\":\"" + fp + "\"";
        }
        mjson += "}";
    }
    mjson += "}}";

    {
        FILE* f = std::fopen("/tmp/qgate/C26/manifest.json", "wb");
        if (f) {
            std::fwrite(mjson.data(), 1, mjson.size(), f);
            std::fclose(f);
        }
    }

    std::string mjs = metrics_json(mf, (long long)outs.size(), 4, planned[0].reps, 6, 0, 1.0, planned);
    mjs.pop_back();
    mjs += ",\"cost_profile\":" + prof.json;
    mjs += ",\"wall_by_rule\":{";
    const char* rule_names[] = {"GATE_Full", "B0", "B1", "B2", "B3", "B4"};
    for (int i = 0; i < 6; ++i) {
        if (i > 0) mjs += ",";
        char buf[64];
        std::snprintf(buf, sizeof(buf), "\"%s\":%.3f", rule_names[i], wall[i]);
        mjs += buf;
    }
    mjs += "}";

    // AC-35：双轨门槛（解析 vs 经验），偏差 > 0.05 SR 记 analytic_threshold_drift
    mjs += ",\"thresholds\":{";
    int drift_flags = 0;
    for (int k = 0; k < 4; ++k) {
        const auto& c = cells[k];
        const double mean_m = (n_rep[k] > 0) ? m_sum[k] / static_cast<double>(n_rep[k]) : 0.0;
        const double analytic = gate_threshold_sr(gate_teff(static_cast<double>(c.T), mean_m),
                                                  static_cast<double>(c.T), static_cast<double>(c.N), 0.05);
        double empirical = 0.0;
        int flagged = 0;
        const bool is_null = (c.comp == Composition::Null);
        if (is_null && maxsr[k].size() >= 2) {
            std::vector<double> v = maxsr[k];
            std::sort(v.begin(), v.end());
            const std::size_t idx = static_cast<std::size_t>(
                std::floor(0.95 * static_cast<double>(v.size() - 1)));
            empirical = v[idx];
            flagged = (std::fabs(empirical - analytic) > 0.05) ? 1 : 0;
            drift_flags += flagged;
        }
        char tb[256];
        std::snprintf(tb, sizeof(tb),
                      "%s\"%lld\":{\"kind\":\"%s\",\"analytic\":%.6f,\"empirical\":%.6f,\"drift\":%.6f,\"flagged\":%d}",
                      (k > 0 ? "," : ""), c.cell_id, (is_null ? "null" : "signal"),
                      analytic, empirical, empirical - analytic, flagged);
        mjs += tb;
    }
    mjs += "}";

    // AC-35：偏差超限时把 analytic_threshold_drift 写进 flags
    mjs += ",\"flags\":[";
    if (drift_flags > 0) mjs += "\"analytic_threshold_drift\"";
    mjs += "]";

    // AC-24：预算守约（单核总量 vs 45 min 预算；降档后同样核对）
    {
        double degraded_ms = 0.0;
        for (const auto& pu : planned) {
            for (const auto& cp : cost_pts) {
                if (cp.T == pu.T && cp.N == pu.N) {
                    degraded_ms += cp.ch_ms * static_cast<double>(pu.reps) * static_cast<double>(pu.calls_per_rep);
                }
            }
        }
        const double budget_ms = 45.0 * 60.0 * 1000.0;   // AC-24 的 45 分钟
        char bb[320];
        std::snprintf(bb, sizeof(bb),
                      ",\"budget\":{\"single_core_ms\":%.3f,\"degraded_ms\":%.3f,"
                      "\"budget_ms\":%.1f,\"over_budget\":%d,\"drift_flags\":%d}",
                      prof.single_core_ms, degraded_ms, budget_ms,
                      (degraded_ms > budget_ms) ? 1 : 0, drift_flags);
        mjs += bb;
    }
    mjs += "}";

    {
        FILE* f = std::fopen("/tmp/qgate/C26/metrics.json", "wb");
        if (f) {
            std::fwrite(mjs.data(), 1, mjs.size(), f);
            std::fclose(f);
        }
    }

    // Replay dataset for cell 0, rep 0
    {
        const auto& c = cells[0];
        std::string key = dataset_key(c.comp_name, c.S, c.N, c.T, 0.30, "gauss", 50.4, 0);
        ReturnMatrix m0 = generate_panel(c.fam, c.comp, c.T, c.N, 0.30, c.S, key, 0, 0, cfg.run_seed);
        write_replay_dataset("/tmp/qgate/C26/replay", 0, m0.r);
    }

    auto exists = [](const char* p) {
        FILE* f = std::fopen(p, "rb");
        if (f) {
            std::fclose(f);
            return 1;
        }
        return 0;
    };

    std::printf("FILES %d %d %d %d\n",
                exists("/tmp/qgate/C26/manifest.json"),
                exists("/tmp/qgate/C26/metrics.json"),
                exists("/tmp/qgate/C26/decisions.jsonl.gz"),
                exists("/tmp/qgate/C26/replay/dataset_0.bin"));

    // Step 4
    std::printf("RUN_DONE\n");
    return 0;
}
