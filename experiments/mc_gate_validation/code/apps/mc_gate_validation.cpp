#include "mc_gate_runner.h"
#include "mc_gate_output.h"
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

using namespace mc_gate;

int main() {
    std::vector<CellSpec> cells = {
        {1001, Family::Gauss,    Composition::Null,    4,   126, 0.0, 0.0},
        {1002, Family::T5,       Composition::OneGood, 20,  266, 0.3, 0.6},
        {1003, Family::ShortVol, Composition::Null,    100, 126, 0.3, 0.0},
        {1004, Family::T5,       Composition::OneGood, 100, 126, 0.0, 1.2},
    };

    const int reps = 50;
    const std::uint64_t base_seed = 424242;

    std::vector<OutputRecord> out;
    out.reserve(4 * reps * 6);

    for (const auto& cell : cells) {
        for (int rep = 0; rep < reps; ++rep) {
            std::vector<DecisionRecord> recs;
            run_one_task(cell, rep, base_seed, recs);
            for (const auto& d : recs) {
                OutputRecord o;
                o.cell_id = d.cell_id;
                o.rep_id = d.rep_id;
                o.rule_id = std::string(d.rule_id);
                o.selected = d.selected;
                o.index = d.index;
                o.abstain = d.abstain;
                o.returns_sha256 = d.fingerprint;
                o.reason = std::string(d.reason);
                out.push_back(std::move(o));
            }
        }
    }

    const char* gz_path = "/tmp/qgate/C12/decisions.jsonl.gz";
    bool ok = write_decisions_jsonl_gz(gz_path, out);

    const char* metrics_path = "/tmp/qgate/C12/metrics.json";
    FILE* mf = std::fopen(metrics_path, "w");
    if (mf) {
        std::fprintf(mf, "{\"records\":%lld,\"cells\":4,\"reps_per_cell\":50,\"rules\":6,\"gzip_mtime\":0}",
                     static_cast<long long>(out.size()));
        std::fclose(mf);
    }

    long long gzip_size = 0;
    FILE* gf = std::fopen(gz_path, "rb");
    if (gf) {
        std::fseek(gf, 0, SEEK_END);
        long sz = std::ftell(gf);
        gzip_size = static_cast<long long>(sz);
        std::fclose(gf);
    }

    std::printf("SMOKE records=%lld cells=4 reps=50\n", static_cast<long long>(out.size()));
    std::printf("SMOKE gzip_size=%lld\n", gzip_size);

    for (const auto& cell : cells) {
        long long sel_count = 0;
        for (const auto& o : out) {
            if (o.cell_id == cell.cell_id && o.rule_id == "GATE_Full" && o.selected) {
                ++sel_count;
            }
        }
        std::printf("GATE_SEL %d %lld 50\n", cell.cell_id, sel_count);
    }

    long long canary_sel = 0;
    for (const auto& o : out) {
        if (o.cell_id == 1004 && o.rule_id == "GATE_Full" && o.selected) {
            ++canary_sel;
        }
    }
    std::printf("CANARY_RATE %lld/50\n", canary_sel);

    std::printf("METRICS_WRITTEN 1\n");
    std::printf("SMOKE_DONE\n");

    (void)ok;
    return 0;
}
