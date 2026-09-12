// Phase 4A/4B 正式 OOS walk-forward driver。
// 合同: docs/phase4_ab_formal_oos_contract.md（预注册，先于任何 OOS 运行冻结）
// 输入: tools/prepare_formal_oos_data.py 的产物（work/formal-oos-data/）
// 输出: formal_oos_report.json（逐期收益 + 每 arm 摘要；统计检验由 Python verifier 独立复算）
// 纪律: 任一 arm 任一步失败关闭 → 整个 run 作废退出非零（合同 §5.4）。

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "portfolio_math/hierarchical_linkage.h"
#include "portfolio_math/nco_ffv.h"
#include "portfolio_math/nco_policy.h"
#include "portfolio_math/posterior.h"
#include "portfolio_math/posterior_direct.h"
#include "quant_math/matrix.h"

namespace {

constexpr std::size_t kEstimationWindow = 252;
constexpr std::size_t kRebalanceEvery = 5;
constexpr std::size_t kMomentumLookback = 20;
constexpr std::size_t kForwardHorizon = 5;
constexpr std::size_t kMinUniverse = 60;
constexpr double kViewConfidence = 0.30;
constexpr double kMomentumClip = 2.0;
constexpr double kRiskAversion = 1.0;
constexpr double kMaxSingleWeight = 0.10;
constexpr double kCostBps = 10.0;
constexpr double kLinkageCut = 0.70;
constexpr double kPeriodsPerYear = 50.4;  // 252 交易日 / 5-timestamp 调仓间隔

struct Dataset {
    std::size_t timestamps = 0;
    std::vector<double> scenarios;
    std::vector<double> forward;
    std::vector<std::int64_t> timestamps_ns;
    std::vector<std::uint8_t> tradable;
    std::vector<std::uint8_t> valid;
    std::vector<std::string> symbols;
    std::uint64_t contract_hash = 0;
    std::uint64_t mapping_hash = 0;
};

std::vector<char> read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("无法读取: " + path);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::uint64_t hex_prefix_u64(const std::string& hex) {
    return static_cast<std::uint64_t>(std::stoull(hex.substr(0, 16), nullptr, 16));
}

std::string extract_sha_field(const std::string& text, const std::string& key) {
    const auto position = text.find("\"" + key + "\"");
    if (position == std::string::npos) throw std::runtime_error("manifest 缺少字段: " + key);
    const auto colon = text.find(':', position + key.size());
    const auto first = text.find('"', colon);
    const auto last = text.find('"', first + 1);
    return text.substr(first + 1, last - first - 1);
}

struct UnionFind {
    std::vector<std::uint32_t> parent;
    explicit UnionFind(std::uint32_t count) : parent(count) {
        std::iota(parent.begin(), parent.end(), 0u);
    }
    std::uint32_t find(std::uint32_t index) {
        while (parent[index] != index) index = parent[index] = parent[parent[index]];
        return index;
    }
    void unite(std::uint32_t left, std::uint32_t right) {
        const auto a = find(left);
        const auto b = find(right);
        if (a != b) parent[b] = a;
    }
};

std::vector<std::uint32_t> cluster_by_linkage(const quant_math::DenseMatrix& covariance) {
    const std::size_t count = covariance.rows();
    quant_math::DenseMatrix correlation(count, count);
    for (std::size_t row = 0; row < count; ++row) {
        for (std::size_t col = 0; col < count; ++col) {
            const double denominator = std::sqrt(covariance(row, row) * covariance(col, col));
            correlation(row, col) = denominator > 0.0 ? covariance(row, col) / denominator : 0.0;
        }
    }
    const auto linkage = portfolio_math::hierarchical_linkage(quant_math::view(correlation));
    if (linkage.status != portfolio_math::HierarchicalLinkageStatus::OK) {
        throw std::runtime_error("hierarchical_linkage 失败");
    }
    // 合并树的 left/right 是子树根节点 id：资产为 0..n-1，第 k 次合并产生内部节点 n+k。
    UnionFind union_find(static_cast<std::uint32_t>(2 * count - 1));
    for (std::size_t merge_index = 0; merge_index < linkage.diagnostics.merge_tree.size(); ++merge_index) {
        const auto& merge = linkage.diagnostics.merge_tree[merge_index];
        if (merge.distance < kLinkageCut) {
            union_find.unite(merge.left, merge.right);
        }
    }
    std::vector<std::uint32_t> cluster_id(count);
    std::vector<std::uint32_t> roots;
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto root = union_find.find(index);
        auto position = std::find(roots.begin(), roots.end(), root);
        if (position == roots.end()) {
            roots.push_back(root);
            position = roots.end() - 1;
        }
        cluster_id[index] = static_cast<std::uint32_t>(position - roots.begin());
    }
    return cluster_id;
}

std::string number(double value) {
    std::ostringstream stream;
    stream.precision(12);
    stream << value;
    return stream.str();
}

struct ArmState {
    std::vector<double> returns;
    std::vector<double> turnovers;
    std::vector<double> weights_full;
    double cumulative = 0.0;
};

void append_summary(std::ostringstream& json, const std::string& name, const ArmState& arm) {
    auto sharpe = [&](const std::vector<double>& values) {
        if (values.size() < 2) return 0.0;
        const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        double variance = 0.0;
        for (const double value : values) variance += (value - mean) * (value - mean);
        variance /= (values.size() - 1);
        return variance > 0.0 ? mean / std::sqrt(variance) * std::sqrt(kPeriodsPerYear) : 0.0;
    };
    double peak = 1.0, drawdown = 0.0, equity = 1.0;
    for (const double value : arm.returns) {
        equity *= (1.0 + value);
        peak = std::max(peak, equity);
        drawdown = std::max(drawdown, (peak - equity) / peak);
    }
    json << "  \"" << name << "\": {"
         << "\"period_count\": " << arm.returns.size() << ", "
         << "\"annualized_sharpe\": " << number(sharpe(arm.returns)) << ", "
         << "\"cumulative_net_return\": " << number(equity - 1.0) << ", "
         << "\"max_drawdown\": " << number(drawdown) << ", "
         << "\"average_turnover\": "
         << number(arm.turnovers.empty() ? 0.0
                 : std::accumulate(arm.turnovers.begin(), arm.turnovers.end(), 0.0) / arm.turnovers.size())
         << "},\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "用法: formal_oos_driver <data_dir> <output_json>\n";
        return 2;
    }
    try {
        Dataset data;
        {
            const auto symbols_text = read_file(std::string(argv[1]) + "/symbols.txt");
            std::istringstream stream(std::string(symbols_text.data(), symbols_text.size()));
            std::string line;
            while (std::getline(stream, line)) {
                while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
                if (!line.empty()) data.symbols.push_back(line);
            }
        }
        const std::size_t symbol_count = data.symbols.size();
        const auto scenarios = read_file(std::string(argv[1]) + "/scenarios.f64");
        const auto forward = read_file(std::string(argv[1]) + "/forward.f64");
        const auto timestamps = read_file(std::string(argv[1]) + "/timestamps.i64");
        const auto tradable = read_file(std::string(argv[1]) + "/tradable.u8");
        const auto valid = read_file(std::string(argv[1]) + "/valid.u8");
        data.timestamps = timestamps.size() / sizeof(std::int64_t);
        if (scenarios.size() != data.timestamps * symbol_count * sizeof(double) ||
            forward.size() != scenarios.size() ||
            tradable.size() != data.timestamps * symbol_count || valid.size() != tradable.size()) {
            throw std::runtime_error("数据文件尺寸与 symbols/timestamps 不一致");
        }
        data.scenarios.assign(reinterpret_cast<const double*>(scenarios.data()),
                              reinterpret_cast<const double*>(scenarios.data()) + scenarios.size() / sizeof(double));
        data.forward.assign(reinterpret_cast<const double*>(forward.data()),
                            reinterpret_cast<const double*>(forward.data()) + forward.size() / sizeof(double));
        data.timestamps_ns.assign(reinterpret_cast<const std::int64_t*>(timestamps.data()),
                                  reinterpret_cast<const std::int64_t*>(timestamps.data()) + data.timestamps);
        data.tradable.assign(tradable.begin(), tradable.end());
        data.valid.assign(valid.begin(), valid.end());
        const auto manifest = read_file(std::string(argv[1]) + "/prep_manifest.json");
        const std::string manifest_text(manifest.data(), manifest.size());
        data.contract_hash = hex_prefix_u64(extract_sha_field(manifest_text, "contract_sha256"));
        data.mapping_hash = hex_prefix_u64(extract_sha_field(manifest_text, "confidence_mapping_sha256"));

        ArmState risk_only, posterior_bl, posterior_ffv, nco_ffv;
        risk_only.weights_full.assign(symbol_count, 0.0);
        posterior_bl.weights_full = risk_only.weights_full;
        posterior_ffv.weights_full = risk_only.weights_full;
        nco_ffv.weights_full = risk_only.weights_full;
        std::ostringstream steps_json;
        std::size_t skipped = 0;

        for (std::size_t t = kEstimationWindow; t + kForwardHorizon < data.timestamps; t += kRebalanceEvery) {
            std::vector<std::size_t> universe;
            for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
                if (!data.tradable[t * symbol_count + symbol]) continue;
                bool complete = true;
                for (std::size_t row = t - kEstimationWindow; row < t; ++row) {
                    if (!data.valid[row * symbol_count + symbol]) {
                        complete = false;
                        break;
                    }
                }
                if (!complete) continue;
                double sum = 0.0, sum_sq = 0.0;
                for (std::size_t row = t - kEstimationWindow; row < t; ++row) {
                    const double value = data.scenarios[row * symbol_count + symbol];
                    sum += value;
                    sum_sq += value * value;
                }
                const double mean = sum / static_cast<double>(kEstimationWindow);
                const double variance = sum_sq / static_cast<double>(kEstimationWindow) - mean * mean;
                if (variance > 1e-16) universe.push_back(symbol);
            }
            if (universe.size() < kMinUniverse) {
                ++skipped;
                continue;
            }
            const std::size_t n = universe.size();

            std::vector<double> window(n * kEstimationWindow);
            for (std::size_t row = 0; row < kEstimationWindow; ++row) {
                for (std::size_t col = 0; col < n; ++col) {
                    window[row * n + col] =
                        data.scenarios[(t - kEstimationWindow + row) * symbol_count + universe[col]];
                }
            }
            std::vector<engine_common::TimestampNs> window_timestamps(kEstimationWindow);
            for (std::size_t row = 0; row < kEstimationWindow; ++row) {
                window_timestamps[row] = data.timestamps_ns[t - kEstimationWindow + row];
            }
            const auto prior = portfolio_math::build_prior_scenario_artifact(
                {window.data(), kEstimationWindow, n, n}, window_timestamps,
                data.timestamps_ns[t], data.timestamps_ns[t]);
            if (prior.status != portfolio_math::PosteriorStatus::OK) {
                throw std::runtime_error("prior 构建失败于 t=" + std::to_string(t));
            }

            double view_scale = 0.0;
            for (const double value : window) view_scale += std::abs(value);
            view_scale /= static_cast<double>(window.size());

            std::vector<double> momentum(n, 0.0);
            for (std::size_t col = 0; col < n; ++col) {
                for (std::size_t row = kEstimationWindow - kMomentumLookback; row < kEstimationWindow; ++row) {
                    momentum[col] += window[row * n + col];
                }
            }
            const double momentum_mean = std::accumulate(momentum.begin(), momentum.end(), 0.0) / n;
            double momentum_variance = 0.0;
            for (const double value : momentum) momentum_variance += (value - momentum_mean) * (value - momentum_mean);
            momentum_variance /= (n > 1 ? n - 1 : 1);
            const double momentum_std = std::sqrt(std::max(momentum_variance, 1e-18));

            std::vector<std::size_t> order(n);
            std::iota(order.begin(), order.end(), std::size_t{0});
            std::vector<double> zscore(n);
            for (std::size_t col = 0; col < n; ++col) {
                zscore[col] = (momentum[col] - momentum_mean) / momentum_std;
            }
            std::sort(order.begin(), order.end(),
                      [&](std::size_t left, std::size_t right) { return zscore[left] > zscore[right]; });

            auto build_views = [&](std::size_t tail) {
                std::vector<portfolio_math::ViewSpecV1> specs;
                specs.reserve(2 * tail);
                for (std::size_t rank = 0; rank < tail; ++rank) {
                    for (const std::size_t col : {order[rank], order[n - 1 - rank]}) {
                        portfolio_math::ViewSpecV1 spec;
                        spec.view_id = "momentum-z-" + std::to_string(universe[col]);
                        spec.available_at = data.timestamps_ns[t];
                        spec.loading.assign(n, 0.0);
                        spec.loading[col] = 1.0;
                        spec.target = std::clamp(zscore[col], -kMomentumClip, kMomentumClip) * view_scale;
                        spec.confidence = kViewConfidence;
                        spec.observation_variance = view_scale * view_scale;
                        spec.confidence_mapping_hash = data.mapping_hash;
                        spec.source_artifact_hash = data.contract_hash;
                        specs.push_back(std::move(spec));
                    }
                }
                return specs;
            };

            portfolio_math::FFVOptions ffv_options;
            ffv_options.max_iterations = 5000;
            const std::size_t decile = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(0.1 * static_cast<double>(n))));
            const std::size_t quintile = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(0.05 * static_cast<double>(n))));
            const std::size_t tiers[] = {decile, quintile, 1, 0};
            const auto clusters = cluster_by_linkage(prior.prior_covariance);
            const std::uint32_t cluster_count =
                1 + *std::max_element(clusters.begin(), clusters.end());
            std::size_t view_tier = 0;
            portfolio_math::PosteriorScenarioArtifactV1 bl;
            portfolio_math::PosteriorScenarioArtifactV1 ffv;
            portfolio_math::NcoFfvPolicyResult nco_result;
            for (std::size_t tier_index = 0; tier_index < 4; ++tier_index) {
                const auto views = build_views(tiers[tier_index]);
                ffv = portfolio_math::apply_ffv_mean_views(
                    prior, std::span<const portfolio_math::ViewSpecV1>(views), {}, ffv_options);
                if (ffv.status != portfolio_math::PosteriorStatus::OK) continue;
                bl = portfolio_math::apply_gaussian_mean_views(
                    prior, std::span<const portfolio_math::ViewSpecV1>(views));
                if (bl.status != portfolio_math::PosteriorStatus::OK) {
                    throw std::runtime_error("BL 失败于可行视图层级 t=" + std::to_string(t));
                }
                auto candidate = portfolio_math::solve_nco_ffv_minvar(ffv, clusters, cluster_count);
                if (candidate.status != portfolio_math::OptimizationStatus::OK) continue;
                nco_result = candidate;
                view_tier = tier_index;
                break;
            }
            if (bl.status != portfolio_math::PosteriorStatus::OK) {
                // 无视图层级：两后验经各自引擎的空视图路径（合同保证=prior 奇偶）
                const std::span<const portfolio_math::ViewSpecV1> no_views{};
                bl = portfolio_math::apply_gaussian_mean_views(prior, no_views);
                ffv = portfolio_math::apply_ffv_mean_views(prior, no_views, {}, ffv_options);
                if (bl.status != portfolio_math::PosteriorStatus::OK ||
                    ffv.status != portfolio_math::PosteriorStatus::OK) {
                    throw std::runtime_error("无视图后验仍失败 t=" + std::to_string(t));
                }
                nco_result = portfolio_math::solve_nco_ffv_minvar(ffv, clusters, cluster_count);
                if (nco_result.status != portfolio_math::OptimizationStatus::OK) {
                    throw std::runtime_error("无视图层级 NCO 仍失败 t=" + std::to_string(t));
                }
                view_tier = 3;
            }



            portfolio_math::PosteriorDirectOptions direct_options;
            direct_options.risk_aversion = kRiskAversion;
            direct_options.max_single_weight = kMaxSingleWeight;
            direct_options.target_investment = 1.0;
            direct_options.tolerance = 1e-8;
            direct_options.max_iterations = 200'000;

            const auto risk_result = portfolio_math::solve_nco_minvar(
                quant_math::view(prior.prior_covariance), clusters, cluster_count);
            const auto bl_result = portfolio_math::solve_posterior_direct(bl, direct_options);
            const auto ffv_result = portfolio_math::solve_posterior_direct(ffv, direct_options);
            if (risk_result.diagnostics.status != portfolio_math::OptimizationStatus::OK ||
                bl_result.diagnostics.status != portfolio_math::OptimizationStatus::OK ||
                ffv_result.diagnostics.status != portfolio_math::OptimizationStatus::OK) {
                throw std::runtime_error("solver 失败关闭于 t=" + std::to_string(t) +
                                         " risk=" + std::to_string(static_cast<int>(risk_result.diagnostics.status)) +
                                         " bl=" + std::to_string(static_cast<int>(bl_result.diagnostics.status)) +
                                         " ffv=" + std::to_string(static_cast<int>(ffv_result.diagnostics.status)) +
                                         " ncoffv=" + std::to_string(static_cast<int>(nco_result.status)) +
                                         " clusters=" + std::to_string(cluster_count) + " universe=" + std::to_string(n));
            }

            struct ArmRun {
                ArmState* state;
                const std::vector<double>* weights;
            };
            ArmRun arms[] = {
                {&risk_only, &risk_result.weights},
                {&posterior_bl, &bl_result.weights},
                {&posterior_ffv, &ffv_result.weights},
                {&nco_ffv, &nco_result.nco.weights},
            };
            double gross_forward[4] = {0, 0, 0, 0};
            for (std::size_t arm_index = 0; arm_index < 4; ++arm_index) {
                auto& arm = arms[arm_index];
                double turnover = 0.0;
                double gross = 0.0;
                for (std::size_t col = 0; col < n; ++col) {
                    const std::size_t symbol = universe[col];
                    const double weight = (*arm.weights)[col];
                    turnover += std::abs(weight - arm.state->weights_full[symbol]);
                    gross += weight * data.forward[t * symbol_count + symbol];
                    arm.state->weights_full[symbol] = weight;
                }
                const double net = gross - kCostBps * 1e-4 * turnover;
                if (!std::isfinite(gross) || !std::isfinite(turnover) || !std::isfinite(net)) {
                    throw std::runtime_error("非有限收益于 t=" + std::to_string(t));
                }
                arm.state->returns.push_back(net);
                arm.state->turnovers.push_back(turnover);
                gross_forward[arm_index] = net;
            }
            steps_json << "   {\"t\": " << t << ", \"timestamp\": " << data.timestamps_ns[t]
                       << ", \"view_tier\": " << view_tier
                       << ", \"universe\": " << n << ", \"clusters\": " << cluster_count
                       << ", \"net\": [" << number(gross_forward[0]) << ", " << number(gross_forward[1])
                       << ", " << number(gross_forward[2]) << ", " << number(gross_forward[3]) << "]},\n";
        }

        std::ostringstream json;
        json << "{\n \"schema_version\": 1,\n \"driver\": \"formal_oos_driver\",\n";
        json << " \"arms\": [\"risk_only\", \"posterior_bl\", \"posterior_ffv\", \"nco_ffv\"],\n";
        json << " \"rebalance_periods\": " << risk_only.returns.size() << ",\n";
        json << " \"skipped_steps\": " << skipped << ",\n";
        json << " \"contract_hash_u64\": " << data.contract_hash << ",\n";
        json << " \"mapping_hash_u64\": " << data.mapping_hash << ",\n \"summary\": {\n";
        append_summary(json, "risk_only", risk_only);
        append_summary(json, "posterior_bl", posterior_bl);
        append_summary(json, "posterior_ffv", posterior_ffv);
        append_summary(json, "nco_ffv", nco_ffv);
        json.seekp(-2, std::ios::end);
        json << "\n },\n \"steps\": [\n" << steps_json.str();
        json.seekp(-2, std::ios::end);
        json << "\n ],\n \"returns\": {\n";
        auto emit_returns = [&](const char* name, const std::vector<double>& values) {
            json << "  \"" << name << "\": [";
            for (std::size_t index = 0; index < values.size(); ++index) {
                json << (index ? ", " : "") << number(values[index]);
            }
            json << "],\n";
        };
        emit_returns("risk_only", risk_only.returns);
        emit_returns("posterior_bl", posterior_bl.returns);
        emit_returns("posterior_ffv", posterior_ffv.returns);
        emit_returns("nco_ffv", nco_ffv.returns);
        json.seekp(-2, std::ios::end);
        json << "\n }\n}\n";

        std::ofstream output(argv[2]);
        output << json.str();
        std::cout << "formal_oos_driver: periods=" << risk_only.returns.size()
                  << " skipped=" << skipped << " -> " << argv[2] << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL_CLOSED: " << error.what() << "\n";
        return 1;
    }
}
