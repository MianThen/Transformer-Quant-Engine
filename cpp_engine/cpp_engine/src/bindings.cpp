// pybind11 绑定:把 C++ 引擎、批处理入口、历史查询和构建元数据暴露为
// Python 可 import 的 `cpp_engine` 模块。

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "arrow_bridge.h"
#include "engine.h"
#include "event.h"
#include "market_data.h"
#include "order.h"
#include "pnl_tracker.h"
#include "position.h"
#include "types.h"

#ifdef QBT_ENABLE_PERFORMANCE_ANALYTICS
#include "performance_analytics/period_contribution_replay_sink.h"
#include "performance_analytics/performance_spec.h"
#include "performance_analytics/return_analysis.h"
#include "performance_analytics/return_ledger.h"
#endif
#ifdef QBT_ENABLE_PORTFOLIO_MATH
#include "portfolio_math/tail_risk.h"
#endif

#ifdef QBT_ENABLE_ML
#include "strategy_runtime/model_strategy_runtime.h"
#endif
#ifdef QBT_ML_ONNXRUNTIME
#include "ml_runtime/onnx_runtime_backend.h"
#endif

namespace py = pybind11;
using namespace qbt;

namespace {

class MarketBatchView {
public:
  explicit MarketBatchView(std::span<const MarketSnapshot> batch)
      : batch_(batch) {}
  size_t size() const { return batch_.size(); }
  const std::string &symbol(size_t index) const { return value(index).symbol; }
  Timestamp timestamp(size_t index) const { return value(index).timestamp; }
  Price open(size_t index) const { return value(index).open; }
  Price high(size_t index) const { return value(index).high; }
  Price low(size_t index) const { return value(index).low; }
  Price close(size_t index) const { return value(index).close; }
  Quantity volume(size_t index) const { return value(index).volume; }

private:
  const MarketSnapshot &value(size_t index) const {
    if (index >= batch_.size())
      throw py::index_error("market batch index out of range");
    return batch_[index];
  }
  std::span<const MarketSnapshot> batch_;
};

std::vector<Order> parse_batch_orders(const py::object &value,
                                      std::span<const MarketSnapshot> market) {
  if (value.is_none())
    return {};
  if (!py::isinstance<py::dict>(value))
    return value.cast<std::vector<Order>>();
  const py::dict columns = value.cast<py::dict>();
  if (!columns.contains("quantity")) {
    throw py::value_error("columnar orders require quantity");
  }
  const py::sequence quantities = columns["quantity"].cast<py::sequence>();
  const size_t count = static_cast<size_t>(py::len(quantities));
  const bool has_symbols = columns.contains("symbol");
  const bool has_symbol_indices = columns.contains("symbol_index");
  if (!has_symbols && !has_symbol_indices && market.size() != 1) {
    throw py::value_error("columnar orders require symbol or symbol_index for "
                          "multi-symbol batches");
  }
  py::sequence symbols;
  py::sequence symbol_indices;
  py::sequence sides;
  py::sequence types;
  py::sequence prices;
  if (has_symbols)
    symbols = columns["symbol"].cast<py::sequence>();
  if (has_symbol_indices)
    symbol_indices = columns["symbol_index"].cast<py::sequence>();
  if (columns.contains("side"))
    sides = columns["side"].cast<py::sequence>();
  if (columns.contains("type"))
    types = columns["type"].cast<py::sequence>();
  if (columns.contains("price"))
    prices = columns["price"].cast<py::sequence>();
  auto require_size = [count](const py::sequence &sequence, const char *name) {
    if (sequence && static_cast<size_t>(py::len(sequence)) != count)
      throw py::value_error(std::string(name) + " length mismatch");
  };
  require_size(symbols, "symbol");
  require_size(symbol_indices, "symbol_index");
  require_size(sides, "side");
  require_size(types, "type");
  require_size(prices, "price");

  std::vector<Order> orders(count);
  for (size_t index = 0; index < count; ++index) {
    Order &order = orders[index];
    order.quantity = quantities[index].cast<Quantity>();
    if (has_symbols) {
      order.symbol = symbols[index].cast<std::string>();
    } else if (has_symbol_indices) {
      const size_t market_index = symbol_indices[index].cast<size_t>();
      if (market_index >= market.size())
        throw py::index_error("symbol_index outside market batch");
      order.symbol = market[market_index].symbol;
    }
    if (sides)
      order.side = static_cast<Side>(sides[index].cast<int>());
    if (types)
      order.type = static_cast<OrderType>(types[index].cast<int>());
    if (prices)
      order.limit_price = prices[index].cast<Price>();
  }
  return orders;
}

#ifdef QBT_ENABLE_PERFORMANCE_ANALYTICS
std::shared_ptr<performance_analytics::PeriodContributionReplaySink>
make_period_contribution_sink(const std::string& calendar_id,
                              double periods_per_year,
                              std::uint64_t config_hash) {
  performance_analytics::PerformanceSpecV1 spec;
  spec.calendar_id = calendar_id;
  spec.calendar_periods_per_year = periods_per_year;
  spec.config_hash = config_hash;
  return std::make_shared<performance_analytics::PeriodContributionReplaySink>(
      std::move(spec));
}

std::string serialize_drift_snapshot_binding(const py::dict& fields) {
  performance_analytics::DriftSnapshotContractV0 contract;
  if (fields.contains("schema_version"))
    contract.schema_version = fields["schema_version"].cast<std::uint32_t>();
  if (fields.contains("labels_mature"))
    contract.labels_mature = fields["labels_mature"].cast<bool>();

  const auto required_string = [&fields](const char* name) {
    if (!fields.contains(name))
      throw py::value_error(std::string("drift snapshot missing ") + name);
    return fields[name].cast<std::string>();
  };
  contract.model_manifest_sha256 = required_string("model_manifest_sha256");
  contract.raw_schema_hash = required_string("raw_schema_hash");
  contract.preprocessing_spec_sha256 =
      required_string("preprocessing_spec_sha256");
  contract.feature_schema_hash = required_string("feature_schema_hash");
  contract.prediction_schema_hash = required_string("prediction_schema_hash");
  contract.raw_fields_sha256 = required_string("raw_fields_sha256");
  contract.preprocessed_features_sha256 =
      required_string("preprocessed_features_sha256");
  contract.prediction_values_sha256 =
      required_string("prediction_values_sha256");
  contract.embedding_values_sha256 =
      required_string("embedding_values_sha256");
  contract.source_snapshot_set_sha256 =
      required_string("source_snapshot_set_sha256");
  contract.ledger_schema_hash = required_string("ledger_schema_hash");
  contract.available_at_utc = required_string("available_at_utc");
  if (fields.contains("label_spec_sha256"))
    contract.label_spec_sha256 = fields["label_spec_sha256"].cast<std::string>();
  if (fields.contains("matured_labels_sha256"))
    contract.matured_labels_sha256 =
        fields["matured_labels_sha256"].cast<std::string>();
  if (fields.contains("report_sha256"))
    contract.report_sha256 = fields["report_sha256"].cast<std::string>();

  const std::string artifact =
      performance_analytics::serialize_drift_snapshot_artifact(contract);
  if (artifact.empty())
    throw py::value_error("invalid drift snapshot contract");
  return artifact;
}
#endif

#ifdef QBT_ENABLE_PORTFOLIO_MATH
py::dict estimate_empirical_cvar_binding(const std::vector<double>& returns,
                                         const std::vector<Timestamp>& timestamps,
                                         double confidence_level,
                                         std::uint64_t config_hash) {
  if (returns.empty() || returns.size() != timestamps.size()) {
    throw py::value_error("returns and timestamps must be non-empty and aligned");
  }
  std::vector<engine_common::SymbolId> symbols{1};
  std::vector<double> weights{1.0};
  portfolio_math::TailRiskSpec spec;
  spec.confidence_level = confidence_level;
  spec.config_hash = config_hash;
  const portfolio_math::TailRiskProblemView problem{
      timestamps.back(), symbols, timestamps, weights, returns, {}, {}, {},
      {}, nullptr, spec};
  const auto estimate = portfolio_math::estimate_tail_risk(problem);
  py::dict result;
  result["status"] = static_cast<int>(estimate.status);
  result["estimator"] = static_cast<int>(estimate.estimator);
  result["confidence_level"] = estimate.confidence_level;
  result["effective_observations"] = estimate.effective_observations;
  result["input_hash"] = estimate.input_hash;
  result["artifact_hash"] = estimate.artifact_hash;
  result["var_loss"] = estimate.value_at_risk_loss
      ? py::cast(*estimate.value_at_risk_loss) : py::none();
  result["expected_shortfall_loss"] = estimate.expected_shortfall_loss
      ? py::cast(*estimate.expected_shortfall_loss) : py::none();
  result["return_cvar"] = estimate.return_cvar
      ? py::cast(*estimate.return_cvar) : py::none();
  portfolio_math::TailRiskArtifactSpec artifact_spec;
  artifact_spec.reference_price_quality = "PROXY";
  artifact_spec.promotion_eligible = false;
  artifact_spec.limitations.push_back("REFERENCE_PRICE_PROXY");
  result["artifact_json"] = portfolio_math::serialize_tail_risk_artifact(
      estimate, spec, artifact_spec);
  return result;
}
#endif

#ifdef QBT_ML_ONNXRUNTIME
std::string sha256_file(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw py::value_error("cannot open model artifact file: " + path.string());
  py::object digest = py::module_::import("hashlib").attr("sha256")();
  std::array<char, 1024 * 1024> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (count > 0)
      digest.attr("update")(py::bytes(buffer.data(), count));
  }
  if (!input.eof())
    throw py::value_error("failed to read artifact file: " + path.string());
  return digest.attr("hexdigest")().cast<std::string>();
}

uint64_t hash64_text(const std::string &value) {
  py::object digest =
      py::module_::import("hashlib").attr("sha256")(py::bytes(value));
  return std::stoull(
      digest.attr("hexdigest")().cast<std::string>().substr(0, 16), nullptr,
      16);
}

template <class Value>
Value config_value(const py::dict &source, const char *name, Value fallback) {
  return source.contains(name) ? source[name].cast<Value>() : fallback;
}

std::shared_ptr<engine_common::IStrategyRuntime>
load_model_strategy(const std::string &artifact_path,
                    const py::dict &policy_config, const py::dict &risk_config,
                    const py::dict &runtime_config,
                    engine_common::StrategySessionContext &context) {
  const std::filesystem::path root =
      std::filesystem::absolute(std::filesystem::path(artifact_path));
  const auto manifest_path = root / "manifest.json";
  const auto model_path = root / "model.onnx";
  const auto schema_path = root / "feature_schema.json";
  if (!std::filesystem::is_regular_file(manifest_path) ||
      !std::filesystem::is_regular_file(model_path) ||
      !std::filesystem::is_regular_file(schema_path)) {
    throw py::value_error(
        "artifact requires manifest.json, model.onnx and feature_schema.json");
  }
  py::object json = py::module_::import("json");
  py::object pathlib = py::module_::import("pathlib").attr("Path");
  py::dict manifest =
      json.attr("loads")(
              pathlib(manifest_path.string()).attr("read_text")("utf-8"))
          .cast<py::dict>();
  const auto required_string = [&](const char *name) {
    if (!manifest.contains(name))
      throw py::value_error(std::string("manifest missing ") + name);
    return manifest[name].cast<std::string>();
  };
  if (manifest["schema_version"].cast<int>() != 1 ||
      required_string("feature_profile") != "BAR_V1" ||
      required_string("execution_alignment") != "NEXT_OPEN" ||
      required_string("input_dtype") != "float32" ||
      required_string("input_layout") != "NTF" ||
      required_string("preferred_provider") != "CPUExecutionProvider") {
    throw py::value_error("unsupported model manifest protocol");
  }
  const std::string expected_model_hash = required_string("model_sha256");
  const std::string expected_schema_hash =
      required_string("feature_schema_sha256");
  if (sha256_file(model_path) != expected_model_hash) {
    throw py::value_error("model.onnx SHA-256 mismatch");
  }
  if (sha256_file(schema_path) != expected_schema_hash) {
    throw py::value_error("feature_schema.json SHA-256 mismatch");
  }
  py::dict feature_schema =
      json.attr("loads")(
              pathlib(schema_path.string()).attr("read_text")("utf-8"))
          .cast<py::dict>();
  if (!feature_schema.contains("schema_version") ||
      feature_schema["schema_version"].cast<int>() != 1 ||
      !feature_schema.contains("profile") ||
      feature_schema["profile"].cast<std::string>() != "BAR_V1" ||
      !feature_schema.contains("layout") ||
      feature_schema["layout"].cast<std::string>() != "NTF" ||
      !feature_schema.contains("value_dtype") ||
      feature_schema["value_dtype"].cast<std::string>() != "float32" ||
      !feature_schema.contains("feature_names")) {
    throw py::value_error("unsupported feature schema protocol");
  }
  const py::sequence feature_names =
      feature_schema["feature_names"].cast<py::sequence>();
  if (static_cast<size_t>(py::len(feature_names)) !=
      qbt::ml::kBarV1FeatureNames.size()) {
    throw py::value_error("BAR_V1 feature count mismatch");
  }
  for (size_t index = 0; index < qbt::ml::kBarV1FeatureNames.size(); ++index) {
    if (feature_names[index].cast<std::string>() !=
        qbt::ml::kBarV1FeatureNames[index]) {
      throw py::value_error("BAR_V1 feature order mismatch");
    }
  }

  qbt::strategy::ModelStrategyConfig config;
  config.artifact.root = root;
  config.artifact.model_path = model_path;
  auto &descriptor = config.artifact.descriptor;
  descriptor.model_id = required_string("model_id");
  descriptor.model_version = required_string("model_version");
  descriptor.feature_schema_hash =
      std::stoull(expected_schema_hash.substr(0, 16), nullptr, 16);
  descriptor.model_version_hash = hash64_text(descriptor.model_version);
  descriptor.lookback = manifest["lookback"].cast<uint32_t>();
  descriptor.feature_count = manifest["feature_count"].cast<uint32_t>();
  descriptor.static_feature_count =
      manifest["static_feature_count"].cast<uint32_t>();
  descriptor.max_batch_size =
      config_value<uint32_t>(runtime_config, "max_batch_size", 4096);
  if (manifest.contains("label_spec_version") &&
      manifest["label_spec_version"].cast<std::string>() == "V2") {
    const std::string label_spec_hash = required_string("label_spec_sha256");
    const std::string ranking_spec_hash =
        required_string("ranking_score_spec_sha256");
    if (label_spec_hash.size() != 64 || ranking_spec_hash.size() != 64 ||
        !manifest.contains("ranking_score_spec") ||
        !manifest.contains("ranking_cutoff")) {
      throw py::value_error("invalid Phase 1B manifest contract");
    }
    const py::dict ranking_spec =
        manifest["ranking_score_spec"].cast<py::dict>();
    if (!ranking_spec.contains("mode") ||
        !ranking_spec.contains("risk_floor")) {
      throw py::value_error("RankingScoreSpec V1 is incomplete");
    }
    const std::string score_mode = ranking_spec["mode"].cast<std::string>();
    if (score_mode == "raw_return") {
      descriptor.ranking_score_mode =
          engine_common::RankingScoreMode::RAW_RETURN;
    } else if (score_mode == "risk_adjusted_return") {
      descriptor.ranking_score_mode =
          engine_common::RankingScoreMode::RISK_ADJUSTED_RETURN;
    } else {
      throw py::value_error("unsupported RankingScoreSpec mode");
    }
    descriptor.ranking_risk_floor = ranking_spec["risk_floor"].cast<float>();
    descriptor.ranking_cutoff = manifest["ranking_cutoff"].cast<uint32_t>();
    descriptor.ranking_score_spec_hash =
        std::stoull(ranking_spec_hash.substr(0, 16), nullptr, 16);
    descriptor.label_spec_hash =
        std::stoull(label_spec_hash.substr(0, 16), nullptr, 16);
    if (descriptor.ranking_cutoff == 0 ||
        !std::isfinite(descriptor.ranking_risk_floor) ||
        descriptor.ranking_risk_floor <= 0.0F) {
      throw py::value_error("invalid RankingScoreSpec cutoff or risk floor");
    }
  }
  if (descriptor.lookback == 0 ||
      descriptor.feature_count != qbt::ml::kBarV1FeatureCount ||
      descriptor.static_feature_count != 0 || descriptor.max_batch_size == 0) {
    throw py::value_error("artifact shape does not match BAR_V1 runtime");
  }
  config.runtime_options.intra_op_threads =
      config_value<uint32_t>(runtime_config, "intra_op_threads", 1);
  config.runtime_options.inter_op_threads =
      config_value<uint32_t>(runtime_config, "inter_op_threads", 1);
  config.runtime_options.max_batch_size = descriptor.max_batch_size;
  config.runtime_options.deadline_ns =
      config_value<int64_t>(runtime_config, "deadline_ns", 0);
  config.policy.max_positions = config_value<uint32_t>(
      policy_config, "max_positions",
      descriptor.ranking_cutoff == 0 ? 20U : descriptor.ranking_cutoff);
  config.policy.max_position_weight =
      config_value<float>(policy_config, "max_position_weight", 0.05F);
  config.policy.minimum_expected_return =
      config_value<float>(policy_config, "minimum_expected_return", 0.0F);
  config.policy.minimum_ranking_score =
      config_value<float>(policy_config, "minimum_ranking_score",
                          std::numeric_limits<float>::quiet_NaN());
  config.policy.minimum_confidence =
      config_value<float>(policy_config, "minimum_confidence", 0.0F);
  config.policy.ranking_score_mode = descriptor.ranking_score_mode;
  config.policy.ranking_risk_floor = descriptor.ranking_risk_floor;
  config.risk.kill_switch =
      config_value<bool>(risk_config, "kill_switch", false);
  config.risk.require_trusted_market =
      config_value<bool>(risk_config, "require_trusted_market", true);
  config.risk.max_order_quantity = config_value<engine_common::Quantity>(
      risk_config, "max_order_quantity", 1'000'000);
  config.max_order_intents =
      config_value<size_t>(runtime_config, "max_order_intents", 1024);

  context.feature_schema_hash = descriptor.feature_schema_hash;
  context.model_version_hash = descriptor.model_version_hash;
  context.allow_orders = true;
  context.live = false;
  context.shadow = false;
  return std::make_shared<qbt::strategy::ModelStrategyRuntime>(
      std::make_unique<qbt::ml::OnnxRuntimeBackend>(), std::move(config));
}
#endif

} // namespace

PYBIND11_MODULE(cpp_engine, m) {
  m.doc() = "quant-backtester C++ 核心引擎";
  m.attr("__build_type__") = QBT_BUILD_TYPE;
  m.attr("__compiler_id__") = QBT_COMPILER_ID;
  m.attr("__compiler_version__") = QBT_COMPILER_VERSION;
  m.attr("__lto_enabled__") = static_cast<bool>(QBT_LTO_ENABLED);
#ifdef QBT_ENABLE_ML
  m.attr("__ml_enabled__") = true;
#ifdef QBT_ML_ONNXRUNTIME
  m.attr("__ml_backend__") = "onnxruntime";
#else
  m.attr("__ml_backend__") = "mock";
#endif
#else
  m.attr("__ml_enabled__") = false;
  m.attr("__ml_backend__") = "disabled";
#endif

#ifdef QBT_ENABLE_PERFORMANCE_ANALYTICS
  m.def("serialize_drift_snapshot_artifact",
        &serialize_drift_snapshot_binding, py::arg("fields"));

  py::class_<engine_common::IReplayAnalyticsSink,
             std::shared_ptr<engine_common::IReplayAnalyticsSink>>(
      m, "ReplayAnalyticsSink");
  py::class_<performance_analytics::PeriodContributionReplaySink,
             engine_common::IReplayAnalyticsSink,
             std::shared_ptr<performance_analytics::PeriodContributionReplaySink>>(
      m, "PeriodContributionReplaySink")
      .def(py::init([](const std::string& calendar_id,
                       double periods_per_year,
                       std::uint64_t config_hash) {
             return make_period_contribution_sink(calendar_id,
                                                  periods_per_year,
                                                  config_hash);
           }),
           py::arg("calendar_id"), py::arg("periods_per_year"),
           py::arg("config_hash"))
      .def("failed",
           &performance_analytics::PeriodContributionReplaySink::failed)
      .def("ledger_hash", [](const performance_analytics::PeriodContributionReplaySink& sink) {
        return sink.return_ledger().ledger_hash();
      })
      .def("ledger_sha256", [](const performance_analytics::PeriodContributionReplaySink& sink) {
        return sink.return_ledger().ledger_sha256();
      })
      .def("period_returns", [](const performance_analytics::PeriodContributionReplaySink& sink) {
        std::vector<double> values;
        for (const auto& record : sink.return_ledger().records()) {
          values.push_back(record.period_return);
        }
        return values;
      })
      .def("period_end_timestamps", [](const performance_analytics::PeriodContributionReplaySink& sink) {
        std::vector<Timestamp> values;
        for (const auto& record : sink.return_ledger().records()) {
          values.push_back(record.period_end);
        }
        return values;
      })
      .def("serialize_return_ledger_artifact",
           [](const performance_analytics::PeriodContributionReplaySink& sink,
              const std::string& source_replay_sha256,
              const std::string& dataset_fingerprint,
              const std::string& reference_price_quality,
              bool benchmark_available,
              const std::vector<std::string>& limitations) {
             performance_analytics::ReturnLedgerArtifactSpec spec;
             spec.source_replay_sha256 = source_replay_sha256;
             spec.dataset_fingerprint = dataset_fingerprint;
             spec.benchmark_available = benchmark_available;
             spec.reference_price_quality = reference_price_quality;
             spec.limitations = limitations;
             return sink.serialize_return_ledger_artifact(spec);
           },
           py::arg("source_replay_sha256"),
           py::arg("dataset_fingerprint"),
           py::arg("reference_price_quality") = "PROXY",
           py::arg("benchmark_available") = false,
           py::arg("limitations") = std::vector<std::string>{})
      .def("serialize_return_analysis_report",
           [](const performance_analytics::PeriodContributionReplaySink& sink,
              const std::string& source_replay_sha256,
              const std::string& dataset_fingerprint,
              const std::string& reference_price_quality,
              bool benchmark_available,
              const std::vector<std::string>& limitations,
              const std::optional<double>& var_loss,
              const std::optional<double>& expected_shortfall_loss,
              const std::optional<double>& return_cvar) {
             performance_analytics::ReturnAnalysisManifest manifest;
             manifest.source_replay_sha256 = source_replay_sha256;
             manifest.dataset_fingerprint = dataset_fingerprint;
             manifest.benchmark_available = benchmark_available;
             manifest.reference_price_quality = reference_price_quality;
             manifest.limitations = limitations;
             manifest.var_loss = var_loss;
             manifest.expected_shortfall_loss = expected_shortfall_loss;
             manifest.return_cvar = return_cvar;
             return sink.serialize_return_analysis_report(std::move(manifest));
           },
           py::arg("source_replay_sha256"),
           py::arg("dataset_fingerprint"),
           py::arg("reference_price_quality") = "PROXY",
           py::arg("benchmark_available") = false,
           py::arg("limitations") = std::vector<std::string>{},
           py::arg("var_loss") = std::nullopt,
           py::arg("expected_shortfall_loss") = std::nullopt,
           py::arg("return_cvar") = std::nullopt);
#endif

#ifdef QBT_ENABLE_PORTFOLIO_MATH
  m.def("estimate_empirical_cvar", &estimate_empirical_cvar_binding,
        py::arg("returns"), py::arg("timestamps"),
        py::arg("confidence_level") = 0.95,
        py::arg("config_hash") = 1);
#endif

  py::enum_<Side>(m, "Side").value("BUY", Side::BUY).value("SELL", Side::SELL);

  py::enum_<OrderType>(m, "OrderType")
      .value("MARKET", OrderType::MARKET)
      .value("LIMIT", OrderType::LIMIT);

  py::enum_<FillTiming>(m, "FillTiming")
      .value("NEXT_OPEN", FillTiming::NEXT_OPEN)
      .value("CLOSE", FillTiming::CLOSE);

  py::enum_<EquitySampling>(m, "EquitySampling")
      .value("EVERY_BAR", EquitySampling::EVERY_BAR)
      .value("DAILY", EquitySampling::DAILY)
      .value("ON_FILL", EquitySampling::ON_FILL)
      .value("DISABLED", EquitySampling::DISABLED);

  py::class_<HistoryConfig>(m, "HistoryConfig")
      .def(py::init<>())
      .def_readwrite("record_orders", &HistoryConfig::record_orders)
      .def_readwrite("record_trades", &HistoryConfig::record_trades)
      .def_readwrite("record_round_trips", &HistoryConfig::record_round_trips)
      .def_readwrite("equity_sampling", &HistoryConfig::equity_sampling);

  py::enum_<OrderStatus>(m, "OrderStatus")
      .value("ACCEPTED", OrderStatus::ACCEPTED)
      .value("PARTIALLY_FILLED", OrderStatus::PARTIALLY_FILLED)
      .value("FILLED", OrderStatus::FILLED)
      .value("CANCELED", OrderStatus::CANCELED)
      .value("REJECTED", OrderStatus::REJECTED)
      .value("EXPIRED", OrderStatus::EXPIRED);

  py::enum_<RejectReason>(m, "RejectReason")
      .value("NONE", RejectReason::NONE)
      .value("INVALID_ORDER", RejectReason::INVALID_ORDER)
      .value("UNKNOWN_SYMBOL", RejectReason::UNKNOWN_SYMBOL)
      .value("NOT_LISTED", RejectReason::NOT_LISTED)
      .value("INVALID_LOT_SIZE", RejectReason::INVALID_LOT_SIZE)
      .value("INSUFFICIENT_CASH", RejectReason::INSUFFICIENT_CASH)
      .value("INSUFFICIENT_POSITION", RejectReason::INSUFFICIENT_POSITION)
      .value("STALE_MARKET_DATA", RejectReason::STALE_MARKET_DATA);

  py::class_<FeeSchedule>(m, "FeeSchedule")
      .def(py::init([](Timestamp effective_from, const py::object &effective_to,
                       double commission_rate, Price min_commission,
                       double stamp_tax_rate, double transfer_fee_rate) {
             FeeSchedule value;
             value.effective_from = effective_from;
             if (!effective_to.is_none()) {
               value.effective_to = effective_to.cast<Timestamp>();
             }
             value.commission_rate = commission_rate;
             value.min_commission = min_commission;
             value.stamp_tax_rate = stamp_tax_rate;
             value.transfer_fee_rate = transfer_fee_rate;
             return value;
           }),
           py::arg("effective_from"), py::arg("effective_to"),
           py::arg("commission_rate"), py::arg("min_commission"),
           py::arg("stamp_tax_rate"), py::arg("transfer_fee_rate") = 0.0)
      .def_readwrite("effective_from", &FeeSchedule::effective_from)
      .def_readwrite("effective_to", &FeeSchedule::effective_to)
      .def_readwrite("commission_rate", &FeeSchedule::commission_rate)
      .def_readwrite("min_commission", &FeeSchedule::min_commission)
      .def_readwrite("stamp_tax_rate", &FeeSchedule::stamp_tax_rate)
      .def_readwrite("transfer_fee_rate", &FeeSchedule::transfer_fee_rate);

  py::class_<ExecutionConfig>(m, "ExecutionConfig")
      .def(py::init<>())
      .def_readwrite("max_volume_participation",
                     &ExecutionConfig::max_volume_participation)
      .def_readwrite("slippage_bps", &ExecutionConfig::slippage_bps)
      .def_readwrite("enforce_price_limits",
                     &ExecutionConfig::enforce_price_limits)
      .def_readwrite("enforce_t_plus_one", &ExecutionConfig::enforce_t_plus_one)
      .def_readwrite("allow_short", &ExecutionConfig::allow_short)
      .def_readwrite("enforce_board_lot", &ExecutionConfig::enforce_board_lot)
      .def_readwrite("enforce_cash", &ExecutionConfig::enforce_cash)
      .def_readwrite("market_order_price_buffer_bps",
                     &ExecutionConfig::market_order_price_buffer_bps);

  py::class_<Order>(m, "Order")
      .def(py::init<>())
      .def_readwrite("id", &Order::id)
      .def_readwrite("decision_id", &Order::decision_id)
      .def_readwrite("symbol", &Order::symbol)
      .def_readwrite("side", &Order::side)
      .def_readwrite("type", &Order::type)
      .def_readwrite("quantity", &Order::quantity)
      .def_readwrite("limit_price", &Order::limit_price)
      .def_readwrite("timestamp", &Order::timestamp);

  py::class_<Fill>(m, "Fill")
      .def(py::init<>())
      .def_readwrite("order_id", &Fill::order_id)
      .def_readwrite("symbol", &Fill::symbol)
      .def_readwrite("side", &Fill::side)
      .def_readwrite("quantity", &Fill::quantity)
      .def_readwrite("price", &Fill::price)
      .def_readwrite("commission", &Fill::commission)
      .def_readwrite("timestamp", &Fill::timestamp);

  py::class_<MarketSnapshot>(m, "MarketSnapshot")
      .def(py::init<>())
      .def_readwrite("symbol", &MarketSnapshot::symbol)
      .def_readwrite("timestamp", &MarketSnapshot::timestamp)
      .def_readwrite("open", &MarketSnapshot::open)
      .def_readwrite("high", &MarketSnapshot::high)
      .def_readwrite("low", &MarketSnapshot::low)
      .def_readwrite("close", &MarketSnapshot::close)
      .def_readwrite("volume", &MarketSnapshot::volume)
      .def_readwrite("upper_limit", &MarketSnapshot::upper_limit)
      .def_readwrite("lower_limit", &MarketSnapshot::lower_limit)
      .def_readwrite("is_suspended", &MarketSnapshot::is_suspended)
      .def_readwrite("is_listed", &MarketSnapshot::is_listed)
      .def_readwrite("is_st", &MarketSnapshot::is_st)
      .def_readwrite("lot_size", &MarketSnapshot::lot_size)
      .def_readwrite("min_buy_quantity", &MarketSnapshot::min_buy_quantity)
      .def_readwrite("board", &MarketSnapshot::board)
      .def_readwrite("industry", &MarketSnapshot::industry)
      .def_readwrite("factor_exposures", &MarketSnapshot::factor_exposures)
      .def_readwrite("adjustment_factor", &MarketSnapshot::adjustment_factor)
      .def_readwrite("signal_open", &MarketSnapshot::signal_open)
      .def_readwrite("signal_high", &MarketSnapshot::signal_high)
      .def_readwrite("signal_low", &MarketSnapshot::signal_low)
      .def_readwrite("signal_close", &MarketSnapshot::signal_close)
      .def("signal_ref_price", &MarketSnapshot::signal_ref_price);

  py::class_<OrderRecord>(m, "OrderRecord")
      .def_readonly("order", &OrderRecord::order)
      .def_readonly("filled_quantity", &OrderRecord::filled_quantity)
      .def_readonly("avg_fill_price", &OrderRecord::avg_fill_price)
      .def_readonly("status", &OrderRecord::status)
      .def_readonly("reject_reason", &OrderRecord::reject_reason)
      .def_readonly("updated_timestamp", &OrderRecord::updated_timestamp)
      .def_readonly("message", &OrderRecord::message);

  py::class_<Position>(m, "Position")
      .def_readonly("symbol", &Position::symbol)
      .def_readonly("quantity", &Position::quantity)
      .def_readonly("avg_cost", &Position::avg_cost)
      .def_readonly("realized_pnl", &Position::realized_pnl)
      .def_readonly("sellable_quantity", &Position::sellable_quantity);

  py::class_<CorporateAction>(m, "CorporateAction")
      .def(py::init<>())
      .def_readwrite("symbol", &CorporateAction::symbol)
      .def_readwrite("timestamp", &CorporateAction::timestamp)
      .def_readwrite("cash_dividend_per_share",
                     &CorporateAction::cash_dividend_per_share)
      .def_readwrite("share_multiplier", &CorporateAction::share_multiplier)
      .def_readwrite("description", &CorporateAction::description);

  py::class_<CorporateActionResult>(m, "CorporateActionResult")
      .def_readonly("action_id", &CorporateActionResult::action_id)
      .def_readonly("symbol", &CorporateActionResult::symbol)
      .def_readonly("timestamp", &CorporateActionResult::timestamp)
      .def_readonly("cash_dividend", &CorporateActionResult::cash_dividend)
      .def_readonly("old_quantity", &CorporateActionResult::old_quantity)
      .def_readonly("new_quantity", &CorporateActionResult::new_quantity);

  py::class_<PortfolioSnapshot>(m, "PortfolioSnapshot")
      .def_readonly("cash", &PortfolioSnapshot::cash)
      .def_readonly("equity", &PortfolioSnapshot::equity)
      .def_readonly("gross_exposure", &PortfolioSnapshot::gross_exposure)
      .def_readonly("net_exposure", &PortfolioSnapshot::net_exposure)
      .def_readonly("largest_position_weight",
                    &PortfolioSnapshot::largest_position_weight)
      .def_readonly("position_count", &PortfolioSnapshot::position_count)
      .def_readonly("industry_exposure", &PortfolioSnapshot::industry_exposure)
      .def_readonly("factor_exposure", &PortfolioSnapshot::factor_exposure);

  py::class_<ArrowBridgeStats>(m, "ArrowBridgeStats")
      .def_readonly("rows", &ArrowBridgeStats::rows)
      .def_readonly("batches", &ArrowBridgeStats::batches)
      .def_readonly("bytes", &ArrowBridgeStats::bytes)
      .def_readonly("decode_seconds", &ArrowBridgeStats::decode_seconds)
      .def_readonly("execution_seconds", &ArrowBridgeStats::execution_seconds);

  py::class_<MarketBatchView>(m, "MarketBatchView")
      .def("__len__", &MarketBatchView::size)
      .def("symbol", &MarketBatchView::symbol,
           py::return_value_policy::reference_internal)
      .def("timestamp", &MarketBatchView::timestamp)
      .def("open", &MarketBatchView::open)
      .def("high", &MarketBatchView::high)
      .def("low", &MarketBatchView::low)
      .def("close", &MarketBatchView::close)
      .def("volume", &MarketBatchView::volume);

  py::class_<TradeRecord>(m, "TradeRecord")
      .def_readonly("order_id", &TradeRecord::order_id)
      .def_readonly("symbol", &TradeRecord::symbol)
      .def_readonly("side", &TradeRecord::side)
      .def_readonly("quantity", &TradeRecord::quantity)
      .def_readonly("price", &TradeRecord::price)
      .def_readonly("commission", &TradeRecord::commission)
      .def_readonly("timestamp", &TradeRecord::timestamp);

  py::class_<RoundTripRecord>(m, "RoundTripRecord")
      .def_readonly("symbol", &RoundTripRecord::symbol)
      .def_readonly("entry_side", &RoundTripRecord::entry_side)
      .def_readonly("quantity", &RoundTripRecord::quantity)
      .def_readonly("entry_price", &RoundTripRecord::entry_price)
      .def_readonly("exit_price", &RoundTripRecord::exit_price)
      .def_readonly("opened_at", &RoundTripRecord::opened_at)
      .def_readonly("closed_at", &RoundTripRecord::closed_at)
      .def_readonly("gross_pnl", &RoundTripRecord::gross_pnl)
      .def_readonly("commission", &RoundTripRecord::commission)
      .def_readonly("net_pnl", &RoundTripRecord::net_pnl);

  py::class_<EquityPoint>(m, "EquityPoint")
      .def_readonly("timestamp", &EquityPoint::timestamp)
      .def_readonly("equity", &EquityPoint::equity)
      .def_readonly("cash", &EquityPoint::cash);

  py::class_<BacktestEngine>(m, "BacktestEngine")
      .def(py::init<Price, FillTiming, ExecutionConfig>(),
           py::arg("initial_cash") = 1'000'000.0,
           py::arg("fill_timing") = FillTiming::NEXT_OPEN,
           py::arg("execution_config") = ExecutionConfig{})
      // 便捷入口:Python 侧统一用 push_market_data → run 驱动,无需构造 Event
      .def("push_market_data", &BacktestEngine::push_market_data)
      .def("process_market_data", &BacktestEngine::process_market_data,
           py::call_guard<py::gil_scoped_release>())
      .def("process_market_data_batch",
           &BacktestEngine::process_market_data_batch,
           py::call_guard<py::gil_scoped_release>())
      .def("process_arrow_stream",
           [](BacktestEngine &engine, const py::object &source) {
             return process_arrow_stream(engine, source);
           })
      .def("set_on_market_data", &BacktestEngine::set_on_market_data)
      .def("set_on_cross_section", &BacktestEngine::set_on_cross_section)
      .def("set_on_cross_section_view",
           [](BacktestEngine &engine, const py::object &callback) {
             if (callback.is_none()) {
               engine.set_on_cross_section_view({});
               return;
             }
             py::function function = callback.cast<py::function>();
             engine.set_on_cross_section_view(
                 [function = std::move(function)](
                     std::span<const MarketSnapshot> batch) {
                   py::gil_scoped_acquire acquire;
                   MarketBatchView view(batch);
                   return parse_batch_orders(function(view), batch);
                 });
           })
      .def("set_on_fill", &BacktestEngine::set_on_fill)
      .def("set_on_order_update", &BacktestEngine::set_on_order_update)
#ifdef QBT_ENABLE_PERFORMANCE_ANALYTICS
      .def("set_replay_analytics_sink",
           &BacktestEngine::set_replay_analytics_sink)
      .def("open_performance_period", &BacktestEngine::open_performance_period,
           py::arg("session_id"), py::arg("timestamp") = 0)
      .def("close_performance_period",
           &BacktestEngine::close_performance_period,
           py::arg("session_id"), py::arg("timestamp") = 0,
           py::arg("cash_interest") = 0.0,
           py::arg("external_cash_flow") = 0.0)
#endif
#ifdef QBT_ML_ONNXRUNTIME
      .def(
          "set_model_strategy",
          [](BacktestEngine &engine, const std::string &artifact_path,
             const py::dict &policy_config, const py::dict &risk_config,
             const py::dict &runtime_config) {
            engine_common::StrategySessionContext context;
            engine.set_strategy_runtime(
                load_model_strategy(artifact_path, policy_config, risk_config,
                                    runtime_config, context),
                context);
          },
          py::arg("artifact_path"), py::arg("policy_config") = py::dict(),
          py::arg("risk_config") = py::dict(),
          py::arg("runtime_config") = py::dict())
#endif
      .def("set_commission_fn", &BacktestEngine::set_commission_fn)
      .def("set_fee_schedules", &BacktestEngine::set_fee_schedules)
      .def("set_execution_config", &BacktestEngine::set_execution_config)
      .def("set_history_config", &BacktestEngine::set_history_config)
      .def("run", &BacktestEngine::run,
           py::call_guard<py::gil_scoped_release>())
      .def("cancel_order", &BacktestEngine::cancel_order, py::arg("order_id"),
           py::arg("timestamp") = 0)
      .def("finalize", &BacktestEngine::finalize, py::arg("timestamp") = 0)
      .def("apply_corporate_action", &BacktestEngine::apply_corporate_action)
      .def("get_equity", &BacktestEngine::get_equity)
      .def("get_cash", &BacktestEngine::get_cash)
      .def("get_sharpe_ratio", &BacktestEngine::get_sharpe_ratio)
      .def("get_max_drawdown", &BacktestEngine::get_max_drawdown)
      .def("get_total_return", &BacktestEngine::get_total_return)
      .def("get_annual_return", &BacktestEngine::get_annual_return)
      .def("get_win_rate", &BacktestEngine::get_win_rate)
      .def("get_trade_history", &BacktestEngine::get_trade_history)
      .def("get_trade_history_page", &BacktestEngine::get_trade_history_page)
      .def("get_round_trip_history", &BacktestEngine::get_round_trip_history)
      .def("get_round_trip_history_page",
           &BacktestEngine::get_round_trip_history_page)
      .def("get_equity_curve", &BacktestEngine::get_equity_curve)
      .def("get_equity_curve_page", &BacktestEngine::get_equity_curve_page)
      .def("get_position", &BacktestEngine::get_position)
      .def("get_positions", &BacktestEngine::get_positions)
      .def("get_order_history", &BacktestEngine::get_order_history)
      .def("get_order_history_page", &BacktestEngine::get_order_history_page)
      .def("get_corporate_action_history",
           &BacktestEngine::get_corporate_action_history)
      .def("get_portfolio_snapshot", &BacktestEngine::get_portfolio_snapshot);
}
