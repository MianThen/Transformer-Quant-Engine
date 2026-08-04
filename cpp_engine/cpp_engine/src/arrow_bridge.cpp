#include "arrow_bridge.h"
#include "arrow_schema_validation.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace qbt {
namespace {

struct ArrowSchema {
    const char* format;
    const char* name;
    const char* metadata;
    int64_t flags;
    int64_t n_children;
    ArrowSchema** children;
    ArrowSchema* dictionary;
    void (*release)(ArrowSchema*);
    void* private_data;
};

struct ArrowArray {
    int64_t length;
    int64_t null_count;
    int64_t offset;
    int64_t n_buffers;
    int64_t n_children;
    const void** buffers;
    ArrowArray** children;
    ArrowArray* dictionary;
    void (*release)(ArrowArray*);
    void* private_data;
};

struct ArrowArrayStream {
    int (*get_schema)(ArrowArrayStream*, ArrowSchema*);
    int (*get_next)(ArrowArrayStream*, ArrowArray*);
    const char* (*get_last_error)(ArrowArrayStream*);
    void (*release)(ArrowArrayStream*);
    void* private_data;
};

struct StreamGuard {
    ArrowArrayStream* value;
    ~StreamGuard() {
        if (value != nullptr && value->release != nullptr) value->release(value);
    }
};

struct SchemaGuard {
    ArrowSchema* value;
    ~SchemaGuard() {
        if (value->release != nullptr) value->release(value);
    }
};

struct ArrayGuard {
    ArrowArray* value;
    ~ArrayGuard() {
        if (value->release != nullptr) value->release(value);
    }
};

struct Column {
    int64_t index = -1;
    std::string format;
};

using Columns = std::unordered_map<std::string, Column>;
using Clock = std::chrono::steady_clock;

std::runtime_error stream_error(ArrowArrayStream* stream,
                                const std::string& operation) {
    const char* detail = stream->get_last_error == nullptr
        ? nullptr : stream->get_last_error(stream);
    return std::runtime_error(operation + (detail == nullptr ? "" : ": " + std::string(detail)));
}

Columns read_columns(const ArrowSchema& schema) {
    Columns columns;
    std::vector<ArrowFieldView> fields;
    fields.reserve(static_cast<size_t>(std::max<int64_t>(schema.n_children, 0)));
    for (int64_t index = 0; index < schema.n_children; ++index) {
        const ArrowSchema* child = schema.children[index];
        if (child == nullptr || child->name == nullptr || child->format == nullptr) continue;
        columns.emplace(child->name, Column{index, child->format});
        fields.push_back({child->name, child->format});
    }
    validate_arrow_market_schema(fields);
    return columns;
}

const ArrowArray& child_array(const ArrowArray& batch, const Column& column) {
    if (column.index < 0 || column.index >= batch.n_children ||
        batch.children[column.index] == nullptr) {
        throw std::invalid_argument("Arrow stream child array mismatch");
    }
    return *batch.children[column.index];
}

bool is_valid(const ArrowArray& array, int64_t row) {
    if (array.null_count == 0 || array.buffers[0] == nullptr) return true;
    const int64_t physical = array.offset + row;
    const auto* bitmap = static_cast<const uint8_t*>(array.buffers[0]);
    return (bitmap[physical / 8] & (1U << (physical % 8))) != 0;
}

int64_t read_int64(const ArrowArray& batch, const Column& column,
                   int64_t row, const char* name) {
    const ArrowArray& array = child_array(batch, column);
    if (!is_valid(array, row)) {
        throw std::invalid_argument(std::string(name) + " cannot be null");
    }
    if (column.format != "l") {
        throw std::invalid_argument(std::string(name) + " must be Arrow int64");
    }
    return static_cast<const int64_t*>(array.buffers[1])[array.offset + row];
}

double read_double(const ArrowArray& batch, const Column& column,
                   int64_t row, const char* name) {
    const ArrowArray& array = child_array(batch, column);
    if (!is_valid(array, row)) {
        throw std::invalid_argument(std::string(name) + " cannot be null");
    }
    if (column.format != "g") {
        throw std::invalid_argument(std::string(name) + " must be Arrow float64");
    }
    return static_cast<const double*>(array.buffers[1])[array.offset + row];
}

double read_optional_double(const ArrowArray& batch, const Column& column,
                            int64_t row, const char* name, double fallback) {
    const ArrowArray& array = child_array(batch, column);
    return is_valid(array, row)
        ? read_double(batch, column, row, name) : fallback;
}

int64_t read_optional_int64(const ArrowArray& batch, const Column& column,
                            int64_t row, const char* name, int64_t fallback) {
    const ArrowArray& array = child_array(batch, column);
    return is_valid(array, row)
        ? read_int64(batch, column, row, name) : fallback;
}

bool read_bool(const ArrowArray& batch, const Column& column,
               int64_t row, bool fallback) {
    const ArrowArray& array = child_array(batch, column);
    if (!is_valid(array, row)) return fallback;
    if (column.format != "b") throw std::invalid_argument("Arrow boolean column expected");
    const int64_t physical = array.offset + row;
    const auto* bitmap = static_cast<const uint8_t*>(array.buffers[1]);
    return (bitmap[physical / 8] & (1U << (physical % 8))) != 0;
}

std::string read_string(const ArrowArray& batch, const Column& column,
                        int64_t row, const char* name, bool nullable = false) {
    const ArrowArray& array = child_array(batch, column);
    if (!is_valid(array, row)) {
        if (nullable) return "";
        throw std::invalid_argument(std::string(name) + " cannot be null");
    }
    const int64_t physical = array.offset + row;
    int64_t begin = 0;
    int64_t end = 0;
    if (column.format == "u") {
        const auto* offsets = static_cast<const int32_t*>(array.buffers[1]);
        begin = offsets[physical];
        end = offsets[physical + 1];
    } else if (column.format == "U") {
        const auto* offsets = static_cast<const int64_t*>(array.buffers[1]);
        begin = offsets[physical];
        end = offsets[physical + 1];
    } else {
        throw std::invalid_argument(std::string(name) + " must be Arrow utf8");
    }
    const auto* data = static_cast<const char*>(array.buffers[2]);
    return std::string(data + begin, static_cast<size_t>(end - begin));
}

const Column* optional_column(const Columns& columns, const char* name) {
    auto it = columns.find(name);
    return it == columns.end() ? nullptr : &it->second;
}

MarketSnapshot decode_row(const ArrowArray& batch, const Columns& columns,
                          int64_t row, int64_t& estimated_bytes) {
    MarketSnapshot md;
    md.timestamp = read_int64(batch, columns.at("timestamp"), row, "timestamp");
    md.symbol = read_string(batch, columns.at("symbol"), row, "symbol");
    md.open = read_double(batch, columns.at("open"), row, "open");
    md.high = read_double(batch, columns.at("high"), row, "high");
    md.low = read_double(batch, columns.at("low"), row, "low");
    md.close = read_double(batch, columns.at("close"), row, "close");
    md.volume = read_int64(batch, columns.at("volume"), row, "volume");
    estimated_bytes += 48 + static_cast<int64_t>(md.symbol.size());

    if (const Column* value = optional_column(columns, "upper_limit"))
        md.upper_limit = read_optional_double(
            batch, *value, row, "upper_limit", 0.0);
    if (const Column* value = optional_column(columns, "lower_limit"))
        md.lower_limit = read_optional_double(
            batch, *value, row, "lower_limit", 0.0);
    if (const Column* value = optional_column(columns, "is_suspended"))
        md.is_suspended = read_bool(batch, *value, row, false);
    if (const Column* value = optional_column(columns, "is_listed"))
        md.is_listed = read_bool(batch, *value, row, true);
    if (const Column* value = optional_column(columns, "is_st"))
        md.is_st = read_bool(batch, *value, row, false);
    if (const Column* value = optional_column(columns, "lot_size"))
        md.lot_size = read_optional_int64(batch, *value, row, "lot_size", 1);
    if (const Column* value = optional_column(columns, "min_buy_quantity"))
        md.min_buy_quantity = read_optional_int64(
            batch, *value, row, "min_buy_quantity", 1);
    if (const Column* value = optional_column(columns, "board"))
        md.board = read_string(batch, *value, row, "board", true);
    if (const Column* value = optional_column(columns, "industry"))
        md.industry = read_string(batch, *value, row, "industry", true);
    if (const Column* value = optional_column(columns, "adjustment_factor"))
        md.adjustment_factor = read_optional_double(
            batch, *value, row, "adjustment_factor", 1.0);
    if (const Column* value = optional_column(columns, "signal_open"))
        md.signal_open = read_optional_double(batch, *value, row, "signal_open", 0.0);
    if (const Column* value = optional_column(columns, "signal_high"))
        md.signal_high = read_optional_double(batch, *value, row, "signal_high", 0.0);
    if (const Column* value = optional_column(columns, "signal_low"))
        md.signal_low = read_optional_double(batch, *value, row, "signal_low", 0.0);
    if (const Column* value = optional_column(columns, "signal_close"))
        md.signal_close = read_optional_double(batch, *value, row, "signal_close", 0.0);

    constexpr std::string_view factor_prefix = "factor_exposure__";
    for (const auto& [name, column] : columns) {
        if (name.size() <= factor_prefix.size() ||
            name.compare(0, factor_prefix.size(), factor_prefix) != 0) {
            continue;
        }
        const ArrowArray& values = child_array(batch, column);
        if (!is_valid(values, row)) continue;
        md.factor_exposures.emplace(
            name.substr(factor_prefix.size()),
            read_double(batch, column, row, name.c_str()));
    }
    return md;
}

double seconds_between(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

}  // namespace

ArrowBridgeStats process_arrow_stream(BacktestEngine& engine,
                                      const py::object& source) {
    if (!py::hasattr(source, "__arrow_c_stream__")) {
        throw std::invalid_argument("object does not implement __arrow_c_stream__");
    }
    py::capsule capsule = source.attr("__arrow_c_stream__")();
    auto* stream = static_cast<ArrowArrayStream*>(capsule.get_pointer());
    if (stream == nullptr || stream->release == nullptr) {
        throw std::invalid_argument("invalid ArrowArrayStream capsule");
    }
    StreamGuard stream_guard{stream};

    ArrowSchema schema{};
    if (stream->get_schema(stream, &schema) != 0) {
        throw stream_error(stream, "ArrowArrayStream.get_schema failed");
    }
    SchemaGuard schema_guard{&schema};
    const Columns columns = read_columns(schema);

    ArrowBridgeStats stats;
    std::vector<MarketSnapshot> cross_section;
    Timestamp current_timestamp = 0;
    std::pair<Timestamp, std::string> last_key{0, ""};
    bool has_last_key = false;

    {
        py::gil_scoped_release release;
        while (true) {
            ArrowArray array{};
            if (stream->get_next(stream, &array) != 0) {
                throw stream_error(stream, "ArrowArrayStream.get_next failed");
            }
            if (array.release == nullptr) break;
            ArrayGuard array_guard{&array};
            ++stats.batches;
            for (int64_t row = 0; row < array.length; ++row) {
                const auto decode_start = Clock::now();
                MarketSnapshot md = decode_row(array, columns, row, stats.bytes);
                stats.decode_seconds += seconds_between(decode_start, Clock::now());
                const std::pair<Timestamp, std::string> key{md.timestamp, md.symbol};
                if (has_last_key && key <= last_key) {
                    throw std::invalid_argument(
                        "Arrow replay must be strictly ordered by (timestamp, symbol)");
                }
                if (!cross_section.empty() && md.timestamp != current_timestamp) {
                    const auto execution_start = Clock::now();
                    engine.process_market_data_batch_sorted(cross_section);
                    stats.execution_seconds += seconds_between(execution_start, Clock::now());
                    cross_section.clear();
                }
                current_timestamp = md.timestamp;
                cross_section.push_back(std::move(md));
                last_key = key;
                has_last_key = true;
                ++stats.rows;
            }
        }
        if (!cross_section.empty()) {
            const auto execution_start = Clock::now();
            engine.process_market_data_batch_sorted(cross_section);
            stats.execution_seconds += seconds_between(execution_start, Clock::now());
        }
    }
    return stats;
}

}  // namespace qbt
