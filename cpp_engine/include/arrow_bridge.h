#pragma once

#include <cstdint>

#include <pybind11/pybind11.h>

#include "engine.h"

namespace qbt {

struct ArrowBridgeStats {
    int64_t rows = 0;
    int64_t batches = 0;
    int64_t bytes = 0;
    double decode_seconds = 0.0;
    double execution_seconds = 0.0;
};

ArrowBridgeStats process_arrow_stream(BacktestEngine& engine,
                                      const pybind11::object& source);

}  // namespace qbt
