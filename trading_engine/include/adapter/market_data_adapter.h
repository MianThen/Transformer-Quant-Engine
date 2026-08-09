#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string_view>

#include "engine_common/types.h"

namespace te {

class IMarketDataAdapter {
public:
    using OnMarketEvent = std::function<void(const engine_common::MarketEvent&)>;

    virtual ~IMarketDataAdapter() = default;
    virtual void set_on_event(OnMarketEvent callback) = 0;
    virtual size_t feed(const uint8_t* data, size_t size) = 0;
    virtual void reset() = 0;
    virtual bool has_unresolved_gap() const = 0;
    virtual uint64_t gap_count() const = 0;
    virtual uint64_t recovered_gap_count() const = 0;
    virtual uint64_t permanent_gap_count() const = 0;
    virtual uint64_t duplicate_count() const = 0;
    virtual uint64_t malformed_count() const = 0;
    virtual std::string_view symbol(engine_common::SymbolId symbol_id) const = 0;
};

}  // namespace te
