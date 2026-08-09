#pragma once

#include "adapter/market_data_adapter.h"
#include "engine_common/symbol_registry.h"
#include "feed/decoder.h"

namespace te {

class MockMarketDataAdapter final : public IMarketDataAdapter {
public:
    MockMarketDataAdapter();

    void set_on_event(OnMarketEvent callback) override;
    size_t feed(const uint8_t* data, size_t size) override;
    void reset() override;
    bool has_unresolved_gap() const override;
    uint64_t gap_count() const override;
    uint64_t recovered_gap_count() const override;
    uint64_t permanent_gap_count() const override;
    uint64_t duplicate_count() const override;
    uint64_t malformed_count() const override;
    std::string_view symbol(engine_common::SymbolId symbol_id) const override;

private:
    void on_update(const MarketUpdate& update);

    Decoder decoder_;
    engine_common::SymbolRegistry symbols_;
    OnMarketEvent on_event_;
};

}  // namespace te
