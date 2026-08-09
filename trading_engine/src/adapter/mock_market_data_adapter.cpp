#include "adapter/mock_market_data_adapter.h"

#include <string_view>

#include "feed/protocol.h"

namespace te {

namespace {

std::string_view symbol_view(const char symbol[8]) {
    size_t length = 0;
    while (length < 8 && symbol[length] != '\0') ++length;
    return {symbol, length};
}

}  // namespace

MockMarketDataAdapter::MockMarketDataAdapter() {
    decoder_.set_on_update([this](const MarketUpdate& update) { on_update(update); });
}

void MockMarketDataAdapter::set_on_event(OnMarketEvent callback) {
    on_event_ = std::move(callback);
}

size_t MockMarketDataAdapter::feed(const uint8_t* data, size_t size) {
    return decoder_.feed(data, size);
}

void MockMarketDataAdapter::reset() { decoder_.reset_sequence(); }
bool MockMarketDataAdapter::has_unresolved_gap() const { return decoder_.has_unresolved_gap(); }
uint64_t MockMarketDataAdapter::gap_count() const { return decoder_.gap_count(); }
uint64_t MockMarketDataAdapter::recovered_gap_count() const { return decoder_.recovered_gap_count(); }
uint64_t MockMarketDataAdapter::permanent_gap_count() const { return decoder_.permanent_gap_count(); }
uint64_t MockMarketDataAdapter::duplicate_count() const { return decoder_.duplicate_count(); }
uint64_t MockMarketDataAdapter::malformed_count() const { return decoder_.malformed_count(); }

std::string_view MockMarketDataAdapter::symbol(engine_common::SymbolId symbol_id) const {
    return symbols_.symbol(symbol_id);
}

void MockMarketDataAdapter::on_update(const MarketUpdate& update) {
    if (!on_event_) return;
    engine_common::MarketEvent event{};
    event.symbol_id = symbols_.intern(symbol_view(update.symbol));
    event.timestamp = update.timestamp;
    event.bid = from_price(update.bid);
    event.ask = from_price(update.ask);
    event.bid_quantity = update.bid_size;
    event.ask_quantity = update.ask_size;
    event.sequence = update.seq_num;
    on_event_(event);
}

}  // namespace te
