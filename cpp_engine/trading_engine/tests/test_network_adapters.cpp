#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "adapter/mock_order_adapter.h"
#include "feed/feed_handler.h"
#include "feed/protocol.h"
#include "transport/transport.h"

using namespace te;

namespace {

int failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL: %s @ %d\n", #condition, __LINE__); ++failures; \
} } while (0)

class FakeTransport final : public ITransport {
public:
    bool connect(const std::string&, uint16_t) override {
        open_ = true;
        return true;
    }

    ssize_t receive(void* data, size_t size) override {
        if (!open_) return -1;
        if (read_offset_ == incoming_.size()) {
            status_ = IoStatus::WOULD_BLOCK;
            return 0;
        }
        const size_t count = std::min(size, incoming_.size() - read_offset_);
        std::memcpy(data, incoming_.data() + read_offset_, count);
        read_offset_ += count;
        status_ = IoStatus::OK;
        return static_cast<ssize_t>(count);
    }

    ssize_t transmit(const void* data, size_t size) override {
        if (!open_) return -1;
        const auto* bytes = static_cast<const uint8_t*>(data);
        outgoing_.insert(outgoing_.end(), bytes, bytes + size);
        status_ = IoStatus::OK;
        return static_cast<ssize_t>(size);
    }

    bool wait_readable(int) override { return read_offset_ < incoming_.size(); }
    bool wait_writable(int) override { return open_; }
    bool is_open() const override { return open_; }
    IoStatus last_status() const override { return status_; }
    int last_error() const override { return 0; }
    intptr_t native_handle() const override { return -1; }
    void close() override { open_ = false; }

    void append_incoming(const void* data, size_t size) {
        const auto* bytes = static_cast<const uint8_t*>(data);
        incoming_.insert(incoming_.end(), bytes, bytes + size);
    }

    const std::vector<uint8_t>& outgoing() const { return outgoing_; }

private:
    bool open_ = false;
    IoStatus status_ = IoStatus::OK;
    std::vector<uint8_t> incoming_;
    std::vector<uint8_t> outgoing_;
    size_t read_offset_ = 0;
};

QuoteMsg make_quote() {
    QuoteMsg message{};
    message.header.type = static_cast<uint8_t>(MsgType::QUOTE);
    message.header.length = host_to_be16(static_cast<uint16_t>(sizeof(message)));
    message.header.seq_num = host_to_be32(1);
    message.header.timestamp = host_to_be64(123456ULL);
    std::memcpy(message.symbol, "AAPL", 4);
    message.bid_price = host_to_be64(from_price(100.25));
    message.ask_price = host_to_be64(from_price(100.50));
    message.bid_size = host_to_be32(10);
    message.ask_size = host_to_be32(20);
    return message;
}

void test_transport_and_market_adapter_injection() {
    auto transport = std::make_unique<FakeTransport>();
    FakeTransport* fake = transport.get();
    const QuoteMsg quote = make_quote();
    fake->append_incoming(&quote, sizeof(quote));
    FeedHandler feed(8, BackpressurePolicy::FAIL_FAST, 100,
                     std::move(transport));
    CHECK(feed.connect("mock-feed", 1));
    CHECK(feed.poll_once() == sizeof(quote));
    engine_common::MarketEvent event{};
    CHECK(feed.try_pop(event));
    CHECK(feed.symbol(event.symbol_id) == "AAPL");
    CHECK(event.bid == from_price(100.25));
    CHECK(event.ask == from_price(100.50));
    CHECK(event.bid_quantity == 10);
    CHECK(event.ask_quantity == 20);
    CHECK(event.sequence == 1);
}

void test_gateway_transport_injection() {
    auto transport = std::make_unique<FakeTransport>();
    FakeTransport* fake = transport.get();
    OrderGateway gateway({}, std::move(transport));
    CHECK(gateway.connect("mock-gateway", 2));
    NewOrderRequest request{};
    request.symbol = "AAPL";
    request.quantity = 10;
    request.price = from_price(100.25);
    CHECK(gateway.send_order(request) > 0);
    CHECK(gateway.flush());
    CHECK(!fake->outgoing().empty());
}

void test_order_adapter_mapping() {
    MockOrderAdapter adapter([](engine_common::SymbolId id) {
        return id == 7 ? std::string("AAPL") : std::string();
    });
    engine_common::OrderIntent intent{};
    intent.client_order_id = 42;
    intent.symbol_id = 7;
    intent.side = engine_common::Side::SELL;
    intent.quantity = 25;
    intent.limit_price = 123400;
    const NewOrderRequest request = adapter.to_gateway_request(intent);
    CHECK(request.client_order_id == 42);
    CHECK(request.symbol == "AAPL");
    CHECK(request.side == 1);
    CHECK(request.quantity == 25);
    CHECK(request.price == 123400);

    const ExecReport report{42, 9, OrderStatus::PARTIALLY_FILLED,
                            5, 10, 123400,
                            engine_common::RejectReason::NONE};
    const auto event = MockOrderAdapter::to_internal_execution(report);
    CHECK(event.client_order_id == 42);
    CHECK(event.execution_id == 9);
    CHECK(event.status == engine_common::ExecutionStatus::PARTIALLY_FILLED);
    CHECK(event.last_quantity == 5);
    CHECK(event.cumulative_quantity == 10);
    CHECK(event.last_price == 123400);
}

}  // namespace

int main() {
    test_transport_and_market_adapter_injection();
    test_gateway_transport_injection();
    test_order_adapter_mapping();
    if (failures != 0) return 1;
    std::printf("test_network_adapters: all checks passed\n");
    return 0;
}
