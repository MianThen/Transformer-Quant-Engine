#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include "oms/order_gateway.h"

using namespace te;

namespace {
int failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL: %s @ %d\n", #condition, __LINE__); ++failures; \
} } while (0)

class FaultTransport final : public ITransport {
public:
    bool connect(const std::string&, uint16_t) override {
        ++connect_attempts;
        open = connect_attempts > failed_connects;
        return open;
    }
    ssize_t receive(void*, size_t) override { return 0; }
    ssize_t transmit(const void*, size_t size) override {
        ++send_calls;
        if (zero_sends != 0) {
            --zero_sends;
            status = IoStatus::WOULD_BLOCK;
            return 0;
        }
        status = IoStatus::OK;
        const size_t sent = chunk_size == 0 ? size : std::min(size, chunk_size);
        bytes_sent += sent;
        return static_cast<ssize_t>(sent);
    }
    bool wait_readable(int) override { return false; }
    bool wait_writable(int) override { return open; }
    bool is_open() const override { return open; }
    IoStatus last_status() const override { return status; }
    int last_error() const override { return 0; }
    intptr_t native_handle() const override { return -1; }
    void close() override { open = false; }

    bool open = false;
    size_t failed_connects = 0;
    size_t connect_attempts = 0;
    size_t zero_sends = 0;
    size_t chunk_size = 0;
    size_t send_calls = 0;
    size_t bytes_sent = 0;
    IoStatus status = IoStatus::OK;
};

NewOrderRequest request() {
    NewOrderRequest value;
    value.symbol = "AAPL";
    value.quantity = 10;
    value.price = 1'000'000;
    return value;
}

void test_partial_send_and_would_block() {
    auto transport = std::make_unique<FaultTransport>();
    FaultTransport* raw = transport.get();
    raw->zero_sends = 1;
    raw->chunk_size = 7;
    GatewayConfig config;
    config.send_batch_bytes = 1024;
    OrderGateway gateway(config, std::move(transport));
    CHECK(gateway.connect("fault", 1));
    CHECK(gateway.send_order(request()) > 0);
    CHECK(gateway.send_queue_high_watermark() > 0);
    CHECK(gateway.flush());
    CHECK(gateway.send_would_block_count() == 1);
    CHECK(gateway.flush());
    CHECK(raw->bytes_sent > 0);
}

void test_binary_wal_tail_recovery_and_checkpoint() {
    const auto path = std::filesystem::temp_directory_path() / "qbt-wal-fault-test.bin";
    std::filesystem::remove(path);
    GatewayConfig config;
    config.wal_path = path.string();
    config.wal_durability = WalDurability::SYNC_EACH;
    {
        auto transport = std::make_unique<FaultTransport>();
        OrderGateway gateway(config, std::move(transport));
        CHECK(gateway.connect("fault", 1));
        CHECK(gateway.send_order(request()) == 1);
        CHECK(gateway.checkpoint_wal());
    }
    const auto valid_size = std::filesystem::file_size(path);
    {
        std::ofstream output(path, std::ios::binary | std::ios::app);
        const char damaged[] = {1, 2, 3, 4, 5, 6, 7};
        output.write(damaged, sizeof(damaged));
    }
    {
        OrderGateway recovered(config);
        CHECK(recovered.wal_recovered_orders() == 1);
        CHECK(recovered.find_order(1) != nullptr);
    }
    CHECK(std::filesystem::file_size(path) == valid_size);
    std::filesystem::remove(path);
}

void test_disk_failure_blocks_intent_send() {
    GatewayConfig config;
    config.wal_path = (std::filesystem::temp_directory_path() /
                       "missing-qbt-directory" / "orders.wal").string();
    auto transport = std::make_unique<FaultTransport>();
    OrderGateway gateway(config, std::move(transport));
    CHECK(gateway.connect("fault", 1));
    CHECK(gateway.send_order(request()) == -1);
}

void test_deterministic_reconnect() {
    auto transport = std::make_unique<FaultTransport>();
    FaultTransport* raw = transport.get();
    raw->failed_connects = 2;
    size_t sleeps = 0;
    GatewayConfig config;
    config.max_retries = 4;
    config.sleep_for = [&](std::chrono::milliseconds) { ++sleeps; };
    OrderGateway gateway(config, std::move(transport));
    CHECK(!gateway.connect("fault", 1));
    CHECK(gateway.reconnect());
    CHECK(gateway.reconnect_count() == 2);
    CHECK(sleeps == 1);
}

}  // namespace

int main() {
    test_partial_send_and_would_block();
    test_binary_wal_tail_recovery_and_checkpoint();
    test_disk_failure_blocks_intent_send();
    test_deterministic_reconnect();
    if (failures != 0) return 1;
    std::printf("test_gateway_faults: all checks passed\n");
    return 0;
}
