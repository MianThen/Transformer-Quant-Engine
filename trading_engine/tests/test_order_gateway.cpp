#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

#include "oms/order_gateway.h"

using namespace te;

static int failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL: %s @ %d\n", #condition, __LINE__); ++failures; \
} } while (0)

int main() {
    const auto wal = std::filesystem::temp_directory_path() / "qbt-gateway-test.wal";
    std::filesystem::remove(wal);
    {
        OrderGateway gateway;
        CHECK(gateway.send_order(NewOrderRequest{}) == -1);
        CHECK(gateway.rejected_not_ready() == 1);
    }
    {
        std::ofstream output(wal);
        output << "1,1,INTENT_NEW,7,100,12\n";
        output << "2,1,REPORT,7,40,3\n";
        output << "3,1,INTENT_NEW,8,50,12\n";
    }
    {
        GatewayConfig config;
        config.wal_path = wal.string();
        OrderGateway gateway(config);
        CHECK(gateway.wal_recovered_orders() == 2);
        CHECK(gateway.find_order(7) != nullptr);
        CHECK(gateway.find_order(7)->filled() == 40);
        CHECK(gateway.find_order(7)->status() == OrderStatus::PARTIALLY_FILLED);

        const std::vector<ExchangeOrderSnapshot> snapshots{
            {7, 100, 60, OrderStatus::PARTIALLY_FILLED},
            {9, 25, 0, OrderStatus::ACK},
        };
        const std::vector<ExecReport> reports{
            {7, 1001, OrderStatus::FILLED, 40, 100, 1234,
             engine_common::RejectReason::NONE},
        };
        const ReconciliationResult result = gateway.reconcile(snapshots, reports);
        CHECK(result.matched == 1);
        CHECK(result.recovered == 1);
        CHECK(result.local_unknown == 1);
        CHECK(result.reports_applied == 1);
        CHECK(gateway.find_order(7)->status() == OrderStatus::FILLED);
        CHECK(gateway.find_order(8)->status() == OrderStatus::UNKNOWN);
        CHECK(gateway.find_order(9)->status() == OrderStatus::ACK);
        gateway.process_report(reports.front());
        CHECK(gateway.duplicate_reports() >= 1);
    }
    std::filesystem::remove(wal);
    if (failures != 0) return 1;
    std::printf("test_order_gateway: all checks passed\n");
    return 0;
}
