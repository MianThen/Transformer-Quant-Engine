#include <cstdio>

#include "net/latency.h"
#include "util/object_pool.h"

using namespace te;

static int g_failures = 0;
#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL: %s @ %d\n", #cond, __LINE__);       \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

int main() {
    LatencyRecorder latency;
    latency.record(10);
    latency.record(20);
    latency.record(30);
    latency.record(40);
    CHECK(latency.percentile(0.50) >= 20 && latency.percentile(0.50) <= 31);
    CHECK(latency.percentile(0.99) >= 40 && latency.percentile(0.99) <= 63);
    CHECK(latency.max() == 40);
    CHECK(latency.mean() == 25.0);
    const LatencySnapshot snapshot = latency.snapshot_and_reset();
    CHECK(snapshot.count == 4);
    CHECK(latency.count() == 0);

    ObjectPool<int> pool(2);
    int* first = pool.allocate();
    int* second = pool.allocate();
    CHECK(first != nullptr);
    CHECK(second != nullptr);
    CHECK(pool.allocate() == nullptr);
    pool.deallocate(first);
    pool.deallocate(first);
    CHECK(pool.available() == 1);
    CHECK(pool.allocate() == first);

    if (g_failures == 0) {
        std::printf("test_utilities: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_utilities: %d failure(s)\n", g_failures);
    return 1;
}
