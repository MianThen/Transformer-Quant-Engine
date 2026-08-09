#pragma once

#include <array>
#include <chrono>
#include <cstdint>

namespace te {

struct LatencySnapshot {
    uint64_t count = 0;
    int64_t minimum = 0;
    int64_t maximum = 0;
    double mean = 0.0;
    int64_t p50 = 0;
    int64_t p90 = 0;
    int64_t p99 = 0;
    int64_t p999 = 0;
    int64_t p9999 = 0;
};

class LatencyRecorder {
public:
    using clock = std::chrono::steady_clock;
    using ns = std::chrono::nanoseconds;

    explicit LatencyRecorder(size_t = 0) {}
    void record(int64_t nanos);

    class ScopedTimer {
    public:
        explicit ScopedTimer(LatencyRecorder& recorder)
            : recorder_(recorder), start_(clock::now()) {}
        ~ScopedTimer() {
            recorder_.record(std::chrono::duration_cast<ns>(clock::now() - start_).count());
        }
    private:
        LatencyRecorder& recorder_;
        clock::time_point start_;
    };

    ScopedTimer scoped() { return ScopedTimer(*this); }
    int64_t percentile(double probability) const;
    int64_t min() const { return count_ == 0 ? 0 : minimum_; }
    int64_t max() const { return maximum_; }
    double mean() const;
    size_t count() const { return static_cast<size_t>(count_); }
    LatencySnapshot snapshot() const;
    LatencySnapshot snapshot_and_reset();
    void clear();

private:
    static constexpr size_t kBucketCount = 64;
    std::array<uint64_t, kBucketCount> buckets_{};
    uint64_t count_ = 0;
    long double sum_ = 0.0;
    int64_t minimum_ = 0;
    int64_t maximum_ = 0;
};

}  // namespace te
