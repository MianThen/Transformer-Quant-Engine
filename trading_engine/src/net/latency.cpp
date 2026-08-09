#include "net/latency.h"

#include <algorithm>
#include <bit>
#include <cmath>

namespace te {

void LatencyRecorder::record(int64_t nanos) {
    const uint64_t value = static_cast<uint64_t>(std::max<int64_t>(nanos, 0));
    const size_t bucket = value == 0 ? 0 : std::min<size_t>(std::bit_width(value), kBucketCount - 1);
    ++buckets_[bucket];
    ++count_;
    sum_ += static_cast<long double>(value);
    if (count_ == 1) minimum_ = static_cast<int64_t>(value);
    else minimum_ = std::min(minimum_, static_cast<int64_t>(value));
    maximum_ = std::max(maximum_, static_cast<int64_t>(value));
}

int64_t LatencyRecorder::percentile(double probability) const {
    if (count_ == 0 || !std::isfinite(probability)) return 0;
    probability = std::clamp(probability, 0.0, 1.0);
    const uint64_t target = std::max<uint64_t>(
        1, static_cast<uint64_t>(std::ceil(probability * static_cast<double>(count_))));
    uint64_t cumulative = 0;
    for (size_t bucket = 0; bucket < buckets_.size(); ++bucket) {
        cumulative += buckets_[bucket];
        if (cumulative >= target) {
            if (bucket == 0) return 0;
            if (bucket >= 63) return maximum_;
            return static_cast<int64_t>((uint64_t{1} << bucket) - 1);
        }
    }
    return maximum_;
}

double LatencyRecorder::mean() const {
    return count_ == 0 ? 0.0 : static_cast<double>(sum_ / count_);
}

LatencySnapshot LatencyRecorder::snapshot() const {
    return {count_, min(), max(), mean(), percentile(0.50), percentile(0.90),
            percentile(0.99), percentile(0.999), percentile(0.9999)};
}

LatencySnapshot LatencyRecorder::snapshot_and_reset() {
    const LatencySnapshot value = snapshot();
    clear();
    return value;
}

void LatencyRecorder::clear() {
    buckets_.fill(0);
    count_ = 0;
    sum_ = 0.0;
    minimum_ = 0;
    maximum_ = 0;
}

}  // namespace te
