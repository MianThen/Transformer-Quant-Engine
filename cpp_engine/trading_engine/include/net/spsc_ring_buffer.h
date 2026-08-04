#pragma once

#include <atomic>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace te {

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winterference-size"
#endif
#if defined(__cpp_lib_hardware_interference_size)
inline constexpr size_t kCacheLineSize = std::hardware_destructive_interference_size;
#else
inline constexpr size_t kCacheLineSize = 64;
#endif
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

// 无锁单生产者单消费者(SPSC)环形缓冲区。
//
// 用途:网络接收线程(生产者)与撮合/策略线程(消费者)之间传递消息,
// 避免锁竞争。热路径,面试重点。
//
// 设计要点:
//   - 容量取 2 的幂,用位与替代取模
//   - head_/tail_ 用 std::atomic,acquire/release 内存序保证可见性
//   - head_ 和 tail_ 分处不同 cache line,避免 false sharing
//
// 单生产者调用 push,单消费者调用 pop;多线程各调各的一端才安全。
template <typename T>
class SpscRingBuffer {
public:
    explicit SpscRingBuffer(size_t capacity_pow2)
        : capacity_(capacity_pow2), mask_(capacity_pow2 - 1), buffer_(capacity_pow2) {
        if (capacity_pow2 <= 1 || (capacity_pow2 & (capacity_pow2 - 1)) != 0) {
            throw std::invalid_argument("SpscRingBuffer capacity must be a power of two greater than one");
        }
    }

    // 生产者:入队。满则返回 false(不覆盖)。
    bool push(const T& item) {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t next = (head + 1) & mask_;
        if (next == producer_cached_tail_) {
            producer_cached_tail_ = tail_.load(std::memory_order_acquire);
            if (next == producer_cached_tail_) return false;
        }
        buffer_[head] = item;
        head_.store(next, std::memory_order_release);
        return true;
    }

    bool push(T&& item) {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t next = (head + 1) & mask_;
        if (next == producer_cached_tail_) {
            producer_cached_tail_ = tail_.load(std::memory_order_acquire);
            if (next == producer_cached_tail_) return false;
        }
        buffer_[head] = std::move(item);
        head_.store(next, std::memory_order_release);
        return true;
    }

    template <typename... Args>
    bool try_emplace(Args&&... args) {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t next = (head + 1) & mask_;
        if (next == producer_cached_tail_) {
            producer_cached_tail_ = tail_.load(std::memory_order_acquire);
            if (next == producer_cached_tail_) return false;
        }
        buffer_[head] = T(std::forward<Args>(args)...);
        head_.store(next, std::memory_order_release);
        return true;
    }

    // 消费者:出队。空则返回 false。
    bool pop(T& out) {
        const size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == consumer_cached_head_) {
            consumer_cached_head_ = head_.load(std::memory_order_acquire);
            if (tail == consumer_cached_head_) return false;
        }
        out = buffer_[tail];
        tail_.store((tail + 1) & mask_, std::memory_order_release);
        return true;
    }

    size_t push_bulk(const T* items, size_t count) {
        size_t pushed = 0;
        while (pushed < count && push(items[pushed])) ++pushed;
        return pushed;
    }

    size_t pop_bulk(T* output, size_t count) {
        size_t popped = 0;
        while (popped < count && pop(output[popped])) ++popped;
        return popped;
    }

    bool empty() const {
        return head_.load(std::memory_order_acquire) ==
               tail_.load(std::memory_order_acquire);
    }

    size_t capacity() const { return capacity_; }
    size_t usable_capacity() const { return capacity_ - 1; }
    size_t size() const {
        const size_t head = head_.load(std::memory_order_acquire);
        const size_t tail = tail_.load(std::memory_order_acquire);
        return (head - tail) & mask_;
    }

private:
    const size_t capacity_;
    const size_t mask_;
    std::vector<T> buffer_;

    alignas(kCacheLineSize) std::atomic<size_t> head_{0};  // 仅生产者写
    alignas(kCacheLineSize) std::atomic<size_t> tail_{0};  // 仅消费者写
    alignas(kCacheLineSize) size_t producer_cached_tail_ = 0;
    alignas(kCacheLineSize) size_t consumer_cached_head_ = 0;
};

}  // namespace te
