#include "runtime/runtime.h"

#include <chrono>
#include <cstdio>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#else
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace te {

const char* to_string(RunMode mode) {
    switch (mode) {
        case RunMode::LOW_LATENCY: return "low-latency";
        case RunMode::BALANCED: return "balanced";
        case RunMode::POWER_SAVE: return "power-save";
    }
    return "unknown";
}

bool parse_run_mode(std::string_view text, RunMode& mode) {
    if (text == "low-latency") mode = RunMode::LOW_LATENCY;
    else if (text == "balanced") mode = RunMode::BALANCED;
    else if (text == "power-save") mode = RunMode::POWER_SAVE;
    else return false;
    return true;
}

void idle_wait(RunMode mode, uint32_t idle_count) {
    if (mode == RunMode::LOW_LATENCY) {
        std::atomic_signal_fence(std::memory_order_seq_cst);
        return;
    }
    if (mode == RunMode::BALANCED && idle_count < 256) {
        std::atomic_signal_fence(std::memory_order_seq_cst);
        return;
    }
    if (mode == RunMode::BALANCED) std::this_thread::yield();
    else std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

bool pin_current_thread(unsigned cpu_index) {
#if defined(_WIN32)
    if (cpu_index >= sizeof(DWORD_PTR) * 8) return false;
    return SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR{1} << cpu_index) != 0;
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu_index, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#endif
}

bool set_current_thread_realtime(bool enabled) {
    if (!enabled) return true;
#if defined(_WIN32)
    return SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL) != 0;
#else
    sched_param parameters{};
    parameters.sched_priority = sched_get_priority_max(SCHED_FIFO);
    return pthread_setschedparam(pthread_self(), SCHED_FIFO, &parameters) == 0;
#endif
}

int64_t steady_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

void ThreadHealth::started() {
    alive.store(true, std::memory_order_release);
    beat();
}

void ThreadHealth::beat() {
    heartbeat_ns.store(steady_now_ns(), std::memory_order_release);
}

void ThreadHealth::stopped() {
    beat();
    alive.store(false, std::memory_order_release);
}

bool ThreadHealth::responsive(int64_t now_ns, int64_t timeout_ns) const {
    return alive.load(std::memory_order_acquire) &&
           now_ns - heartbeat_ns.load(std::memory_order_acquire) <= timeout_ns;
}

ProcessMetrics process_metrics() {
    ProcessMetrics metrics;
#if defined(_WIN32)
    FILETIME creation{}, exit{}, kernel{}, user{};
    if (GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user)) {
        ULARGE_INTEGER kernel_value{}, user_value{};
        kernel_value.LowPart = kernel.dwLowDateTime;
        kernel_value.HighPart = kernel.dwHighDateTime;
        user_value.LowPart = user.dwLowDateTime;
        user_value.HighPart = user.dwHighDateTime;
        metrics.cpu_seconds = static_cast<double>(kernel_value.QuadPart + user_value.QuadPart) / 1e7;
    }
    PROCESS_MEMORY_COUNTERS counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        metrics.resident_bytes = static_cast<uint64_t>(counters.WorkingSetSize);
    }
#else
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        metrics.cpu_seconds = usage.ru_utime.tv_sec + usage.ru_stime.tv_sec +
            (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / 1e6;
#if defined(__APPLE__)
        metrics.resident_bytes = static_cast<uint64_t>(usage.ru_maxrss);
#else
        metrics.resident_bytes = static_cast<uint64_t>(usage.ru_maxrss) * 1024;
#endif
    }
#endif
    return metrics;
}

}  // namespace te
