#pragma once

#include <atomic>
#include <cstdint>
#include <string_view>

namespace te {

enum class RunMode : uint8_t { LOW_LATENCY, BALANCED, POWER_SAVE };

const char* to_string(RunMode mode);
bool parse_run_mode(std::string_view text, RunMode& mode);
void idle_wait(RunMode mode, uint32_t idle_count);
bool pin_current_thread(unsigned cpu_index);
bool set_current_thread_realtime(bool enabled);
int64_t steady_now_ns();

struct ThreadHealth {
    std::atomic<bool> alive{false};
    std::atomic<int64_t> heartbeat_ns{0};

    void started();
    void beat();
    void stopped();
    bool responsive(int64_t now_ns, int64_t timeout_ns) const;
};

struct ProcessMetrics {
    double cpu_seconds = 0.0;
    uint64_t resident_bytes = 0;
};

ProcessMetrics process_metrics();

}  // namespace te
