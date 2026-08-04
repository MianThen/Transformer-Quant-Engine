#pragma once

#include <string>
#include <variant>

#include "market_data.h"
#include "order.h"
#include "types.h"

namespace qbt {

enum class EventType {
    MARKET_DATA,  // 行情更新
    SIGNAL,       // 策略信号
    ORDER,        // 订单提交
    FILL,         // 成交回报
    TIMER,        // 定时器(如收盘平仓)
};

// 事件载荷:用 std::variant 取代 shared_ptr<void>。
//   - 无堆分配(载荷内联在 Event 里),cache 友好
//   - 编译期类型安全,取错类型直接编译报错而非运行期 UB
//   - std::monostate 对应 TIMER/SIGNAL 这类无载荷事件
using EventPayload = std::variant<std::monostate, MarketSnapshot, Order, Fill>;

// 事件:按时间戳排序驱动回测
struct Event {
    EventType type = EventType::TIMER;
    Timestamp timestamp = 0;
    std::string symbol;
    EventPayload data;  // 按 type 取对应 alternative
};

// 事件优先级:先按时间戳升序,同时刻 MARKET_DATA 优先(先更新行情再决策)
struct EventComparator {
    bool operator()(const Event& a, const Event& b) const {
        if (a.timestamp != b.timestamp) return a.timestamp > b.timestamp;
        return static_cast<int>(a.type) > static_cast<int>(b.type);
    }
};

}  // namespace qbt
