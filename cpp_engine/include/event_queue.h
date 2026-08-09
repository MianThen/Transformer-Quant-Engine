#pragma once

#include <queue>
#include <vector>

#include "event.h"

namespace qbt {

// 事件优先级队列:按时间戳排序驱动回测
class EventQueue {
public:
    void push(const Event& event);
    const Event& top() const;
    Event pop();
    bool empty() const;
    size_t size() const;

private:
    std::priority_queue<Event, std::vector<Event>, EventComparator> queue_;
};

}  // namespace qbt
