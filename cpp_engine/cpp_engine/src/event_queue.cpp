#include "event_queue.h"

namespace qbt {

void EventQueue::push(const Event& event) {
    queue_.push(event);
}

const Event& EventQueue::top() const {
    return queue_.top();
}

Event EventQueue::pop() {
    Event top = queue_.top();
    queue_.pop();
    return top;
}

bool EventQueue::empty() const {
    return queue_.empty();
}

size_t EventQueue::size() const {
    return queue_.size();
}

}  // namespace qbt
