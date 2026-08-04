#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "net/tcp_socket.h"

namespace te {

enum class EventLoopBackend : uint8_t { EPOLL, IOCP };

struct ReadyEvent {
    uint64_t token = 0;
    bool readable = false;
    bool closed = false;
    bool error = false;
};

class NativeEventLoop {
public:
    explicit NativeEventLoop(size_t capacity = 64);
    ~NativeEventLoop();
    NativeEventLoop(const NativeEventLoop&) = delete;
    NativeEventLoop& operator=(const NativeEventLoop&) = delete;

    bool add(TcpSocket& socket, uint64_t token);
    bool remove(TcpSocket& socket);
    bool rearm(uint64_t token);
    size_t wait(std::span<ReadyEvent> events, int timeout_ms);
    EventLoopBackend backend() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace te
