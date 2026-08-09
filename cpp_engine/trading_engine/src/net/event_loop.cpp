#include "net/event_loop.h"

#include <algorithm>
#include <vector>

#if defined(_WIN32)
#include <winsock2.h>
#include <windows.h>
#elif defined(__APPLE__)
#include <poll.h>
#include <unistd.h>
#else
#include <sys/epoll.h>
#include <unistd.h>
#endif

namespace te {

struct NativeEventLoop::Impl {
#if defined(_WIN32)
    struct Registration {
        OVERLAPPED overlapped{};
        WSABUF buffer{};
        SOCKET socket = INVALID_SOCKET;
        uint64_t token = 0;
        bool occupied = false;
        bool armed = false;
    };
    HANDLE port = nullptr;
    std::vector<Registration> registrations;
#elif defined(__APPLE__)
    struct Registration {
        pollfd descriptor{};
        uint64_t token = 0;
    };
    std::vector<Registration> registrations;
#else
    int descriptor = -1;
    std::vector<epoll_event> native_events;
#endif
};

NativeEventLoop::NativeEventLoop(size_t capacity) : impl_(std::make_unique<Impl>()) {
#if defined(_WIN32)
    impl_->port = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 0);
    impl_->registrations.resize(std::max<size_t>(capacity, 1));
#elif defined(__APPLE__)
    impl_->registrations.reserve(std::max<size_t>(capacity, 1));
#else
    impl_->descriptor = epoll_create1(EPOLL_CLOEXEC);
    impl_->native_events.resize(std::max<size_t>(capacity, 1));
#endif
}

NativeEventLoop::~NativeEventLoop() {
#if defined(_WIN32)
    if (impl_->port != nullptr) CloseHandle(impl_->port);
#elif defined(__APPLE__)
    impl_->registrations.clear();
#else
    if (impl_->descriptor >= 0) ::close(impl_->descriptor);
#endif
}

bool NativeEventLoop::add(TcpSocket& socket, uint64_t token) {
    if (!socket.is_open()) return false;
#if defined(_WIN32)
    if (impl_->port == nullptr) return false;
    auto found = std::find_if(impl_->registrations.begin(), impl_->registrations.end(),
                              [](const Impl::Registration& item) { return !item.occupied; });
    if (found == impl_->registrations.end()) return false;
    found->socket = static_cast<SOCKET>(socket.native_handle());
    found->token = token;
    found->occupied = true;
    if (CreateIoCompletionPort(reinterpret_cast<HANDLE>(found->socket), impl_->port,
                               reinterpret_cast<ULONG_PTR>(&*found), 0) == nullptr) {
        found->occupied = false;
        return false;
    }
    return rearm(token);
#elif defined(__APPLE__)
    const auto found = std::find_if(
        impl_->registrations.begin(), impl_->registrations.end(),
        [&](const Impl::Registration& item) {
            return item.descriptor.fd == static_cast<int>(socket.native_handle());
        });
    if (found != impl_->registrations.end()) return false;
    impl_->registrations.push_back({
        pollfd{static_cast<int>(socket.native_handle()), POLLIN, 0}, token});
    return true;
#else
    if (impl_->descriptor < 0) return false;
    epoll_event event{};
    event.events = EPOLLIN | EPOLLRDHUP | EPOLLERR;
    event.data.u64 = token;
    return epoll_ctl(impl_->descriptor, EPOLL_CTL_ADD,
                     static_cast<int>(socket.native_handle()), &event) == 0;
#endif
}

bool NativeEventLoop::remove(TcpSocket& socket) {
#if defined(_WIN32)
    const SOCKET value = static_cast<SOCKET>(socket.native_handle());
    auto found = std::find_if(impl_->registrations.begin(), impl_->registrations.end(),
                              [&](const Impl::Registration& item) {
                                  return item.occupied && item.socket == value;
                              });
    if (found == impl_->registrations.end()) return false;
    CancelIoEx(reinterpret_cast<HANDLE>(found->socket), &found->overlapped);
    *found = Impl::Registration{};
    return true;
#elif defined(__APPLE__)
    const auto found = std::find_if(
        impl_->registrations.begin(), impl_->registrations.end(),
        [&](const Impl::Registration& item) {
            return item.descriptor.fd == static_cast<int>(socket.native_handle());
        });
    if (found == impl_->registrations.end()) return false;
    impl_->registrations.erase(found);
    return true;
#else
    return impl_->descriptor >= 0 &&
           epoll_ctl(impl_->descriptor, EPOLL_CTL_DEL,
                     static_cast<int>(socket.native_handle()), nullptr) == 0;
#endif
}

bool NativeEventLoop::rearm(uint64_t token) {
#if defined(_WIN32)
    auto found = std::find_if(impl_->registrations.begin(), impl_->registrations.end(),
                              [&](const Impl::Registration& item) {
                                  return item.occupied && item.token == token;
                              });
    if (found == impl_->registrations.end() || found->armed) return found != impl_->registrations.end();
    found->overlapped = OVERLAPPED{};
    found->buffer = WSABUF{};
    DWORD flags = 0;
    DWORD received = 0;
    const int result = WSARecv(found->socket, &found->buffer, 1, &received, &flags,
                               &found->overlapped, nullptr);
    if (result == SOCKET_ERROR && WSAGetLastError() != WSA_IO_PENDING) return false;
    found->armed = true;
    return true;
#elif defined(__APPLE__)
    (void)token;
    return true;
#else
    (void)token;
    return true;
#endif
}

size_t NativeEventLoop::wait(std::span<ReadyEvent> events, int timeout_ms) {
    if (events.empty()) return 0;
#if defined(_WIN32)
    if (impl_->port == nullptr) return 0;
    DWORD bytes = 0;
    ULONG_PTR key = 0;
    OVERLAPPED* overlapped = nullptr;
    const BOOL ok = GetQueuedCompletionStatus(
        impl_->port, &bytes, &key, &overlapped,
        timeout_ms < 0 ? INFINITE : static_cast<DWORD>(timeout_ms));
    if (overlapped == nullptr) return 0;
    auto* registration = reinterpret_cast<Impl::Registration*>(key);
    registration->armed = false;
    events[0] = {registration->token, ok != 0, bytes == 0 && ok == 0, ok == 0};
    return 1;
#elif defined(__APPLE__)
    const size_t limit = std::min(events.size(), impl_->registrations.size());
    if (limit == 0) return 0;
    std::vector<pollfd> descriptors;
    descriptors.reserve(limit);
    for (size_t index = 0; index < limit; ++index) {
        descriptors.push_back(impl_->registrations[index].descriptor);
    }
    const int count = ::poll(descriptors.data(), static_cast<nfds_t>(limit),
                             timeout_ms);
    if (count <= 0) return 0;
    size_t output = 0;
    for (size_t index = 0; index < limit && output < events.size(); ++index) {
        const short flags = descriptors[index].revents;
        if (flags == 0) continue;
        events[output++] = {
            impl_->registrations[index].token,
            (flags & POLLIN) != 0,
            (flags & (POLLHUP | POLLNVAL)) != 0,
            (flags & POLLERR) != 0,
        };
    }
    return output;
#else
    if (impl_->descriptor < 0) return 0;
    const int count = epoll_wait(impl_->descriptor, impl_->native_events.data(),
                                 static_cast<int>(std::min(events.size(), impl_->native_events.size())),
                                 timeout_ms);
    if (count <= 0) return 0;
    for (int index = 0; index < count; ++index) {
        const uint32_t flags = impl_->native_events[index].events;
        events[static_cast<size_t>(index)] = {
            impl_->native_events[index].data.u64,
            (flags & EPOLLIN) != 0,
            (flags & (EPOLLRDHUP | EPOLLHUP)) != 0,
            (flags & EPOLLERR) != 0,
        };
    }
    return static_cast<size_t>(count);
#endif
}

EventLoopBackend NativeEventLoop::backend() const {
#if defined(_WIN32)
    return EventLoopBackend::IOCP;
#elif defined(__APPLE__)
    return EventLoopBackend::POLL;
#else
    return EventLoopBackend::EPOLL;
#endif
}

}  // namespace te
