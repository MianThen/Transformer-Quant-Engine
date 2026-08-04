#include "net/tcp_socket.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <limits>
#include <mutex>
#include <utility>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#if defined(_MSC_VER)
#pragma comment(lib, "ws2_32.lib")
#endif
using socket_t = SOCKET;
static constexpr socket_t kInvalidSocket = INVALID_SOCKET;
#else
#include <fcntl.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>
using socket_t = int;
static constexpr socket_t kInvalidSocket = -1;
#endif

namespace te {
namespace {

socket_t as_socket(intptr_t handle) {
    return static_cast<socket_t>(handle);
}

bool initialize_sockets() {
#if defined(_WIN32)
    static std::once_flag flag;
    static bool initialized = false;
    std::call_once(flag, [] {
        WSADATA data{};
        initialized = WSAStartup(MAKEWORD(2, 2), &data) == 0;
    });
    return initialized;
#else
    return true;
#endif
}

bool would_block() {
#if defined(_WIN32)
    const int error = WSAGetLastError();
    return error == WSAEWOULDBLOCK || error == WSAEINPROGRESS;
#else
    return errno == EAGAIN || errno == EWOULDBLOCK;
#endif
}

int socket_error() {
#if defined(_WIN32)
    return WSAGetLastError();
#else
    return errno;
#endif
}

}  // namespace

TcpSocket::~TcpSocket() { close(); }

TcpSocket::TcpSocket(TcpSocket&& other) noexcept : fd_(other.fd_) {
    last_status_ = other.last_status_;
    last_error_ = other.last_error_;
    other.fd_ = -1;
}

TcpSocket& TcpSocket::operator=(TcpSocket&& other) noexcept {
    if (this != &other) {
        close();
        fd_ = other.fd_;
        last_status_ = other.last_status_;
        last_error_ = other.last_error_;
        other.fd_ = -1;
    }
    return *this;
}

bool TcpSocket::connect(const std::string& host, uint16_t port) {
    close();
    if (host.empty() || port == 0 || !initialize_sockets()) return false;

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    addrinfo* addresses = nullptr;
    const std::string service = std::to_string(port);
    if (getaddrinfo(host.c_str(), service.c_str(), &hints, &addresses) != 0) return false;

    for (addrinfo* address = addresses; address != nullptr; address = address->ai_next) {
        socket_t candidate = ::socket(address->ai_family, address->ai_socktype, address->ai_protocol);
        if (candidate == kInvalidSocket) continue;

        if (::connect(candidate, address->ai_addr, static_cast<int>(address->ai_addrlen)) == 0) {
            fd_ = static_cast<intptr_t>(candidate);
            break;
        }

#if defined(_WIN32)
        closesocket(candidate);
#else
        ::close(candidate);
#endif
    }

    freeaddrinfo(addresses);
    if (!is_open()) return false;
    if (!set_nonblocking(true) || !set_no_delay(true) || !set_keepalive(true) ||
        !set_buffer_sizes(1 << 20, 1 << 20)) {
        close();
        return false;
    }
    return true;
}

bool TcpSocket::set_nonblocking(bool on) {
    if (!is_open()) return false;
#if defined(_WIN32)
    u_long mode = on ? 1UL : 0UL;
    return ioctlsocket(as_socket(fd_), FIONBIO, &mode) == 0;
#else
    const int flags = fcntl(as_socket(fd_), F_GETFL, 0);
    if (flags == -1) return false;
    const int updated = on ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK);
    return fcntl(as_socket(fd_), F_SETFL, updated) == 0;
#endif
}

bool TcpSocket::set_no_delay(bool on) {
    if (!is_open()) return false;
    const int value = on ? 1 : 0;
    return setsockopt(as_socket(fd_), IPPROTO_TCP, TCP_NODELAY,
                      reinterpret_cast<const char*>(&value), sizeof(value)) == 0;
}

bool TcpSocket::set_keepalive(bool on) {
    if (!is_open()) return false;
    const int value = on ? 1 : 0;
    return setsockopt(as_socket(fd_), SOL_SOCKET, SO_KEEPALIVE,
                      reinterpret_cast<const char*>(&value), sizeof(value)) == 0;
}

bool TcpSocket::set_buffer_sizes(int receive_bytes, int send_bytes) {
    if (!is_open() || receive_bytes <= 0 || send_bytes <= 0) return false;
    return setsockopt(as_socket(fd_), SOL_SOCKET, SO_RCVBUF,
                      reinterpret_cast<const char*>(&receive_bytes), sizeof(receive_bytes)) == 0 &&
           setsockopt(as_socket(fd_), SOL_SOCKET, SO_SNDBUF,
                      reinterpret_cast<const char*>(&send_bytes), sizeof(send_bytes)) == 0;
}

ssize_t TcpSocket::recv(void* buf, size_t len) {
    if (!is_open() || buf == nullptr || len == 0) {
        last_status_ = IoStatus::FATAL_ERROR;
        return -1;
    }
    const size_t capped = std::min(len, static_cast<size_t>(std::numeric_limits<int>::max()));
#if defined(_WIN32)
    const int result = ::recv(as_socket(fd_), static_cast<char*>(buf), static_cast<int>(capped), 0);
    if (result == SOCKET_ERROR) {
        last_error_ = socket_error();
        last_status_ = would_block() ? IoStatus::WOULD_BLOCK : IoStatus::FATAL_ERROR;
        return last_status_ == IoStatus::WOULD_BLOCK ? 0 : -1;
    }
#else
    const ssize_t result = ::recv(as_socket(fd_), buf, capped, 0);
    if (result < 0) {
        last_error_ = socket_error();
        last_status_ = would_block() ? IoStatus::WOULD_BLOCK : IoStatus::FATAL_ERROR;
        return last_status_ == IoStatus::WOULD_BLOCK ? 0 : -1;
    }
#endif
    if (result == 0) {
        last_status_ = IoStatus::PEER_CLOSED;
        close();
        return -1;
    }
    last_status_ = IoStatus::OK;
    return static_cast<ssize_t>(result);
}

ssize_t TcpSocket::send(const void* buf, size_t len) {
    if (!is_open() || buf == nullptr || len == 0) {
        last_status_ = IoStatus::FATAL_ERROR;
        return -1;
    }
    const size_t capped = std::min(len, static_cast<size_t>(std::numeric_limits<int>::max()));
#if defined(_WIN32)
    const int result = ::send(as_socket(fd_), static_cast<const char*>(buf), static_cast<int>(capped), 0);
    if (result == SOCKET_ERROR) {
        last_error_ = socket_error();
        last_status_ = would_block() ? IoStatus::WOULD_BLOCK : IoStatus::FATAL_ERROR;
        return last_status_ == IoStatus::WOULD_BLOCK ? 0 : -1;
    }
#else
#if defined(MSG_NOSIGNAL)
    const ssize_t result = ::send(as_socket(fd_), buf, capped, MSG_NOSIGNAL);
#else
    const ssize_t result = ::send(as_socket(fd_), buf, capped, 0);
#endif
    if (result < 0) {
        last_error_ = socket_error();
        last_status_ = would_block() ? IoStatus::WOULD_BLOCK : IoStatus::FATAL_ERROR;
        return last_status_ == IoStatus::WOULD_BLOCK ? 0 : -1;
    }
#endif
    last_status_ = IoStatus::OK;
    return static_cast<ssize_t>(result);
}

ssize_t TcpSocket::recvv(const MutableIoSlice* slices, size_t count) {
    if (!is_open() || slices == nullptr || count == 0) {
        last_status_ = IoStatus::FATAL_ERROR;
        return -1;
    }
    const size_t capped = std::min(count, static_cast<size_t>(64));
#if defined(_WIN32)
    std::array<WSABUF, 64> buffers{};
    for (size_t index = 0; index < capped; ++index) {
        buffers[index].buf = static_cast<char*>(slices[index].data);
        buffers[index].len = static_cast<ULONG>(std::min(
            slices[index].size, static_cast<size_t>(std::numeric_limits<ULONG>::max())));
    }
    DWORD received = 0;
    DWORD flags = 0;
    if (WSARecv(as_socket(fd_), buffers.data(), static_cast<DWORD>(capped),
                &received, &flags, nullptr, nullptr) == SOCKET_ERROR) {
        last_error_ = socket_error();
        last_status_ = would_block() ? IoStatus::WOULD_BLOCK : IoStatus::FATAL_ERROR;
        return last_status_ == IoStatus::WOULD_BLOCK ? 0 : -1;
    }
    const ssize_t result = static_cast<ssize_t>(received);
#else
    std::array<iovec, 64> buffers{};
    for (size_t index = 0; index < capped; ++index) {
        buffers[index] = {slices[index].data, slices[index].size};
    }
    const ssize_t result = ::readv(as_socket(fd_), buffers.data(), static_cast<int>(capped));
    if (result < 0) {
        last_error_ = socket_error();
        last_status_ = would_block() ? IoStatus::WOULD_BLOCK : IoStatus::FATAL_ERROR;
        return last_status_ == IoStatus::WOULD_BLOCK ? 0 : -1;
    }
#endif
    if (result == 0) {
        last_status_ = IoStatus::PEER_CLOSED;
        close();
        return -1;
    }
    last_status_ = IoStatus::OK;
    return result;
}

ssize_t TcpSocket::sendv(const ConstIoSlice* slices, size_t count) {
    if (!is_open() || slices == nullptr || count == 0) {
        last_status_ = IoStatus::FATAL_ERROR;
        return -1;
    }
    const size_t capped = std::min(count, static_cast<size_t>(64));
#if defined(_WIN32)
    std::array<WSABUF, 64> buffers{};
    for (size_t index = 0; index < capped; ++index) {
        buffers[index].buf = const_cast<char*>(static_cast<const char*>(slices[index].data));
        buffers[index].len = static_cast<ULONG>(std::min(
            slices[index].size, static_cast<size_t>(std::numeric_limits<ULONG>::max())));
    }
    DWORD sent = 0;
    if (WSASend(as_socket(fd_), buffers.data(), static_cast<DWORD>(capped),
                &sent, 0, nullptr, nullptr) == SOCKET_ERROR) {
        last_error_ = socket_error();
        last_status_ = would_block() ? IoStatus::WOULD_BLOCK : IoStatus::FATAL_ERROR;
        return last_status_ == IoStatus::WOULD_BLOCK ? 0 : -1;
    }
    const ssize_t result = static_cast<ssize_t>(sent);
#else
    std::array<iovec, 64> buffers{};
    for (size_t index = 0; index < capped; ++index) {
        buffers[index] = {const_cast<void*>(slices[index].data), slices[index].size};
    }
    const ssize_t result = ::writev(as_socket(fd_), buffers.data(), static_cast<int>(capped));
    if (result < 0) {
        last_error_ = socket_error();
        last_status_ = would_block() ? IoStatus::WOULD_BLOCK : IoStatus::FATAL_ERROR;
        return last_status_ == IoStatus::WOULD_BLOCK ? 0 : -1;
    }
#endif
    last_status_ = IoStatus::OK;
    return result;
}

bool TcpSocket::wait_readable(int timeout_ms) {
    if (!is_open()) return false;
#if defined(_WIN32)
    WSAPOLLFD descriptor{as_socket(fd_), POLLRDNORM, 0};
    return WSAPoll(&descriptor, 1, timeout_ms) > 0 &&
           (descriptor.revents & (POLLRDNORM | POLLHUP | POLLERR)) != 0;
#else
    pollfd descriptor{as_socket(fd_), POLLIN, 0};
    return ::poll(&descriptor, 1, timeout_ms) > 0 &&
           (descriptor.revents & (POLLIN | POLLHUP | POLLERR)) != 0;
#endif
}

bool TcpSocket::wait_writable(int timeout_ms) {
    if (!is_open()) return false;
#if defined(_WIN32)
    WSAPOLLFD descriptor{as_socket(fd_), POLLWRNORM, 0};
    return WSAPoll(&descriptor, 1, timeout_ms) > 0 &&
           (descriptor.revents & (POLLWRNORM | POLLHUP | POLLERR)) != 0;
#else
    pollfd descriptor{as_socket(fd_), POLLOUT, 0};
    return ::poll(&descriptor, 1, timeout_ms) > 0 &&
           (descriptor.revents & (POLLOUT | POLLHUP | POLLERR)) != 0;
#endif
}

bool TcpSocket::is_open() const { return fd_ != -1; }

void TcpSocket::close() {
    if (!is_open()) return;
#if defined(_WIN32)
    closesocket(as_socket(fd_));
#else
    ::close(as_socket(fd_));
#endif
    fd_ = -1;
    last_status_ = IoStatus::PEER_CLOSED;
}

}  // namespace te
