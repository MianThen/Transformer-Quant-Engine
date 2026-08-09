#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

#include "net/tcp_socket.h"
#include "net/event_loop.h"

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
using native_socket_t = SOCKET;
static constexpr native_socket_t kInvalidSocket = INVALID_SOCKET;
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using native_socket_t = int;
static constexpr native_socket_t kInvalidSocket = -1;
#endif

using namespace te;

static int g_failures = 0;
#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL: %s @ %d\n", #cond, __LINE__);       \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

namespace {

bool initialize_native_sockets() {
#if defined(_WIN32)
    WSADATA data{};
    return WSAStartup(MAKEWORD(2, 2), &data) == 0;
#else
    return true;
#endif
}

void close_native(native_socket_t socket) {
#if defined(_WIN32)
    closesocket(socket);
#else
    ::close(socket);
#endif
}

}  // namespace

int main() {
    CHECK(initialize_native_sockets());

    native_socket_t listener = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    CHECK(listener != kInvalidSocket);
    if (listener == kInvalidSocket) return 1;

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    CHECK(::bind(listener, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0);
    CHECK(::listen(listener, 1) == 0);

    socklen_t address_length = sizeof(address);
    CHECK(::getsockname(listener, reinterpret_cast<sockaddr*>(&address), &address_length) == 0);
    const uint16_t port = ntohs(address.sin_port);

    std::thread server([&] {
        native_socket_t peer = ::accept(listener, nullptr, nullptr);
        if (peer == kInvalidSocket) {
            ++g_failures;
            return;
        }

        std::array<char, 4> request{};
        int received = ::recv(peer, request.data(), static_cast<int>(request.size()), 0);
        if (received != static_cast<int>(request.size()) ||
            std::memcmp(request.data(), "ping", request.size()) != 0) {
            ++g_failures;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        const char response[] = "pong";
        if (::send(peer, response, 4, 0) != 4) ++g_failures;
        close_native(peer);
    });

    TcpSocket client;
    CHECK(client.connect("127.0.0.1", port));
    CHECK(client.is_open());
    CHECK(client.native_handle() != -1);

    size_t sent_total = 0;
    const char request_first[] = "pi";
    const char request_second[] = "ng";
    const ConstIoSlice request_slices[]{{request_first, 2}, {request_second, 2}};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (sent_total < 4 && std::chrono::steady_clock::now() < deadline) {
        const ssize_t sent = sent_total == 0
            ? client.sendv(request_slices, 2)
            : client.send("ping" + sent_total, 4 - sent_total);
        CHECK(sent >= 0);
        if (sent < 0) break;
        sent_total += static_cast<size_t>(sent);
        if (sent == 0) std::this_thread::yield();
    }
    CHECK(sent_total == 4);

    std::array<char, 4> response{};
    size_t received_total = 0;
    NativeEventLoop event_loop;
    CHECK(event_loop.add(client, 42));
    std::array<ReadyEvent, 1> ready{};
    CHECK(event_loop.wait(ready, 2'000) == 1);
    CHECK(ready[0].token == 42);
    CHECK(ready[0].readable || ready[0].closed);
    while (received_total < response.size() && std::chrono::steady_clock::now() < deadline) {
        ssize_t received = 0;
        if (received_total == 0) {
            MutableIoSlice slices[]{{response.data(), 2}, {response.data() + 2, 2}};
            received = client.recvv(slices, 2);
        } else {
            received = client.recv(response.data() + received_total,
                                   response.size() - received_total);
        }
        CHECK(received >= 0);
        if (received < 0) break;
        received_total += static_cast<size_t>(received);
        if (received == 0) std::this_thread::yield();
    }
    CHECK(received_total == response.size());
    CHECK(std::memcmp(response.data(), "pong", response.size()) == 0);

    client.close();
    CHECK(!client.is_open());
    server.join();
    close_native(listener);

    if (g_failures == 0) {
        std::printf("test_tcp_socket: connect/send/recv/close passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_tcp_socket: %d failure(s)\n", g_failures);
    return 1;
}
