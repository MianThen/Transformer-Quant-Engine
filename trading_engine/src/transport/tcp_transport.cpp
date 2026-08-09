#include "transport/tcp_transport.h"

namespace te {

bool TcpTransport::connect(const std::string& endpoint, uint16_t port) {
    return socket_.connect(endpoint, port);
}

ssize_t TcpTransport::receive(void* data, size_t size) {
    return socket_.recv(data, size);
}

ssize_t TcpTransport::transmit(const void* data, size_t size) {
    return socket_.send(data, size);
}

bool TcpTransport::wait_readable(int timeout_ms) {
    return socket_.wait_readable(timeout_ms);
}

bool TcpTransport::wait_writable(int timeout_ms) {
    return socket_.wait_writable(timeout_ms);
}

bool TcpTransport::is_open() const { return socket_.is_open(); }
IoStatus TcpTransport::last_status() const { return socket_.last_status(); }
int TcpTransport::last_error() const { return socket_.last_error(); }
intptr_t TcpTransport::native_handle() const { return socket_.native_handle(); }
void TcpTransport::close() { socket_.close(); }

}  // namespace te
