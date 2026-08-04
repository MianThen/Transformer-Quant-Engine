#pragma once

#include "net/tcp_socket.h"
#include "transport/transport.h"

namespace te {

class TcpTransport final : public ITransport {
public:
    bool connect(const std::string& endpoint, uint16_t port) override;
    ssize_t receive(void* data, size_t size) override;
    ssize_t transmit(const void* data, size_t size) override;
    bool wait_readable(int timeout_ms) override;
    bool wait_writable(int timeout_ms) override;
    bool is_open() const override;
    IoStatus last_status() const override;
    int last_error() const override;
    intptr_t native_handle() const override;
    void close() override;

    TcpSocket& socket() { return socket_; }
    const TcpSocket& socket() const { return socket_; }

private:
    TcpSocket socket_;
};

}  // namespace te
