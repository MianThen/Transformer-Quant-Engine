#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "transport/io_types.h"

namespace te {

class ITransport {
public:
    virtual ~ITransport() = default;

    virtual bool connect(const std::string& endpoint, uint16_t port) = 0;
    virtual ssize_t receive(void* data, size_t size) = 0;
    virtual ssize_t transmit(const void* data, size_t size) = 0;
    virtual bool wait_readable(int timeout_ms) = 0;
    virtual bool wait_writable(int timeout_ms) = 0;
    virtual bool is_open() const = 0;
    virtual IoStatus last_status() const = 0;
    virtual int last_error() const = 0;
    virtual intptr_t native_handle() const = 0;
    virtual void close() = 0;
};

}  // namespace te
