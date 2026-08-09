#pragma once

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#include <BaseTsd.h>
using ssize_t = SSIZE_T;
#endif

namespace te {

enum class IoStatus : uint8_t {
    OK,
    WOULD_BLOCK,
    PEER_CLOSED,
    FATAL_ERROR,
};

struct ConstIoSlice { const void* data = nullptr; size_t size = 0; };
struct MutableIoSlice { void* data = nullptr; size_t size = 0; };

}  // namespace te
