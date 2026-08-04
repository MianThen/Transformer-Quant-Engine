#include <cstddef>
#include <cstdint>

#include "feed/decoder.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    te::Decoder decoder;
    size_t offset = 0;
    while (offset < size) {
        const size_t chunk = 1 + data[offset] % 64;
        const size_t available = size - offset;
        decoder.feed(data + offset, chunk < available ? chunk : available);
        offset += chunk < available ? chunk : available;
    }
    decoder.reset_sequence();
    return 0;
}
