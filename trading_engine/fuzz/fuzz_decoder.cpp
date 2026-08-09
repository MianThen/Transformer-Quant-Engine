#include <cstddef>
#include <cstdint>

#include "feed/decoder.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    te::Decoder decoder;
    decoder.set_on_update([](const te::MarketUpdate&) {});
    decoder.feed(data, size);
    return 0;
}
