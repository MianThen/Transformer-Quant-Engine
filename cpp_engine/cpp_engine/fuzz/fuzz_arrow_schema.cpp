#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "arrow_schema_validation.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (data == nullptr || size == 0) return 0;
    size_t offset = 1;
    const size_t count = data[0] % 32;
    std::vector<std::string> names;
    std::vector<std::string> formats;
    names.reserve(count);
    formats.reserve(count);
    for (size_t index = 0; index < count && offset < size; ++index) {
        const size_t name_length = data[offset++] % 24;
        if (offset + name_length > size) break;
        names.emplace_back(reinterpret_cast<const char*>(data + offset), name_length);
        offset += name_length;
        if (offset >= size) break;
        formats.emplace_back(1, static_cast<char>(data[offset++]));
    }
    std::vector<qbt::ArrowFieldView> fields;
    fields.reserve(names.size());
    for (size_t index = 0; index < names.size(); ++index) {
        fields.push_back({names[index], formats[index]});
    }
    try {
        qbt::validate_arrow_market_schema(fields);
    } catch (const std::exception&) {
    }
    return 0;
}
