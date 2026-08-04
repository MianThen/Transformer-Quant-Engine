#pragma once

#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace qbt {

struct ArrowFieldView {
    std::string_view name;
    std::string_view format;
};

inline void validate_arrow_market_schema(std::span<const ArrowFieldView> fields) {
    std::unordered_map<std::string_view, std::string_view> formats;
    formats.reserve(fields.size());
    for (const ArrowFieldView field : fields) {
        if (field.name.empty() || field.format.empty() ||
            !formats.emplace(field.name, field.format).second) {
            throw std::invalid_argument("invalid or duplicate Arrow field");
        }
    }
    const auto require = [&](std::string_view name, std::string_view expected) {
        const auto found = formats.find(name);
        if (found == formats.end()) {
            throw std::invalid_argument("Arrow stream missing required column: " +
                                        std::string(name));
        }
        if (found->second != expected) {
            throw std::invalid_argument("Arrow column has unexpected format: " +
                                        std::string(name));
        }
    };
    require("timestamp", "l");
    const auto symbol = formats.find("symbol");
    if (symbol == formats.end() || (symbol->second != "u" && symbol->second != "U")) {
        throw std::invalid_argument("Arrow symbol must be utf8 or large_utf8");
    }
    for (std::string_view name : {"open", "high", "low", "close"}) require(name, "g");
    require("volume", "l");
}

}  // namespace qbt
