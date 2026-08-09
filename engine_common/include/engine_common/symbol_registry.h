#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "engine_common/types.h"

namespace engine_common {

struct TransparentStringHash {
    using is_transparent = void;
    size_t operator()(std::string_view value) const noexcept {
        return std::hash<std::string_view>{}(value);
    }
    size_t operator()(const std::string& value) const noexcept {
        return (*this)(std::string_view(value));
    }
};

class SymbolRegistry {
public:
    SymbolId intern(std::string_view symbol) {
        if (symbol.empty()) throw std::invalid_argument("symbol cannot be empty");
        const auto found = ids_.find(symbol);
        if (found != ids_.end()) return found->second;
        const SymbolId id = static_cast<SymbolId>(symbols_.size());
        symbols_.emplace_back(symbol);
        ids_.emplace(symbols_.back(), id);
        return id;
    }

    bool find(std::string_view symbol, SymbolId& id) const {
        const auto found = ids_.find(symbol);
        if (found == ids_.end()) return false;
        id = found->second;
        return true;
    }

    const std::string& symbol(SymbolId id) const {
        if (id >= symbols_.size()) throw std::out_of_range("invalid SymbolId");
        return symbols_[id];
    }

    size_t size() const { return symbols_.size(); }

private:
    std::unordered_map<std::string, SymbolId, TransparentStringHash,
                       std::equal_to<>> ids_;
    std::vector<std::string> symbols_;
};

}  // namespace engine_common
