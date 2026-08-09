#include <cstdio>

#include "engine_common/symbol_registry.h"

int main() {
    engine_common::SymbolRegistry registry;
    const auto first = registry.intern("AAA");
    const auto second = registry.intern("BBB");
    if (first != 0 || second != 1 || registry.intern("AAA") != first ||
        registry.symbol(second) != "BBB" || registry.size() != 2) {
        std::fprintf(stderr, "test_symbol_registry failed\n");
        return 1;
    }
    std::printf("test_symbol_registry: all checks passed\n");
    return 0;
}
