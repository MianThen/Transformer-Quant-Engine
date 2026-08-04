#include <cstdio>
#include <stdexcept>
#include <vector>

#include "arrow_schema_validation.h"

int main() {
    const std::vector<qbt::ArrowFieldView> valid{
        {"timestamp", "l"}, {"symbol", "u"}, {"open", "g"},
        {"high", "g"}, {"low", "g"}, {"close", "g"}, {"volume", "l"},
    };
    qbt::validate_arrow_market_schema(valid);
    auto invalid = valid;
    invalid.back().format = "g";
    try {
        qbt::validate_arrow_market_schema(invalid);
    } catch (const std::invalid_argument&) {
        std::printf("test_arrow_schema: all checks passed\n");
        return 0;
    }
    return 1;
}
