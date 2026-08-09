#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

#include "engine_common/types.h"

namespace te {

class IOrderAdapter {
public:
    using OnExecutionEvent = std::function<void(const engine_common::ExecutionEvent&)>;

    virtual ~IOrderAdapter() = default;
    virtual bool connect(const std::string& endpoint, uint16_t port) = 0;
    virtual int64_t submit(const engine_common::OrderIntent& intent) = 0;
    virtual bool cancel(int64_t client_order_id) = 0;
    virtual size_t poll() = 0;
    virtual bool ready() const = 0;
    virtual void set_on_execution(OnExecutionEvent callback) = 0;
};

}  // namespace te
