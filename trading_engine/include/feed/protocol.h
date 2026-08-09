#pragma once

#include <cstring>
#include <cstdint>
#include <type_traits>

namespace te {

// Mock 二进制行情协议(模仿 ITCH 风格的定长消息)。
//
// 该协议只属于 MockCodec 测试后端。真实 ITCH/OUCH/FIX/CTP 必须通过独立
// Session 和 Adapter 接入,不能通过修改本文件替换整个网络栈。
//
// 面试 talking point:零拷贝解码——直接把接收缓冲区里的字节
// reinterpret 成消息结构体(需处理对齐和字节序)。

#pragma pack(push, 1)  // 紧凑排列,匹配网络线格式

enum class MsgType : uint8_t {
    HEARTBEAT = 0,
    QUOTE = 1,       // 盘口报价
    TRADE = 2,       // 逐笔成交
};

// 定长消息头
struct MsgHeader {
    uint8_t type;         // MsgType
    uint8_t _pad;
    uint16_t length;      // 整条消息字节数(含头)
    uint32_t seq_num;     // 序列号,用于乱序检测/重排
    uint64_t timestamp;   // 交易所时间戳(纳秒)
};

// 报价消息体
struct QuoteMsg {
    MsgHeader header;
    char symbol[8];       // 定长标的代码,右侧补 '\0'
    int64_t bid_price;    // 定点数:实际价 * 10000
    int64_t ask_price;
    int32_t bid_size;
    int32_t ask_size;
};

// 成交消息体
struct TradeMsg {
    MsgHeader header;
    char symbol[8];
    int64_t price;        // 定点数
    int32_t size;
    uint8_t side;         // 0=买 1=卖
    uint8_t _pad[3];
};

#pragma pack(pop)

// 定点价格 ↔ double 换算(避免网络传输浮点)
constexpr int64_t kPriceScale = 10000;
inline double to_price(int64_t fixed) { return static_cast<double>(fixed) / kPriceScale; }
inline int64_t from_price(double p) { return static_cast<int64_t>(p * kPriceScale); }

inline uint16_t byte_swap16(uint16_t value) {
    return static_cast<uint16_t>((value << 8) | (value >> 8));
}

inline uint32_t byte_swap32(uint32_t value) {
    return ((value & 0x000000ffU) << 24) |
           ((value & 0x0000ff00U) << 8) |
           ((value & 0x00ff0000U) >> 8) |
           ((value & 0xff000000U) >> 24);
}

inline uint64_t byte_swap64(uint64_t value) {
    return (static_cast<uint64_t>(byte_swap32(static_cast<uint32_t>(value))) << 32) |
           byte_swap32(static_cast<uint32_t>(value >> 32));
}

inline bool host_is_little_endian() {
    const uint16_t value = 1;
    return *reinterpret_cast<const uint8_t*>(&value) == 1;
}

inline uint16_t host_to_be16(uint16_t value) {
    return host_is_little_endian() ? byte_swap16(value) : value;
}

inline uint32_t host_to_be32(uint32_t value) {
    return host_is_little_endian() ? byte_swap32(value) : value;
}

inline uint64_t host_to_be64(uint64_t value) {
    return host_is_little_endian() ? byte_swap64(value) : value;
}

inline uint16_t be16_to_host(uint16_t value) { return host_to_be16(value); }
inline uint32_t be32_to_host(uint32_t value) { return host_to_be32(value); }
inline uint64_t be64_to_host(uint64_t value) { return host_to_be64(value); }

inline int32_t host_to_be32(int32_t value) {
    return static_cast<int32_t>(host_to_be32(static_cast<uint32_t>(value)));
}

inline int64_t host_to_be64(int64_t value) {
    return static_cast<int64_t>(host_to_be64(static_cast<uint64_t>(value)));
}

inline int32_t be32_to_host(int32_t value) { return host_to_be32(value); }
inline int64_t be64_to_host(int64_t value) { return host_to_be64(value); }

template <typename T>
T read_unaligned(const void* data) {
    static_assert(std::is_trivially_copyable<T>::value, "wire values must be trivially copyable");
    T value{};
    std::memcpy(&value, data, sizeof(T));
    return value;
}

}  // namespace te
