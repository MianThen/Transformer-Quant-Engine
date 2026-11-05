#ifndef MC_GATE_SHA256_H
#define MC_GATE_SHA256_H

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace mc_gate {

namespace detail {

inline std::uint32_t ror32(std::uint32_t x, unsigned int n) noexcept {
    return (x >> n) | (x << (32u - n));
}

inline void process_block(unsigned int h[8], const unsigned char buf[64]) noexcept {
    static constexpr std::uint32_t K[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };

    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (std::uint32_t(buf[i * 4]) << 24) |
               (std::uint32_t(buf[i * 4 + 1]) << 16) |
               (std::uint32_t(buf[i * 4 + 2]) << 8) |
               (std::uint32_t(buf[i * 4 + 3]));
    }
    for (int i = 16; i < 64; ++i) {
        std::uint32_t s0 = ror32(w[i - 15], 7) ^ ror32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        std::uint32_t s1 = ror32(w[i - 2], 17) ^ ror32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    std::uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];

    for (int i = 0; i < 64; ++i) {
        std::uint32_t S1 = ror32(e, 6) ^ ror32(e, 11) ^ ror32(e, 25);
        std::uint32_t ch = (e & f) ^ ((~e) & g);
        std::uint32_t t1 = hh + S1 + ch + K[i] + w[i];
        std::uint32_t S0 = ror32(a, 2) ^ ror32(a, 13) ^ ror32(a, 22);
        std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        std::uint32_t t2 = S0 + maj;
        hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }

    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

inline std::string to_hex(const unsigned int h[8]) {
    static constexpr char hex_digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (int i = 0; i < 8; ++i) {
        std::uint32_t v = h[i];
        out.push_back(hex_digits[(v >> 28) & 0xF]);
        out.push_back(hex_digits[(v >> 24) & 0xF]);
        out.push_back(hex_digits[(v >> 20) & 0xF]);
        out.push_back(hex_digits[(v >> 16) & 0xF]);
        out.push_back(hex_digits[(v >> 12) & 0xF]);
        out.push_back(hex_digits[(v >> 8) & 0xF]);
        out.push_back(hex_digits[(v >> 4) & 0xF]);
        out.push_back(hex_digits[(v) & 0xF]);
    }
    return out;
}

} // namespace detail

void sha256_init(unsigned int h[8], std::uint64_t& total_len,
                 unsigned char buf[64], std::size_t& buf_len) {
    (void)buf;
    static constexpr unsigned int H0[8] = {
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
    };
    for (int i = 0; i < 8; ++i) h[i] = H0[i];
    total_len = 0;
    buf_len = 0;
}

void sha256_update(unsigned int h[8], std::uint64_t& total_len,
                   unsigned char buf[64], std::size_t& buf_len,
                   const void* data, std::size_t n) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    total_len += n;
    std::size_t i = 0;
    while (i < n) {
        std::size_t space = 64 - buf_len;
        std::size_t take = (n - i) < space ? (n - i) : space;
        std::memcpy(buf + buf_len, p + i, take);
        buf_len += take;
        i += take;
        if (buf_len == 64) {
            detail::process_block(h, buf);
            buf_len = 0;
        }
    }
}

std::string sha256_final(unsigned int h[8], std::uint64_t total_len,
                         unsigned char buf[64], std::size_t& buf_len) {
    // 1) buf[buf_len++] = 0x80;
    buf[buf_len++] = 0x80;
    // 2) if (buf_len > 56) { ... }
    if (buf_len > 56) {
        while (buf_len < 64) buf[buf_len++] = 0x00;
        detail::process_block(h, buf);
        buf_len = 0;
    }
    // 3) while (buf_len < 56) buf[buf_len++] = 0x00;
    while (buf_len < 56) buf[buf_len++] = 0x00;
    // 4) 把 64 位大端的 (total_len * 8) 写到 buf[56..63]
    std::uint64_t bits = total_len * 8;
    for (int i = 0; i < 8; ++i) {
        buf[56 + i] = static_cast<unsigned char>((bits >> (56 - i * 8)) & 0xFF);
    }
    // 5) process_block(h, buf);
    detail::process_block(h, buf);
    buf_len = 0;
    return detail::to_hex(h);
}

std::string sha256_hex(const void* data, std::size_t n) {
    unsigned int h[8];
    std::uint64_t total_len;
    unsigned char buf[64];
    std::size_t buf_len;
    sha256_init(h, total_len, buf, buf_len);
    if (n > 0) {
        sha256_update(h, total_len, buf, buf_len, data, n);
    }
    return sha256_final(h, total_len, buf, buf_len);
}

} // namespace mc_gate

#ifdef MC_GATE_SHA256_SELFTEST

int main() {
    using namespace mc_gate;

    {
        std::string hex = sha256_hex("", 0);
        std::printf("SHA empty %s\n", hex.c_str());
    }
    {
        std::string hex = sha256_hex("abc", 3);
        std::printf("SHA abc %s\n", hex.c_str());
    }
    {
        const char* s = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        std::string hex = sha256_hex(s, 56);
        std::printf("SHA 448 %s\n", hex.c_str());
    }
    {
        std::vector<unsigned char> v(1000000, 'a');
        std::string hex = sha256_hex(v.data(), v.size());
        std::printf("SHA million_a %s\n", hex.c_str());
    }
    {
        std::vector<unsigned char> v(256);
        for (std::size_t i = 0; i < 256; ++i) v[i] = static_cast<unsigned char>(i);
        std::string hex = sha256_hex(v.data(), v.size());
        std::printf("SHA bytes256 %s\n", hex.c_str());
    }
    {
        std::vector<unsigned char> v(1024);
        for (std::size_t i = 0; i < 1024; ++i) v[i] = static_cast<unsigned char>(i % 256);
        std::string hex = sha256_hex(v.data(), v.size());
        std::printf("SHA bytes256x4 %s\n", hex.c_str());
    }

    std::printf("SHA256_DONE\n");
    return 0;
}

#endif // MC_GATE_SHA256_SELFTEST

#endif // MC_GATE_SHA256_H
