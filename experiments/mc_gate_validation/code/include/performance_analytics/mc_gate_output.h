#ifndef MC_GATE_OUTPUT_H
#define MC_GATE_OUTPUT_H

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <fstream>
#include <numeric>
#include <map>

namespace mc_gate {

struct OutputRecord {
    long long cell_id;
    long long rep_id;
    std::string rule_id;
    bool selected;
    long long index;
    bool abstain;
    std::uint64_t returns_sha256;
    std::string reason;
};

std::uint32_t crc32_ieee(const void* data, std::size_t n) {
    std::uint32_t crc = 0xFFFFFFFFu;
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n; ++i) {
        crc ^= p[i];
        for (int k = 0; k < 8; ++k) {
            if (crc & 1u) {
                crc = (crc >> 1) ^ 0xEDB88320u;
            } else {
                crc >>= 1;
            }
        }
    }
    return crc ^ 0xFFFFFFFFu;
}

std::vector<unsigned char> gzip_store(const void* data, std::size_t n, std::uint32_t mtime) {
    std::vector<unsigned char> out;
    out.reserve(10 + n + 8);

    // Header
    out.push_back(0x1f);
    out.push_back(0x8b);
    out.push_back(0x08);
    out.push_back(0x00);

    // MTIME (4 bytes, little-endian)
    out.push_back(static_cast<unsigned char>(mtime & 0xFF));
    out.push_back(static_cast<unsigned char>((mtime >> 8) & 0xFF));
    out.push_back(static_cast<unsigned char>((mtime >> 16) & 0xFF));
    out.push_back(static_cast<unsigned char>((mtime >> 24) & 0xFF));

    // XFL
    out.push_back(0x00);
    // OS
    out.push_back(0xff);

    const auto* src = static_cast<const unsigned char*>(data);
    std::size_t offset = 0;
    bool first_block = true;

    while (offset < n) {
        std::size_t chunk = n - offset;
        if (chunk > 65535) chunk = 65535;

        unsigned char bfinal_btype = first_block ? 0x00 : 0x00;
        if (offset + chunk >= n) {
            bfinal_btype = 0x01;
        }
        out.push_back(bfinal_btype);

        std::uint16_t len = static_cast<std::uint16_t>(chunk);
        std::uint16_t nlen = static_cast<std::uint16_t>(~len);

        out.push_back(static_cast<unsigned char>(len & 0xFF));
        out.push_back(static_cast<unsigned char>((len >> 8) & 0xFF));
        out.push_back(static_cast<unsigned char>(nlen & 0xFF));
        out.push_back(static_cast<unsigned char>((nlen >> 8) & 0xFF));

        out.insert(out.end(), src + offset, src + offset + chunk);
        offset += chunk;
        first_block = false;
    }

    // If n == 0, we still need a valid gzip stream? The spec says "after each block".
    // If n==0, no blocks. But gzip requires at least one block?
    // Actually, if n==0, the loop doesn't run.
    // Let's assume n > 0 for this task based on context (JSON lines).
    // But to be safe, if n==0, we might need an empty stored block?
    // The spec says "after each block". If there are no blocks, is it valid?
    // Standard gzip allows empty input.
    // However, the loop above handles n>0.
    // If n==0, we skip the loop.
    
    // Wait, if n==0, the while loop doesn't execute.
    // Is a gzip file with just header and trailer valid?
    // Yes, it represents an empty stream.
    // But let's check the logic. If n==0, offset=0, loop skipped.
    // Then we go to trailer.

    // Trailer
    std::uint32_t crc = crc32_ieee(data, n);
    std::uint32_t isize = static_cast<std::uint32_t>(n);

    out.push_back(static_cast<unsigned char>(crc & 0xFF));
    out.push_back(static_cast<unsigned char>((crc >> 8) & 0xFF));
    out.push_back(static_cast<unsigned char>((crc >> 16) & 0xFF));
    out.push_back(static_cast<unsigned char>((crc >> 24) & 0xFF));

    out.push_back(static_cast<unsigned char>(isize & 0xFF));
    out.push_back(static_cast<unsigned char>((isize >> 8) & 0xFF));
    out.push_back(static_cast<unsigned char>((isize >> 16) & 0xFF));
    out.push_back(static_cast<unsigned char>((isize >> 24) & 0xFF));

    return out;
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '"') {
            out += '\\';
            out += '"';
        } else if (c == '\\') {
            out += '\\';
            out += '\\';
        } else if (c == '\n') {
            out += '\\';
            out += 'n';
        } else if (c == '\t') {
            out += '\\';
            out += 't';
        } else {
            out += c;
        }
    }
    return out;
}

std::string decision_json_line(const OutputRecord& r) {
    std::string line;
    line += "{\"cell_id\":";
    line += std::to_string(r.cell_id);
    line += ",\"rep_id\":";
    line += std::to_string(r.rep_id);
    line += ",\"rule_id\":\"";
    line += json_escape(r.rule_id);
    line += "\",\"selected\":";
    line += (r.selected ? "true" : "false");
    line += ",\"index\":";
    line += std::to_string(r.index);
    line += ",\"abstain\":";
    line += (r.abstain ? "true" : "false");
    line += ",\"returns_sha256\":\"";
    { char buf[32]; std::snprintf(buf, sizeof(buf), "%016llx",
        static_cast<unsigned long long>(r.returns_sha256)); line += buf; }
    line += "\",\"reason\":\"";
    line += json_escape(r.reason);
    line += "\"}";
    return line;
}

bool write_decisions_jsonl_gz(const char* path, const std::vector<OutputRecord>& recs) {
    std::string payload;
    for (const auto& r : recs) {
        payload += decision_json_line(r);
        payload += '\n';
    }

    auto gz = gzip_store(payload.data(), payload.size(), 0);

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        return false;
    }

    ofs.write(reinterpret_cast<const char*>(gz.data()), static_cast<std::streamsize>(gz.size()));
    ofs.close();

    return ofs.good();
}

} // namespace mc_gate

#ifdef MC_GATE_OUTPUT_SELFTEST

int main() {
    using namespace mc_gate;

    // 1) CRC
    std::uint32_t crc_val = crc32_ieee("123456789", 9);
    std::printf("CRC 0x%08X\n", crc_val);

    // Records
    std::vector<OutputRecord> recs;
    recs.push_back({101, 0, "GATE_Full", false, 1, false, 0x92b4f2a290c24fe4ULL, "below_threshold"});
    recs.push_back({101, 0, "B0", true, 1, false, 0x92b4f2a290c24fe4ULL, "baseline"});
    recs.push_back({205, 2, "B4", true, 4, true, 0x0123456789abcdefULL, "ocn_abstain"});

    // Write to file
    write_decisions_jsonl_gz("/tmp/qgate/C08/out.jsonl.gz", recs);

    // Prepare payload for size check and gzip header check
    std::string payload;
    for (const auto& r : recs) {
        payload += decision_json_line(r);
        payload += '\n';
    }

    auto gz = gzip_store(payload.data(), payload.size(), 0);

    // 2) GZIP_HDR
    std::printf("GZIP_HDR");
    for (std::size_t i = 0; i < 10 && i < gz.size(); ++i) {
        std::printf(" %02x", gz[i]);
    }
    std::printf("\n");

    // 3) GZIP_SIZE
    std::printf("GZIP_SIZE %zu\n", gz.size());

    // 4) PAYLOAD_SIZE
    std::printf("PAYLOAD_SIZE %zu\n", payload.size());

    // 5) JSONL
    for (const auto& r : recs) {
        std::printf("JSONL %s\n", decision_json_line(r).c_str());
    }

    // 6) ESC
    std::string s;
    s += 'q';
    s += '"';
    s += 'w';
    s += '\\';
    s += 'e';
    std::printf("ESC %s\n", json_escape(s).c_str());

    // 7) OUTPUT_DONE
    std::printf("OUTPUT_DONE\n");

    return 0;
}

#endif // MC_GATE_OUTPUT_SELFTEST

#endif // MC_GATE_OUTPUT_H
