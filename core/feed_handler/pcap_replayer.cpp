#include "pcap_replayer.hpp"
#include <stdexcept>
#include <fstream>
#include <vector>
#include <cstring>

namespace sigma_edge {
namespace itch {

// ── host-endian helpers ───────────────────────────────────────────────────────

static inline uint16_t le16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) |
           (static_cast<uint16_t>(p[1]) << 8);
}
static inline uint32_t le32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0])       |
           (static_cast<uint32_t>(p[1]) <<  8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}
static inline uint16_t be16(const uint8_t* p) {
    return (static_cast<uint16_t>(p[0]) << 8) |
            static_cast<uint16_t>(p[1]);
}

// ── PcapReplayer ──────────────────────────────────────────────────────────────

PcapReplayer::PcapReplayer(const std::string& path) : path_(path) {
    // Validate the file exists and has at least the global header.
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open())
        throw std::runtime_error("PcapReplayer: cannot open file: " + path);
    auto size = f.tellg();
    if (size < 24)
        throw std::runtime_error("PcapReplayer: file too small to be a pcap: " + path);
}

ReplayStats PcapReplayer::replay(const MessageCallback& cb) {
    std::ifstream f(path_, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("PcapReplayer: cannot open file: " + path_);

    // Slurp entire file into memory — typical ITCH pcap files are processed
    // in segments anyway; for production use mmap.
    f.seekg(0, std::ios::end);
    const size_t file_size = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(file_size);
    if (!f.read(reinterpret_cast<char*>(data.data()),
                static_cast<std::streamsize>(file_size)))
        throw std::runtime_error("PcapReplayer: read error: " + path_);

    return replay_buffer(data.data(), data.size(), cb);
}

// ── in-memory replay (also used by tests) ────────────────────────────────────

ReplayStats PcapReplayer::replay_buffer(const uint8_t* buf, size_t len,
                                        const MessageCallback& cb)
{
    ReplayStats stats;

    // ── 1. Global header (24 bytes) ───────────────────────────────────────────
    if (len < 24) return stats;

    const uint32_t magic = le32(buf);
    const bool swapped   = (magic == 0xd4c3b2a1u);
    if (magic != 0xa1b2c3d4u && magic != 0xd4c3b2a1u &&
        magic != 0xa1b23c4du && magic != 0x4d3cb2a1u)
        return stats;  // not a pcap file

    // network type: offset 20 in global header (4 bytes, little-endian)
    const uint32_t network = swapped ? le32(buf + 20) : le32(buf + 20);
    // We only handle Ethernet (linktype 1); skip if not.
    // (still try — some captures use "RAW IP" (101) or "Linux cooked" (113);
    //  for test purposes we enforce Ethernet only)
    if (network != 1) return stats;

    size_t pos = 24;  // start of first packet record

    // ── 2. Packet iteration ───────────────────────────────────────────────────
    while (pos + 16 <= len) {
        // Per-packet header (16 bytes, always little-endian in the file)
        const uint32_t incl_len = le32(buf + pos + 8);
        pos += 16;

        if (pos + incl_len > len) {
            ++stats.packets_skipped;
            break;
        }

        ++stats.packets_read;
        const uint8_t* pkt = buf + pos;
        pos += incl_len;

        if (incl_len < 14u) { ++stats.packets_skipped; continue; }

        // ── 3. Ethernet II (14 bytes) ─────────────────────────────────────────
        // dst(6) src(6) ethertype(2)
        const uint16_t ethertype = be16(pkt + 12);
        if (ethertype != 0x0800u) { ++stats.packets_skipped; continue; }  // IPv4 only

        // ── 4. IPv4 header ────────────────────────────────────────────────────
        // offset 0: version+IHL (nibble each); offset 9: protocol
        if (incl_len < 14u + 20u) { ++stats.packets_skipped; continue; }
        const uint8_t* ip = pkt + 14;
        const uint8_t  ihl = (ip[0] & 0x0fu) * 4u;
        if (ihl < 20u || incl_len < 14u + ihl + 8u) { ++stats.packets_skipped; continue; }
        if (ip[9] != 17u) { ++stats.packets_skipped; continue; }  // UDP = 17

        // ── 5. UDP header (8 bytes) ───────────────────────────────────────────
        const uint8_t* udp        = ip + ihl;
        const uint16_t udp_length = be16(udp + 4);  // includes 8-byte header
        if (udp_length < 8u) { ++stats.packets_skipped; continue; }

        const uint8_t* payload     = udp + 8;
        const size_t   payload_len = static_cast<size_t>(udp_length - 8);

        // Guard: incl_len might be smaller than udp_length indicates
        const size_t avail = incl_len - 14u - static_cast<size_t>(ihl) - 8u;
        const size_t use   = (payload_len < avail) ? payload_len : avail;

        // ── 6. MoldUDP64 + ITCH dispatch ─────────────────────────────────────
        auto [msgs, errs] = process_moldudp64(payload, use, cb);
        stats.itch_messages += msgs;
        stats.parse_errors  += errs;
    }

    return stats;
}

// ── MoldUDP64 framing ─────────────────────────────────────────────────────────
// Layout: session(10) + sequence(8) + message_count(2) = 20 bytes preamble,
//         then message_count × (length(2 BE) + payload)

std::pair<uint64_t, uint64_t>
PcapReplayer::process_moldudp64(const uint8_t* payload, size_t payload_len,
                                 const MessageCallback& cb)
{
    uint64_t msgs = 0, errs = 0;

    // Need at least the 20-byte preamble
    if (payload_len < 20) return {msgs, errs};

    // session:   10 bytes (ASCII, not used here)
    // sequence:   8 bytes (big-endian uint64)
    // count:      2 bytes (big-endian uint16)
    const uint16_t msg_count = be16(payload + 18);
    size_t offset = 20;

    for (uint16_t i = 0; i < msg_count; ++i) {
        if (offset + 2 > payload_len) { ++errs; break; }

        const uint16_t msg_len = be16(payload + offset);
        offset += 2;

        if (offset + msg_len > payload_len) { ++errs; break; }

        auto result = ItchParser::parse(payload + offset, msg_len);
        if (result.has_value()) {
            cb(*result);
            ++msgs;
        } else {
            ++errs;
        }
        offset += msg_len;
    }

    return {msgs, errs};
}

} // namespace itch
} // namespace sigma_edge
