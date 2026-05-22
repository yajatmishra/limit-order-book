#pragma once
// PCAP file replayer for NASDAQ TotalView-ITCH 5.0 feeds.
//
// Wire encapsulation expected:
//   Ethernet II (14 bytes) → IP (variable, IHL*4) → UDP (8 bytes)
//   → MoldUDP64 (session 10 + seq 8 + count 2 = 20 bytes)
//   → length-prefixed ITCH messages (2-byte big-endian length, then payload)
//
// The replayer iterates every packet in the pcap, strips the network headers,
// and for each ITCH message in the MoldUDP64 block invokes the provided
// MessageCallback.  Unknown or malformed packets are silently skipped.

#include "itch_parser.hpp"
#include <cstdint>
#include <cstddef>
#include <string>
#include <functional>

namespace sigma_edge {
namespace itch {

// ── pcap on-disk structures ───────────────────────────────────────────────────
// We decode these manually so we don't depend on libpcap.

// pcap global file header (24 bytes)
struct PcapGlobalHeader {
    uint32_t magic_number;   // 0xa1b2c3d4 (native) or 0xd4c3b2a1 (swap)
    uint16_t version_major;
    uint16_t version_minor;
    int32_t  thiszone;       // GMT offset (usually 0)
    uint32_t sigfigs;
    uint32_t snaplen;
    uint32_t network;        // link-layer type; 1 = Ethernet
};

// pcap per-packet record header (16 bytes)
struct PcapPacketHeader {
    uint32_t ts_sec;
    uint32_t ts_usec;        // may be nanoseconds if magic = 0xa1b23c4d
    uint32_t incl_len;       // bytes actually captured
    uint32_t orig_len;       // original wire length
};

// ── Replay statistics ─────────────────────────────────────────────────────────
struct ReplayStats {
    uint64_t packets_read    = 0;
    uint64_t packets_skipped = 0;  // bad headers / too short / wrong link type
    uint64_t itch_messages   = 0;
    uint64_t parse_errors    = 0;
};

// ── PcapReplayer ──────────────────────────────────────────────────────────────
class PcapReplayer {
public:
    // Open a pcap file.  Throws std::runtime_error on I/O or format errors.
    explicit PcapReplayer(const std::string& path);

    // Replay all packets; invoke cb for every successfully parsed ITCH message.
    // Returns aggregate statistics.
    ReplayStats replay(const MessageCallback& cb);

    // Replay from an in-memory buffer (useful for unit tests without disk I/O).
    // buf must remain valid for the duration of the call.
    static ReplayStats replay_buffer(const uint8_t* buf, size_t len,
                                     const MessageCallback& cb);

private:
    std::string path_;

    // Process a single UDP payload (past the UDP header).
    // Parses MoldUDP64 framing and dispatches each ITCH message to cb.
    // Returns {messages dispatched, parse errors}.
    static std::pair<uint64_t, uint64_t>
    process_moldudp64(const uint8_t* payload, size_t payload_len,
                      const MessageCallback& cb);
};

} // namespace itch
} // namespace sigma_edge
