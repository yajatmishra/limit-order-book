// Tests for ItchParser and PcapReplayer.
// Uses the catch.hpp shim (Catch2 v2-compatible) — no network access needed.
#include "../../third_party/catch.hpp"
#include "../../core/feed_handler/itch_parser.hpp"
#include "../../core/feed_handler/pcap_replayer.hpp"
#include <cstring>
#include <vector>
#include <variant>

using namespace sigma_edge::itch;

// ── Buffer-builder helpers ────────────────────────────────────────────────────

// Write a big-endian uint16 at p
static void put16(uint8_t* p, uint16_t v) {
    p[0] = static_cast<uint8_t>(v >> 8);
    p[1] = static_cast<uint8_t>(v);
}
// Write a big-endian uint32 at p
static void put32(uint8_t* p, uint32_t v) {
    p[0] = static_cast<uint8_t>(v >> 24);
    p[1] = static_cast<uint8_t>(v >> 16);
    p[2] = static_cast<uint8_t>(v >>  8);
    p[3] = static_cast<uint8_t>(v);
}
// Write a big-endian uint64 at p
static void put64(uint8_t* p, uint64_t v) {
    for (int i = 7; i >= 0; --i) { p[i] = static_cast<uint8_t>(v & 0xff); v >>= 8; }
}
// Write a 48-bit big-endian timestamp at p (6 bytes)
static void put48(uint8_t* p, uint64_t v) {
    p[0] = static_cast<uint8_t>((v >> 40) & 0xff);
    p[1] = static_cast<uint8_t>((v >> 32) & 0xff);
    p[2] = static_cast<uint8_t>((v >> 24) & 0xff);
    p[3] = static_cast<uint8_t>((v >> 16) & 0xff);
    p[4] = static_cast<uint8_t>((v >>  8) & 0xff);
    p[5] = static_cast<uint8_t>( v        & 0xff);
}
// Fill 8-byte stock field with space-padded ASCII
static void put_stock(uint8_t* p, const char* sym) {
    std::memset(p, ' ', 8);
    size_t n = std::strlen(sym);
    if (n > 8) n = 8;
    std::memcpy(p, sym, n);
}

// Write the common 11-byte header
static void put_header(uint8_t* buf, uint8_t type,
                        uint16_t locate, uint16_t tracking,
                        uint64_t ts_ns)
{
    buf[0] = type;
    put16(buf + 1, locate);
    put16(buf + 3, tracking);
    put48(buf + 5, ts_ns);
}

// ── expected_length ───────────────────────────────────────────────────────────

TEST_CASE("expected_length returns correct sizes", "[itch]") {
    REQUIRE(ItchParser::expected_length('S') == 12);
    REQUIRE(ItchParser::expected_length('R') == 39);
    REQUIRE(ItchParser::expected_length('H') == 25);
    REQUIRE(ItchParser::expected_length('Y') == 20);
    REQUIRE(ItchParser::expected_length('L') == 26);
    REQUIRE(ItchParser::expected_length('A') == 36);
    REQUIRE(ItchParser::expected_length('F') == 40);
    REQUIRE(ItchParser::expected_length('E') == 31);
    REQUIRE(ItchParser::expected_length('C') == 36);
    REQUIRE(ItchParser::expected_length('X') == 23);
    REQUIRE(ItchParser::expected_length('D') == 19);
    REQUIRE(ItchParser::expected_length('U') == 35);
    REQUIRE(ItchParser::expected_length('P') == 44);
    REQUIRE(ItchParser::expected_length('Q') == 40);
    REQUIRE(ItchParser::expected_length('B') == 19);
    REQUIRE(ItchParser::expected_length('I') == 50);
    REQUIRE(ItchParser::expected_length('N') == 20);
    REQUIRE(ItchParser::expected_length('Z') == 0);   // unknown
}

// ── truncated buffer → nullopt ────────────────────────────────────────────────

TEST_CASE("parse returns nullopt on truncated buffer", "[itch]") {
    SECTION("empty buffer") {
        REQUIRE(!ItchParser::parse(nullptr, 0).has_value());
    }
    SECTION("Add Order truncated to 10 bytes") {
        uint8_t buf[10] = {};
        buf[0] = 'A';
        REQUIRE(!ItchParser::parse(buf, 10).has_value());
    }
    SECTION("Order Delete truncated to 18 bytes") {
        uint8_t buf[18] = {};
        buf[0] = 'D';
        REQUIRE(!ItchParser::parse(buf, 18).has_value());
    }
}

// ── S: System Event ───────────────────────────────────────────────────────────

TEST_CASE("parse System Event", "[itch][S]") {
    uint8_t buf[12] = {};
    put_header(buf, 'S', 0, 42, 36000000000000ULL);  // ts = 10 hours in ns
    buf[11] = 'O';

    auto msg = ItchParser::parse(buf, 12);
    REQUIRE(msg.has_value());
    REQUIRE(std::holds_alternative<SystemEvent>(*msg));

    const auto& m = std::get<SystemEvent>(*msg);
    REQUIRE(m.hdr.message_type    == 'S');
    REQUIRE(m.hdr.tracking_number == 42);
    REQUIRE(m.hdr.timestamp_ns    == 36000000000000ULL);
    REQUIRE(m.event_code          == 'O');
}

// ── A: Add Order (no MPID) ────────────────────────────────────────────────────

TEST_CASE("parse Add Order", "[itch][A]") {
    uint8_t buf[36] = {};
    put_header(buf, 'A', 1, 0, 123456789ULL);
    put64(buf + 11, 99887766ULL);   // order_ref
    buf[19] = 'B';                   // buy
    put32(buf + 20, 200u);           // shares
    put_stock(buf + 24, "AAPL");     // stock
    put32(buf + 32, 1502500u);       // price $150.2500

    auto msg = ItchParser::parse(buf, 36);
    REQUIRE(msg.has_value());
    REQUIRE(std::holds_alternative<AddOrder>(*msg));

    const auto& m = std::get<AddOrder>(*msg);
    REQUIRE(m.hdr.message_type   == 'A');
    REQUIRE(m.hdr.stock_locate   == 1);
    REQUIRE(m.hdr.timestamp_ns   == 123456789ULL);
    REQUIRE(m.order_ref_number   == 99887766ULL);
    REQUIRE(m.buy_sell_indicator == 'B');
    REQUIRE(m.shares             == 200u);
    REQUIRE(std::string(m.stock).substr(0, 4) == "AAPL");
    REQUIRE(m.price              == 1502500u);
}

TEST_CASE("parse Add Order sell side", "[itch][A]") {
    uint8_t buf[36] = {};
    put_header(buf, 'A', 2, 0, 0ULL);
    put64(buf + 11, 1ULL);
    buf[19] = 'S';
    put32(buf + 20, 100u);
    put_stock(buf + 24, "MSFT");
    put32(buf + 32, 3750000u);   // $375.0000

    auto msg = ItchParser::parse(buf, 36);
    REQUIRE(std::holds_alternative<AddOrder>(*msg));
    REQUIRE(std::get<AddOrder>(*msg).buy_sell_indicator == 'S');
    REQUIRE(std::get<AddOrder>(*msg).price == 3750000u);
}

// ── F: Add Order with MPID ────────────────────────────────────────────────────

TEST_CASE("parse Add Order MPID", "[itch][F]") {
    uint8_t buf[40] = {};
    put_header(buf, 'F', 1, 0, 0ULL);
    put64(buf + 11, 555ULL);
    buf[19] = 'B';
    put32(buf + 20, 50u);
    put_stock(buf + 24, "TSLA");
    put32(buf + 32, 2000000u);   // $200.0000
    std::memcpy(buf + 36, "GSCO", 4);  // attribution

    auto msg = ItchParser::parse(buf, 40);
    REQUIRE(std::holds_alternative<AddOrderMPID>(*msg));
    const auto& m = std::get<AddOrderMPID>(*msg);
    REQUIRE(m.order_ref_number == 555ULL);
    REQUIRE(std::string(m.attribution) == "GSCO");
}

// ── E: Order Executed ─────────────────────────────────────────────────────────

TEST_CASE("parse Order Executed", "[itch][E]") {
    uint8_t buf[31] = {};
    put_header(buf, 'E', 0, 0, 0ULL);
    put64(buf + 11, 12345ULL);   // order_ref
    put32(buf + 19, 75u);         // executed_shares
    put64(buf + 23, 9999ULL);    // match_number

    auto msg = ItchParser::parse(buf, 31);
    REQUIRE(std::holds_alternative<OrderExecuted>(*msg));
    const auto& m = std::get<OrderExecuted>(*msg);
    REQUIRE(m.order_ref_number == 12345ULL);
    REQUIRE(m.executed_shares  == 75u);
    REQUIRE(m.match_number     == 9999ULL);
}

// ── C: Order Executed with Price ─────────────────────────────────────────────

TEST_CASE("parse Order Executed with Price", "[itch][C]") {
    uint8_t buf[36] = {};
    put_header(buf, 'C', 0, 0, 0ULL);
    put64(buf + 11, 77777ULL);
    put32(buf + 19, 100u);
    put64(buf + 23, 8888ULL);
    buf[31] = 'Y';          // printable
    put32(buf + 32, 1000000u);  // $100.0000

    auto msg = ItchParser::parse(buf, 36);
    REQUIRE(std::holds_alternative<OrderExecutedPrice>(*msg));
    const auto& m = std::get<OrderExecutedPrice>(*msg);
    REQUIRE(m.order_ref_number == 77777ULL);
    REQUIRE(m.executed_shares  == 100u);
    REQUIRE(m.match_number     == 8888ULL);
    REQUIRE(m.printable        == true);
    REQUIRE(m.execution_price  == 1000000u);
}

TEST_CASE("parse Order Executed with Price non-printable", "[itch][C]") {
    uint8_t buf[36] = {};
    put_header(buf, 'C', 0, 0, 0ULL);
    buf[31] = 'N';  // not printable
    auto msg = ItchParser::parse(buf, 36);
    REQUIRE(!std::get<OrderExecutedPrice>(*msg).printable);
}

// ── X: Order Cancel ───────────────────────────────────────────────────────────

TEST_CASE("parse Order Cancel", "[itch][X]") {
    uint8_t buf[23] = {};
    put_header(buf, 'X', 0, 0, 0ULL);
    put64(buf + 11, 33333ULL);
    put32(buf + 19, 400u);

    auto msg = ItchParser::parse(buf, 23);
    REQUIRE(std::holds_alternative<OrderCancel>(*msg));
    const auto& m = std::get<OrderCancel>(*msg);
    REQUIRE(m.order_ref_number == 33333ULL);
    REQUIRE(m.cancelled_shares == 400u);
}

// ── D: Order Delete ───────────────────────────────────────────────────────────

TEST_CASE("parse Order Delete", "[itch][D]") {
    uint8_t buf[19] = {};
    put_header(buf, 'D', 0, 0, 0ULL);
    put64(buf + 11, 11111ULL);

    auto msg = ItchParser::parse(buf, 19);
    REQUIRE(std::holds_alternative<OrderDelete>(*msg));
    REQUIRE(std::get<OrderDelete>(*msg).order_ref_number == 11111ULL);
}

// ── U: Order Replace ─────────────────────────────────────────────────────────

TEST_CASE("parse Order Replace", "[itch][U]") {
    uint8_t buf[35] = {};
    put_header(buf, 'U', 0, 0, 0ULL);
    put64(buf + 11, 1001ULL);    // original_ref
    put64(buf + 19, 2002ULL);    // new_ref
    put32(buf + 27, 300u);       // shares
    put32(buf + 31, 5000000u);   // price $500.0000

    auto msg = ItchParser::parse(buf, 35);
    REQUIRE(std::holds_alternative<OrderReplace>(*msg));
    const auto& m = std::get<OrderReplace>(*msg);
    REQUIRE(m.original_order_ref == 1001ULL);
    REQUIRE(m.new_order_ref      == 2002ULL);
    REQUIRE(m.shares             == 300u);
    REQUIRE(m.price              == 5000000u);
}

// ── P: Trade (Non-Cross) ─────────────────────────────────────────────────────

TEST_CASE("parse Trade", "[itch][P]") {
    uint8_t buf[44] = {};
    put_header(buf, 'P', 3, 0, 0ULL);
    put64(buf + 11, 54321ULL);
    buf[19] = 'B';
    put32(buf + 20, 500u);
    put_stock(buf + 24, "SPY");
    put32(buf + 32, 4500000u);  // $450.0000
    put64(buf + 36, 7777777ULL);

    auto msg = ItchParser::parse(buf, 44);
    REQUIRE(std::holds_alternative<Trade>(*msg));
    const auto& m = std::get<Trade>(*msg);
    REQUIRE(m.order_ref_number   == 54321ULL);
    REQUIRE(m.shares             == 500u);
    REQUIRE(m.price              == 4500000u);
    REQUIRE(m.match_number       == 7777777ULL);
}

// ── Q: Cross Trade ────────────────────────────────────────────────────────────

TEST_CASE("parse Cross Trade", "[itch][Q]") {
    uint8_t buf[40] = {};
    put_header(buf, 'Q', 0, 0, 0ULL);
    put64(buf + 11, 1000000ULL);  // shares
    put_stock(buf + 19, "QQQ");
    put32(buf + 27, 3600000u);    // $360.0000
    put64(buf + 31, 123ULL);      // match_number
    buf[39] = 'O';                 // cross_type = opening

    auto msg = ItchParser::parse(buf, 40);
    REQUIRE(std::holds_alternative<CrossTrade>(*msg));
    const auto& m = std::get<CrossTrade>(*msg);
    REQUIRE(m.shares       == 1000000ULL);
    REQUIRE(m.cross_price  == 3600000u);
    REQUIRE(m.match_number == 123ULL);
    REQUIRE(m.cross_type   == 'O');
}

// ── B: Broken Trade ───────────────────────────────────────────────────────────

TEST_CASE("parse Broken Trade", "[itch][B]") {
    uint8_t buf[19] = {};
    put_header(buf, 'B', 0, 0, 0ULL);
    put64(buf + 11, 42424242ULL);

    auto msg = ItchParser::parse(buf, 19);
    REQUIRE(std::holds_alternative<BrokenTrade>(*msg));
    REQUIRE(std::get<BrokenTrade>(*msg).match_number == 42424242ULL);
}

// ── L: Market Participant Position ───────────────────────────────────────────

TEST_CASE("parse Market Participant Position", "[itch][L]") {
    uint8_t buf[26] = {};
    put_header(buf, 'L', 0, 0, 0ULL);
    std::memcpy(buf + 11, "GSCO", 4);  // MPID
    put_stock(buf + 15, "AAPL");
    buf[23] = 'Y';   // primary market maker
    buf[24] = 'N';   // normal mode
    buf[25] = 'A';   // active

    auto msg = ItchParser::parse(buf, 26);
    REQUIRE(std::holds_alternative<MarketParticipantPos>(*msg));
    const auto& m = std::get<MarketParticipantPos>(*msg);
    REQUIRE(std::string(m.mpid) == "GSCO");
    REQUIRE(m.primary_market_maker == true);
    REQUIRE(m.market_maker_mode    == 'N');
    REQUIRE(m.market_participant_state == 'A');
}

// ── I: NOII ───────────────────────────────────────────────────────────────────

TEST_CASE("parse NOII", "[itch][I]") {
    uint8_t buf[50] = {};
    put_header(buf, 'I', 0, 0, 0ULL);
    put64(buf + 11, 500000ULL);  // paired_shares
    put64(buf + 19, 200000ULL);  // imbalance_shares
    buf[27] = 'B';               // direction: buy
    put_stock(buf + 28, "IWM");
    put32(buf + 36, 1900000u);   // far_price
    put32(buf + 40, 1895000u);   // near_price
    put32(buf + 44, 1897500u);   // current_ref
    buf[48] = 'C';               // cross_type: closing
    buf[49] = 'L';               // price_variation

    auto msg = ItchParser::parse(buf, 50);
    REQUIRE(std::holds_alternative<NOII>(*msg));
    const auto& m = std::get<NOII>(*msg);
    REQUIRE(m.paired_shares    == 500000ULL);
    REQUIRE(m.imbalance_shares == 200000ULL);
    REQUIRE(m.imbalance_direction == 'B');
    REQUIRE(m.far_price        == 1900000u);
    REQUIRE(m.near_price       == 1895000u);
    REQUIRE(m.current_ref_price== 1897500u);
    REQUIRE(m.cross_type       == 'C');
    REQUIRE(m.price_variation_indicator == 'L');
}

// ── N: RPII ───────────────────────────────────────────────────────────────────

TEST_CASE("parse RPII", "[itch][N]") {
    uint8_t buf[20] = {};
    put_header(buf, 'N', 0, 0, 0ULL);
    put_stock(buf + 11, "GS");
    buf[19] = 'B';   // buy side interest

    auto msg = ItchParser::parse(buf, 20);
    REQUIRE(std::holds_alternative<RPII>(*msg));
    const auto& m = std::get<RPII>(*msg);
    REQUIRE(m.interest_flag == 'B');
}

// ── Unknown message type ──────────────────────────────────────────────────────

TEST_CASE("parse unknown message type returns UnknownMsg", "[itch]") {
    uint8_t buf[20] = {};
    put_header(buf, 'Z', 0, 0, 99ULL);

    auto msg = ItchParser::parse(buf, 20);
    REQUIRE(msg.has_value());
    REQUIRE(std::holds_alternative<UnknownMsg>(*msg));
    REQUIRE(std::get<UnknownMsg>(*msg).hdr.message_type == 'Z');
    REQUIRE(std::get<UnknownMsg>(*msg).hdr.timestamp_ns == 99ULL);
}

// ── Big-endian decoding correctness ──────────────────────────────────────────

TEST_CASE("big-endian decoding is correct for all field widths", "[itch][endian]") {
    SECTION("uint16 in stock_locate") {
        uint8_t buf[12] = {};
        buf[0] = 'S';
        buf[1] = 0x12; buf[2] = 0x34;  // stock_locate = 0x1234 = 4660
        put16(buf + 3, 0);
        put48(buf + 5, 0);
        buf[11] = 'S';

        auto msg = ItchParser::parse(buf, 12);
        REQUIRE(std::get<SystemEvent>(*msg).hdr.stock_locate == 0x1234u);
    }

    SECTION("uint64 order_ref in Add Order") {
        uint8_t buf[36] = {};
        put_header(buf, 'A', 0, 0, 0ULL);
        put64(buf + 11, 0x0102030405060708ULL);
        auto msg = ItchParser::parse(buf, 36);
        REQUIRE(std::get<AddOrder>(*msg).order_ref_number == 0x0102030405060708ULL);
    }

    SECTION("48-bit timestamp") {
        uint8_t buf[12] = {};
        buf[0] = 'S';
        put16(buf + 1, 0); put16(buf + 3, 0);
        // Write 0xABCDEF012345 as 6 bytes big-endian
        buf[5]  = 0xAB; buf[6]  = 0xCD; buf[7]  = 0xEF;
        buf[8]  = 0x01; buf[9]  = 0x23; buf[10] = 0x45;
        buf[11] = 'X';
        auto msg = ItchParser::parse(buf, 12);
        REQUIRE(std::get<SystemEvent>(*msg).hdr.timestamp_ns == 0xABCDEF012345ULL);
    }
}

// ── Stock field padding ────────────────────────────────────────────────────────

TEST_CASE("stock field is null-terminated regardless of padding", "[itch]") {
    uint8_t buf[36] = {};
    put_header(buf, 'A', 0, 0, 0ULL);
    // Stock = "GS      " (space-padded, 8 bytes)
    std::memset(buf + 24, ' ', 8);
    buf[24] = 'G'; buf[25] = 'S';

    auto msg = ItchParser::parse(buf, 36);
    const auto& m = std::get<AddOrder>(*msg);
    REQUIRE(m.stock[8] == '\0');  // always null-terminated
    REQUIRE(m.stock[0] == 'G');
    REQUIRE(m.stock[1] == 'S');
}

// ── PcapReplayer: in-memory replay ────────────────────────────────────────────

// Build a minimal valid Ethernet/IP/UDP/MoldUDP64/ITCH pcap in memory
static std::vector<uint8_t> make_synthetic_pcap(
    const std::vector<std::vector<uint8_t>>& itch_msgs)
{
    // ── MoldUDP64 payload ─────────────────────────────────────────────────────
    std::vector<uint8_t> mold(20, 0);  // session(10)+seq(8)+count(2)
    put16(mold.data() + 18, static_cast<uint16_t>(itch_msgs.size()));
    for (const auto& m : itch_msgs) {
        uint8_t len_bytes[2];
        put16(len_bytes, static_cast<uint16_t>(m.size()));
        mold.push_back(len_bytes[0]);
        mold.push_back(len_bytes[1]);
        mold.insert(mold.end(), m.begin(), m.end());
    }

    // ── UDP header (8 bytes) ──────────────────────────────────────────────────
    const uint16_t udp_total = static_cast<uint16_t>(8 + mold.size());
    std::vector<uint8_t> udp_hdr(8, 0);
    put16(udp_hdr.data() + 0, 12300);  // src port
    put16(udp_hdr.data() + 2, 25214);  // dst port (NASDAQ ITCH)
    put16(udp_hdr.data() + 4, udp_total);
    // checksum = 0 (ignored in pcap replay)

    // ── IPv4 header (20 bytes) ────────────────────────────────────────────────
    const uint16_t ip_total = static_cast<uint16_t>(20 + udp_total);
    std::vector<uint8_t> ip_hdr(20, 0);
    ip_hdr[0] = 0x45;  // version=4, IHL=5 (20 bytes)
    ip_hdr[1] = 0;
    put16(ip_hdr.data() + 2, ip_total);
    ip_hdr[8]  = 64;   // TTL
    ip_hdr[9]  = 17;   // UDP

    // ── Ethernet II header (14 bytes) ─────────────────────────────────────────
    std::vector<uint8_t> eth_hdr(14, 0);
    eth_hdr[12] = 0x08; eth_hdr[13] = 0x00;  // EtherType = IPv4

    // ── Assemble packet payload ────────────────────────────────────────────────
    std::vector<uint8_t> pkt_payload;
    pkt_payload.insert(pkt_payload.end(), eth_hdr.begin(), eth_hdr.end());
    pkt_payload.insert(pkt_payload.end(), ip_hdr.begin(), ip_hdr.end());
    pkt_payload.insert(pkt_payload.end(), udp_hdr.begin(), udp_hdr.end());
    pkt_payload.insert(pkt_payload.end(), mold.begin(), mold.end());

    // ── Per-packet record header (16 bytes, LE) ───────────────────────────────
    const uint32_t incl_len = static_cast<uint32_t>(pkt_payload.size());
    std::vector<uint8_t> pkt_hdr(16, 0);
    // ts_sec=1, ts_usec=0
    pkt_hdr[0] = 1;
    // incl_len at offset 8 (LE)
    pkt_hdr[8]  = static_cast<uint8_t>(incl_len);
    pkt_hdr[9]  = static_cast<uint8_t>(incl_len >> 8);
    pkt_hdr[10] = static_cast<uint8_t>(incl_len >> 16);
    pkt_hdr[11] = static_cast<uint8_t>(incl_len >> 24);
    // orig_len at offset 12 (same)
    pkt_hdr[12] = pkt_hdr[8]; pkt_hdr[13] = pkt_hdr[9];
    pkt_hdr[14] = pkt_hdr[10]; pkt_hdr[15] = pkt_hdr[11];

    // ── Global pcap header (24 bytes) ─────────────────────────────────────────
    std::vector<uint8_t> global_hdr(24, 0);
    // magic 0xa1b2c3d4 LE
    global_hdr[0] = 0xd4; global_hdr[1] = 0xc3;
    global_hdr[2] = 0xb2; global_hdr[3] = 0xa1;
    // version 2.4
    global_hdr[4] = 2; global_hdr[5] = 0; global_hdr[6] = 4; global_hdr[7] = 0;
    // snaplen = 65535 LE
    global_hdr[16] = 0xff; global_hdr[17] = 0xff;
    // network = 1 (Ethernet) LE
    global_hdr[20] = 1;

    std::vector<uint8_t> pcap;
    pcap.insert(pcap.end(), global_hdr.begin(), global_hdr.end());
    pcap.insert(pcap.end(), pkt_hdr.begin(), pkt_hdr.end());
    pcap.insert(pcap.end(), pkt_payload.begin(), pkt_payload.end());
    return pcap;
}

// Helper: build a canonical AddOrder buffer
static std::vector<uint8_t> make_add_order(uint64_t ref, char side,
                                            uint32_t shares, const char* sym,
                                            uint32_t price, uint64_t ts = 0)
{
    std::vector<uint8_t> buf(36, 0);
    put_header(buf.data(), 'A', 1, 0, ts);
    put64(buf.data() + 11, ref);
    buf[19] = static_cast<uint8_t>(side);
    put32(buf.data() + 20, shares);
    put_stock(buf.data() + 24, sym);
    put32(buf.data() + 32, price);
    return buf;
}

TEST_CASE("PcapReplayer: replay_buffer delivers one AddOrder", "[pcap]") {
    auto ao = make_add_order(42ULL, 'B', 100, "AAPL", 1502500u, 500ULL);
    auto pcap = make_synthetic_pcap({ao});

    std::vector<ItchMessage> received;
    auto stats = PcapReplayer::replay_buffer(pcap.data(), pcap.size(),
        [&](const ItchMessage& m) { received.push_back(m); });

    REQUIRE(stats.itch_messages == 1);
    REQUIRE(stats.parse_errors  == 0);
    REQUIRE(received.size()     == 1);
    REQUIRE(std::holds_alternative<AddOrder>(received[0]));

    const auto& m = std::get<AddOrder>(received[0]);
    REQUIRE(m.order_ref_number == 42ULL);
    REQUIRE(m.buy_sell_indicator == 'B');
    REQUIRE(m.shares == 100u);
    REQUIRE(m.price  == 1502500u);
    REQUIRE(m.hdr.timestamp_ns == 500ULL);
}

TEST_CASE("PcapReplayer: replay_buffer delivers multiple messages in order", "[pcap]") {
    std::vector<std::vector<uint8_t>> msgs;
    for (uint64_t i = 1; i <= 5; ++i)
        msgs.push_back(make_add_order(i, 'B', 100u * static_cast<uint32_t>(i),
                                      "SPY", 4000000u + static_cast<uint32_t>(i)));

    auto pcap = make_synthetic_pcap(msgs);
    std::vector<uint64_t> refs;
    PcapReplayer::replay_buffer(pcap.data(), pcap.size(),
        [&](const ItchMessage& m) {
            refs.push_back(std::get<AddOrder>(m).order_ref_number);
        });

    REQUIRE(refs.size() == 5);
    for (uint64_t i = 0; i < 5; ++i) REQUIRE(refs[i] == i + 1);
}

TEST_CASE("PcapReplayer: empty pcap buffer returns zero stats", "[pcap]") {
    auto stats = PcapReplayer::replay_buffer(nullptr, 0,
        [](const ItchMessage&) {});
    REQUIRE(stats.packets_read   == 0);
    REQUIRE(stats.itch_messages  == 0);
}

TEST_CASE("PcapReplayer: non-pcap magic is silently ignored", "[pcap]") {
    uint8_t junk[24] = {};  // all zeros → bad magic
    auto stats = PcapReplayer::replay_buffer(junk, sizeof(junk),
        [](const ItchMessage&) {});
    REQUIRE(stats.itch_messages == 0);
}

TEST_CASE("PcapReplayer: mixed message types in one packet", "[pcap]") {
    // AddOrder + OrderDelete
    auto ao = make_add_order(100ULL, 'S', 250, "NVDA", 5000000u);
    std::vector<uint8_t> del(19, 0);
    put_header(del.data(), 'D', 0, 0, 0ULL);
    put64(del.data() + 11, 100ULL);

    auto pcap = make_synthetic_pcap({ao, del});
    std::vector<uint8_t> types;
    PcapReplayer::replay_buffer(pcap.data(), pcap.size(),
        [&](const ItchMessage& m) {
            std::visit([&](const auto& v){ types.push_back(v.hdr.message_type); }, m);
        });

    REQUIRE(types.size() == 2);
    REQUIRE(types[0] == 'A');
    REQUIRE(types[1] == 'D');
}

TEST_CASE("PcapReplayer: stats track packet and message counts", "[pcap]") {
    auto ao = make_add_order(1ULL, 'B', 10, "GS", 3000000u);
    auto pcap = make_synthetic_pcap({ao});

    auto stats = PcapReplayer::replay_buffer(pcap.data(), pcap.size(),
        [](const ItchMessage&) {});

    REQUIRE(stats.packets_read  >= 1);
    REQUIRE(stats.itch_messages == 1);
}

// ── LOB integration: feed messages into LimitOrderBook via variant visitor ────

#include "../../core/lob/limit_order_book.hpp"

TEST_CASE("ITCH AddOrder/OrderDelete round-trip into LOB", "[itch][lob]") {
    using namespace sigma_edge;

    LimitOrderBook lob;

    // Build 3 ITCH AddOrder messages and feed them into the LOB
    std::vector<std::vector<uint8_t>> msgs;
    msgs.push_back(make_add_order(1ULL, 'B', 100, "AAPL", 1500000u));  // bid $150.00
    msgs.push_back(make_add_order(2ULL, 'B', 200, "AAPL", 1490000u));  // bid $149.00
    msgs.push_back(make_add_order(3ULL, 'S', 150, "AAPL", 1510000u));  // ask $151.00

    auto pcap = make_synthetic_pcap(msgs);

    PcapReplayer::replay_buffer(pcap.data(), pcap.size(),
        [&](const ItchMessage& msg) {
            std::visit([&](const auto& m) {
                using T = std::decay_t<decltype(m)>;
                if constexpr (std::is_same_v<T, AddOrder>) {
                    Order o;
                    o.id        = m.order_ref_number;
                    o.price     = m.price;
                    o.quantity  = m.shares;
                    o.side      = (m.buy_sell_indicator == 'B') ? Side::BID : Side::ASK;
                    o.timestamp = m.hdr.timestamp_ns;
                    lob.add_order(o);
                }
            }, msg);
        });

    REQUIRE(lob.total_orders()  == 3);
    REQUIRE(lob.best_bid()      == 1500000u);
    REQUIRE(lob.best_ask()      == 1510000u);
    REQUIRE(lob.spread()        == 10000u);  // $1.0000

    // Delete order 1 (best bid)
    REQUIRE(lob.cancel_order(1));
    REQUIRE(lob.best_bid() == 1490000u);
}
