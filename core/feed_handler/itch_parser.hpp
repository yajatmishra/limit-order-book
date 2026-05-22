#pragma once
// NASDAQ TotalView-ITCH 5.0 message structs and parser.
// All integer fields are big-endian on the wire; structs here hold host-endian values.
// Price fields use 4 implied decimal places (e.g. $150.2500 → 1502500).
// Timestamps are nanoseconds since midnight (6-byte / 48-bit on wire).

#include <cstdint>
#include <cstddef>
#include <optional>
#include <variant>
#include <string>
#include <functional>
#include <span>

namespace lob {
namespace itch {

// ── Message type bytes (ITCH 5.0 §4) ─────────────────────────────────────────
inline constexpr uint8_t MSG_SYSTEM_EVENT            = 'S';
inline constexpr uint8_t MSG_STOCK_DIRECTORY         = 'R';
inline constexpr uint8_t MSG_STOCK_TRADING_ACTION    = 'H';
inline constexpr uint8_t MSG_REG_SHO                 = 'Y';
inline constexpr uint8_t MSG_MARKET_PARTICIPANT_POS  = 'L';
inline constexpr uint8_t MSG_MWCB_DECLINE            = 'V';
inline constexpr uint8_t MSG_MWCB_STATUS             = 'W';
inline constexpr uint8_t MSG_IPO_QUOTING_PERIOD      = 'K';
inline constexpr uint8_t MSG_LULD_AUCTION_COLLAR     = 'J';
inline constexpr uint8_t MSG_ADD_ORDER               = 'A';
inline constexpr uint8_t MSG_ADD_ORDER_MPID          = 'F';
inline constexpr uint8_t MSG_ORDER_EXECUTED          = 'E';
inline constexpr uint8_t MSG_ORDER_EXECUTED_PRICE    = 'C';
inline constexpr uint8_t MSG_ORDER_CANCEL            = 'X';
inline constexpr uint8_t MSG_ORDER_DELETE            = 'D';
inline constexpr uint8_t MSG_ORDER_REPLACE           = 'U';
inline constexpr uint8_t MSG_TRADE                   = 'P';
inline constexpr uint8_t MSG_CROSS_TRADE             = 'Q';
inline constexpr uint8_t MSG_BROKEN_TRADE            = 'B';
inline constexpr uint8_t MSG_NOII                    = 'I';
inline constexpr uint8_t MSG_RPII                    = 'N';

// ── Common header (first 11 bytes of every ITCH message) ─────────────────────
// offset 0  : message_type (1)
// offset 1  : stock_locate (2)
// offset 3  : tracking_number (2)
// offset 5  : timestamp (6)  → nanoseconds since midnight

struct MsgHeader {
    uint8_t  message_type;
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint64_t timestamp_ns;   // nanoseconds since midnight (decoded from 6 bytes)
};

// ── Individual message structs ────────────────────────────────────────────────

// S — System Event  (12 bytes total)
struct SystemEvent {
    MsgHeader hdr;
    char      event_code;    // O=start of messages, S=start of system hours, …
};

// R — Stock Directory  (39 bytes total)
struct StockDirectory {
    MsgHeader hdr;
    char      stock[9];      // 8-byte padded + null terminator we add
    char      market_category;
    char      financial_status;
    uint32_t  lot_size;
    bool      round_lots_only;
    char      issue_classification;
    char      issue_subtype[3];
    char      authenticity;
    char      short_sale_threshold;
    char      ipo_flag;
    char      luld_tier;
    char      etp_flag;
    uint32_t  etp_leverage_factor;
    char      inverse_indicator;
};

// H — Stock Trading Action  (25 bytes total)
struct StockTradingAction {
    MsgHeader hdr;
    char      stock[9];
    char      trading_state;
    char      reserved;
    char      reason[5];
};

// Y — Reg SHO Restriction  (20 bytes total)
struct RegSHO {
    MsgHeader hdr;
    char      stock[9];
    char      reg_sho_action;
};

// L — Market Participant Position  (26 bytes total)
struct MarketParticipantPos {
    MsgHeader hdr;
    char      mpid[5];       // 4-byte + null
    char      stock[9];      // 8-byte + null
    bool      primary_market_maker;
    char      market_maker_mode;
    char      market_participant_state;
};

// A — Add Order (no MPID)  (36 bytes total)
struct AddOrder {
    MsgHeader hdr;
    uint64_t  order_ref_number;
    char      buy_sell_indicator;  // 'B' or 'S'
    uint32_t  shares;
    char      stock[9];            // 8-byte padded + null
    uint32_t  price;               // 4 implied decimal places
};

// F — Add Order with MPID  (40 bytes total)
struct AddOrderMPID {
    MsgHeader hdr;
    uint64_t  order_ref_number;
    char      buy_sell_indicator;
    uint32_t  shares;
    char      stock[9];
    uint32_t  price;
    char      attribution[5];      // 4-byte + null
};

// E — Order Executed  (31 bytes total)
struct OrderExecuted {
    MsgHeader hdr;
    uint64_t  order_ref_number;
    uint32_t  executed_shares;
    uint64_t  match_number;
};

// C — Order Executed with Price  (36 bytes total)
struct OrderExecutedPrice {
    MsgHeader hdr;
    uint64_t  order_ref_number;
    uint32_t  executed_shares;
    uint64_t  match_number;
    bool      printable;
    uint32_t  execution_price;
};

// X — Order Cancel  (23 bytes total)
struct OrderCancel {
    MsgHeader hdr;
    uint64_t  order_ref_number;
    uint32_t  cancelled_shares;
};

// D — Order Delete  (19 bytes total)
struct OrderDelete {
    MsgHeader hdr;
    uint64_t  order_ref_number;
};

// U — Order Replace  (35 bytes total)
struct OrderReplace {
    MsgHeader hdr;
    uint64_t  original_order_ref;
    uint64_t  new_order_ref;
    uint32_t  shares;
    uint32_t  price;
};

// P — Trade (Non-Cross)  (44 bytes total)
struct Trade {
    MsgHeader hdr;
    uint64_t  order_ref_number;
    char      buy_sell_indicator;
    uint32_t  shares;
    char      stock[9];
    uint32_t  price;
    uint64_t  match_number;
};

// Q — Cross Trade  (40 bytes total)
struct CrossTrade {
    MsgHeader hdr;
    uint64_t  shares;
    char      stock[9];
    uint32_t  cross_price;
    uint64_t  match_number;
    char      cross_type;
};

// B — Broken Trade  (19 bytes total)
struct BrokenTrade {
    MsgHeader hdr;
    uint64_t  match_number;
};

// I — Net Order Imbalance Indicator  (50 bytes total)
struct NOII {
    MsgHeader hdr;
    uint64_t  paired_shares;
    uint64_t  imbalance_shares;
    char      imbalance_direction;
    char      stock[9];
    uint32_t  far_price;
    uint32_t  near_price;
    uint32_t  current_ref_price;
    char      cross_type;
    char      price_variation_indicator;
};

// N — Retail Price Improvement Indicator  (20 bytes total)
struct RPII {
    MsgHeader hdr;
    char      stock[9];
    char      interest_flag;
};

// ── Unknown / unsupported message passthrough ─────────────────────────────────
struct UnknownMsg {
    MsgHeader hdr;
};

// ── Variant covering all supported message types ──────────────────────────────
using ItchMessage = std::variant<
    SystemEvent,
    StockDirectory,
    StockTradingAction,
    RegSHO,
    MarketParticipantPos,
    AddOrder,
    AddOrderMPID,
    OrderExecuted,
    OrderExecutedPrice,
    OrderCancel,
    OrderDelete,
    OrderReplace,
    Trade,
    CrossTrade,
    BrokenTrade,
    NOII,
    RPII,
    UnknownMsg
>;

// ── ItchParser ────────────────────────────────────────────────────────────────
// Stateless: parse() takes a raw buffer pointer + length and returns the
// decoded message.  The caller is responsible for framing (MoldUDP64 or pcap).
class ItchParser {
public:
    // Parse one ITCH message from raw bytes.
    // buf[0] must be the message_type byte; len must be >= 1.
    // Returns std::nullopt if the buffer is shorter than expected_length(buf[0]).
    static std::optional<ItchMessage> parse(const uint8_t* buf, size_t len);

    // Returns the expected wire length (including type byte) for a given
    // message type, or 0 if not recognised.
    static size_t expected_length(uint8_t msg_type);
};

// ── Callback type for the feed handler and replayer ──────────────────────────
using MessageCallback = std::function<void(const ItchMessage&)>;

} // namespace itch
} // namespace lob
