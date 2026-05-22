#include "itch_parser.hpp"
#include <cstring>

namespace sigma_edge {
namespace itch {

// ── Big-endian read helpers ───────────────────────────────────────────────────

static inline uint16_t rd16(const uint8_t* p) {
    return static_cast<uint16_t>(
        (static_cast<uint16_t>(p[0]) << 8) |
         static_cast<uint16_t>(p[1]));
}

static inline uint32_t rd32(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24) |
           (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) <<  8) |
            static_cast<uint32_t>(p[3]);
}

// 8-byte big-endian
static inline uint64_t rd64(const uint8_t* p) {
    return (static_cast<uint64_t>(p[0]) << 56) |
           (static_cast<uint64_t>(p[1]) << 48) |
           (static_cast<uint64_t>(p[2]) << 40) |
           (static_cast<uint64_t>(p[3]) << 32) |
           (static_cast<uint64_t>(p[4]) << 24) |
           (static_cast<uint64_t>(p[5]) << 16) |
           (static_cast<uint64_t>(p[6]) <<  8) |
            static_cast<uint64_t>(p[7]);
}

// 6-byte (48-bit) big-endian timestamp
static inline uint64_t rd48(const uint8_t* p) {
    return (static_cast<uint64_t>(p[0]) << 40) |
           (static_cast<uint64_t>(p[1]) << 32) |
           (static_cast<uint64_t>(p[2]) << 24) |
           (static_cast<uint64_t>(p[3]) << 16) |
           (static_cast<uint64_t>(p[4]) <<  8) |
            static_cast<uint64_t>(p[5]);
}

// Copy 8-byte space-padded stock field and null-terminate (result: char[9])
static inline void rd_stock(char* dst, const uint8_t* src) {
    std::memcpy(dst, src, 8);
    dst[8] = '\0';
}

// ── Common header decoder (bytes 0-10) ───────────────────────────────────────
// offset 0: type (1), offset 1: stock_locate (2), offset 3: tracking (2),
// offset 5: timestamp (6)
static inline MsgHeader decode_header(const uint8_t* buf) {
    MsgHeader h;
    h.message_type     = buf[0];
    h.stock_locate     = rd16(buf + 1);
    h.tracking_number  = rd16(buf + 3);
    h.timestamp_ns     = rd48(buf + 5);
    return h;
}

// ── expected_length ───────────────────────────────────────────────────────────

size_t ItchParser::expected_length(uint8_t t) {
    switch (t) {
        case MSG_SYSTEM_EVENT:           return 12;
        case MSG_STOCK_DIRECTORY:        return 39;
        case MSG_STOCK_TRADING_ACTION:   return 25;
        case MSG_REG_SHO:                return 20;
        case MSG_MARKET_PARTICIPANT_POS: return 26;
        case MSG_ADD_ORDER:              return 36;
        case MSG_ADD_ORDER_MPID:         return 40;
        case MSG_ORDER_EXECUTED:         return 31;
        case MSG_ORDER_EXECUTED_PRICE:   return 36;
        case MSG_ORDER_CANCEL:           return 23;
        case MSG_ORDER_DELETE:           return 19;
        case MSG_ORDER_REPLACE:          return 35;
        case MSG_TRADE:                  return 44;
        case MSG_CROSS_TRADE:            return 40;
        case MSG_BROKEN_TRADE:           return 19;
        case MSG_NOII:                   return 50;
        case MSG_RPII:                   return 20;
        default:                         return 0;
    }
}

// ── parse ─────────────────────────────────────────────────────────────────────

std::optional<ItchMessage> ItchParser::parse(const uint8_t* buf, size_t len) {
    if (len < 1) return std::nullopt;

    const uint8_t t = buf[0];
    const size_t  expected = expected_length(t);

    // Unknown or zero → return UnknownMsg (we still need at least the header)
    if (expected == 0) {
        if (len < 11) return std::nullopt;
        UnknownMsg m;
        m.hdr = decode_header(buf);
        return m;
    }
    if (len < expected) return std::nullopt;

    const MsgHeader hdr = decode_header(buf);

    switch (t) {

    // ── S: System Event ──────────────────────────────────────────────────────
    // offset 11: event_code (1)
    case MSG_SYSTEM_EVENT: {
        SystemEvent m;
        m.hdr        = hdr;
        m.event_code = static_cast<char>(buf[11]);
        return m;
    }

    // ── R: Stock Directory ───────────────────────────────────────────────────
    // offset 11: stock(8), 19: market_cat(1), 20: fin_status(1),
    //        21: lot_size(4), 25: round_lots_only(1), 26: issue_class(1),
    //        27: issue_subtype(2), 29: authenticity(1), 30: short_sale(1),
    //        31: ipo_flag(1), 32: luld_tier(1), 33: etp_flag(1),
    //        34: etp_leverage(4), 38: inverse_indicator(1)
    case MSG_STOCK_DIRECTORY: {
        StockDirectory m;
        m.hdr = hdr;
        rd_stock(m.stock, buf + 11);
        m.market_category      = static_cast<char>(buf[19]);
        m.financial_status     = static_cast<char>(buf[20]);
        m.lot_size             = rd32(buf + 21);
        m.round_lots_only      = (buf[25] == 'Y');
        m.issue_classification = static_cast<char>(buf[26]);
        m.issue_subtype[0]     = static_cast<char>(buf[27]);
        m.issue_subtype[1]     = static_cast<char>(buf[28]);
        m.issue_subtype[2]     = '\0';
        m.authenticity         = static_cast<char>(buf[29]);
        m.short_sale_threshold = static_cast<char>(buf[30]);
        m.ipo_flag             = static_cast<char>(buf[31]);
        m.luld_tier            = static_cast<char>(buf[32]);
        m.etp_flag             = static_cast<char>(buf[33]);
        m.etp_leverage_factor  = rd32(buf + 34);
        m.inverse_indicator    = static_cast<char>(buf[38]);
        return m;
    }

    // ── H: Stock Trading Action ──────────────────────────────────────────────
    // offset 11: stock(8), 19: trading_state(1), 20: reserved(1), 21: reason(4)
    case MSG_STOCK_TRADING_ACTION: {
        StockTradingAction m;
        m.hdr = hdr;
        rd_stock(m.stock, buf + 11);
        m.trading_state = static_cast<char>(buf[19]);
        m.reserved      = static_cast<char>(buf[20]);
        std::memcpy(m.reason, buf + 21, 4);
        m.reason[4] = '\0';
        return m;
    }

    // ── Y: Reg SHO Restriction ───────────────────────────────────────────────
    // offset 11: stock(8), 19: reg_sho_action(1)
    case MSG_REG_SHO: {
        RegSHO m;
        m.hdr            = hdr;
        rd_stock(m.stock, buf + 11);
        m.reg_sho_action = static_cast<char>(buf[19]);
        return m;
    }

    // ── L: Market Participant Position ───────────────────────────────────────
    // offset 11: MPID(4), 15: stock(8), 23: primary_mm(1), 24: mm_mode(1),
    //        25: state(1)
    case MSG_MARKET_PARTICIPANT_POS: {
        MarketParticipantPos m;
        m.hdr = hdr;
        std::memcpy(m.mpid, buf + 11, 4);
        m.mpid[4] = '\0';
        rd_stock(m.stock, buf + 15);
        m.primary_market_maker        = (buf[23] == 'Y');
        m.market_maker_mode           = static_cast<char>(buf[24]);
        m.market_participant_state    = static_cast<char>(buf[25]);
        return m;
    }

    // ── A: Add Order (no MPID) ───────────────────────────────────────────────
    // offset 11: order_ref(8), 19: buy_sell(1), 20: shares(4), 24: stock(8),
    //        32: price(4)
    case MSG_ADD_ORDER: {
        AddOrder m;
        m.hdr                 = hdr;
        m.order_ref_number    = rd64(buf + 11);
        m.buy_sell_indicator  = static_cast<char>(buf[19]);
        m.shares              = rd32(buf + 20);
        rd_stock(m.stock, buf + 24);
        m.price               = rd32(buf + 32);
        return m;
    }

    // ── F: Add Order with MPID ───────────────────────────────────────────────
    // offsets same as A + offset 36: attribution(4)
    case MSG_ADD_ORDER_MPID: {
        AddOrderMPID m;
        m.hdr                 = hdr;
        m.order_ref_number    = rd64(buf + 11);
        m.buy_sell_indicator  = static_cast<char>(buf[19]);
        m.shares              = rd32(buf + 20);
        rd_stock(m.stock, buf + 24);
        m.price               = rd32(buf + 32);
        std::memcpy(m.attribution, buf + 36, 4);
        m.attribution[4]      = '\0';
        return m;
    }

    // ── E: Order Executed ────────────────────────────────────────────────────
    // offset 11: order_ref(8), 19: executed_shares(4), 23: match_number(8)
    case MSG_ORDER_EXECUTED: {
        OrderExecuted m;
        m.hdr                 = hdr;
        m.order_ref_number    = rd64(buf + 11);
        m.executed_shares     = rd32(buf + 19);
        m.match_number        = rd64(buf + 23);
        return m;
    }

    // ── C: Order Executed with Price ─────────────────────────────────────────
    // offset 11: order_ref(8), 19: executed_shares(4), 23: match_number(8),
    //        31: printable(1), 32: execution_price(4)
    case MSG_ORDER_EXECUTED_PRICE: {
        OrderExecutedPrice m;
        m.hdr                 = hdr;
        m.order_ref_number    = rd64(buf + 11);
        m.executed_shares     = rd32(buf + 19);
        m.match_number        = rd64(buf + 23);
        m.printable           = (buf[31] == 'Y');
        m.execution_price     = rd32(buf + 32);
        return m;
    }

    // ── X: Order Cancel ──────────────────────────────────────────────────────
    // offset 11: order_ref(8), 19: cancelled_shares(4)
    case MSG_ORDER_CANCEL: {
        OrderCancel m;
        m.hdr                 = hdr;
        m.order_ref_number    = rd64(buf + 11);
        m.cancelled_shares    = rd32(buf + 19);
        return m;
    }

    // ── D: Order Delete ──────────────────────────────────────────────────────
    // offset 11: order_ref(8)
    case MSG_ORDER_DELETE: {
        OrderDelete m;
        m.hdr              = hdr;
        m.order_ref_number = rd64(buf + 11);
        return m;
    }

    // ── U: Order Replace ─────────────────────────────────────────────────────
    // offset 11: original_ref(8), 19: new_ref(8), 27: shares(4), 31: price(4)
    case MSG_ORDER_REPLACE: {
        OrderReplace m;
        m.hdr                   = hdr;
        m.original_order_ref    = rd64(buf + 11);
        m.new_order_ref         = rd64(buf + 19);
        m.shares                = rd32(buf + 27);
        m.price                 = rd32(buf + 31);
        return m;
    }

    // ── P: Trade (Non-Cross) ─────────────────────────────────────────────────
    // offset 11: order_ref(8), 19: buy_sell(1), 20: shares(4), 24: stock(8),
    //        32: price(4), 36: match_number(8)
    case MSG_TRADE: {
        Trade m;
        m.hdr                 = hdr;
        m.order_ref_number    = rd64(buf + 11);
        m.buy_sell_indicator  = static_cast<char>(buf[19]);
        m.shares              = rd32(buf + 20);
        rd_stock(m.stock, buf + 24);
        m.price               = rd32(buf + 32);
        m.match_number        = rd64(buf + 36);
        return m;
    }

    // ── Q: Cross Trade ───────────────────────────────────────────────────────
    // offset 11: shares(8), 19: stock(8), 27: cross_price(4),
    //        31: match_number(8), 39: cross_type(1)
    case MSG_CROSS_TRADE: {
        CrossTrade m;
        m.hdr         = hdr;
        m.shares      = rd64(buf + 11);
        rd_stock(m.stock, buf + 19);
        m.cross_price = rd32(buf + 27);
        m.match_number= rd64(buf + 31);
        m.cross_type  = static_cast<char>(buf[39]);
        return m;
    }

    // ── B: Broken Trade ──────────────────────────────────────────────────────
    // offset 11: match_number(8)
    case MSG_BROKEN_TRADE: {
        BrokenTrade m;
        m.hdr          = hdr;
        m.match_number = rd64(buf + 11);
        return m;
    }

    // ── I: NOII ──────────────────────────────────────────────────────────────
    // offset 11: paired_shares(8), 19: imbalance_shares(8),
    //        27: direction(1), 28: stock(8), 36: far_price(4),
    //        40: near_price(4), 44: current_ref_price(4),
    //        48: cross_type(1), 49: price_variation(1)
    case MSG_NOII: {
        NOII m;
        m.hdr                      = hdr;
        m.paired_shares            = rd64(buf + 11);
        m.imbalance_shares         = rd64(buf + 19);
        m.imbalance_direction      = static_cast<char>(buf[27]);
        rd_stock(m.stock, buf + 28);
        m.far_price                = rd32(buf + 36);
        m.near_price               = rd32(buf + 40);
        m.current_ref_price        = rd32(buf + 44);
        m.cross_type               = static_cast<char>(buf[48]);
        m.price_variation_indicator= static_cast<char>(buf[49]);
        return m;
    }

    // ── N: RPII ──────────────────────────────────────────────────────────────
    // offset 11: stock(8), 19: interest_flag(1)
    case MSG_RPII: {
        RPII m;
        m.hdr           = hdr;
        rd_stock(m.stock, buf + 11);
        m.interest_flag = static_cast<char>(buf[19]);
        return m;
    }

    default: {
        UnknownMsg m;
        m.hdr = hdr;
        return m;
    }
    }
}

} // namespace itch
} // namespace sigma_edge
