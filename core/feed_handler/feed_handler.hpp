#pragma once
// FeedHandler: wires the ITCH 5.0 parser to the LimitOrderBook.
//
// Responsibilities:
//   - Accept parsed ItchMessage variants and apply each to the LOB.
//   - Emit MarketEvent (TopOfBookUpdate / TradeEvent) on observable changes.
//   - Track per-session statistics (messages processed, TOB updates, errors).
//
// ITCH → LOB mapping (per TotalView-ITCH 5.0 §4):
//   A / F  Add Order            → lob.add_order()
//   E      Order Executed        → lob.execute_order(ref, executed_shares)
//   C      Order Executed/Price  → lob.execute_order(ref, executed_shares)
//   X      Order Cancel          → lob.execute_order(ref, cancelled_shares)
//          (X is a *partial* cancel — semantically identical to a partial fill)
//   D      Order Delete          → lob.cancel_order(ref)
//   U      Order Replace         → lob.replace_order(old_ref, new_order)
//   All others are informational and do not mutate the LOB.

#include "itch_parser.hpp"
#include "../lob/limit_order_book.hpp"
#include <cstdint>
#include <functional>
#include <variant>

namespace lob {

// ── MarketEvent types emitted by FeedHandler ─────────────────────────────────

// Fires whenever the best bid or best ask changes (including when either
// disappears).  Prices are 0 when that side of the book is empty.
struct TopOfBookUpdate {
    uint16_t stock_locate;
    uint64_t bid_price;    // 0 = no resting bid
    uint64_t ask_price;    // 0 = no resting ask
    uint64_t timestamp_ns;
};

// Fires on every Order Executed (E) or Order Executed with Price (C) message.
struct TradeEvent {
    uint64_t order_ref;
    uint32_t price;        // execution price (from C msg if available; else 0)
    uint32_t shares;
    bool     is_buy;       // true if the resting order was a bid
    uint64_t timestamp_ns;
};

using MarketEvent = std::variant<TopOfBookUpdate, TradeEvent>;

// ── FeedHandler ───────────────────────────────────────────────────────────────

class FeedHandler {
public:
    using EventCallback = std::function<void(const MarketEvent&)>;

    // lob       : the book to maintain (caller owns it; must outlive FeedHandler)
    // event_cb  : optional callback fired on TOB changes and trade events
    explicit FeedHandler(LimitOrderBook& lob, EventCallback event_cb = {}) noexcept;

    // Process one decoded ITCH message.  Thread-safety: not thread-safe —
    // call from a single feed-processing thread.
    void process(const itch::ItchMessage& msg);

    // ── Statistics ────────────────────────────────────────────────────────────
    uint64_t messages_processed() const noexcept { return msg_count_; }
    uint64_t tob_updates()        const noexcept { return tob_count_; }
    uint64_t trade_events()       const noexcept { return trade_count_; }
    uint64_t lob_errors()         const noexcept { return error_count_; }

    // Reset counters (does NOT clear the LOB).
    void reset_stats() noexcept;

private:
    LimitOrderBook& lob_;
    EventCallback   event_cb_;

    uint64_t msg_count_{0};
    uint64_t tob_count_{0};
    uint64_t trade_count_{0};
    uint64_t error_count_{0};

    // Last observed TOB — used to detect changes without re-querying after
    // every message.  Stored as raw uint64 so 0 = absent.
    uint64_t last_bid_{0};
    uint64_t last_ask_{0};
    uint16_t last_locate_{0};

    // Re-check TOB and fire TopOfBookUpdate if it changed.
    void check_tob(uint16_t stock_locate, uint64_t ts_ns);

    // Fire an event via the callback (noop if no callback registered).
    void emit(const MarketEvent& ev);
};

} // namespace lob
