#pragma once
// All application-level event types for the limit-order-book event bus.
//
// Architecture note:
//   Market-data events (TopOfBookUpdate, TradeEvent) travel on the low-latency
//   direct-callback path inside FeedHandler and are intentionally NOT routed
//   through the event bus to avoid pub/sub overhead on the hot path.
//
//   The SigmaEvent bus carries only execution-layer events:
//     SignalEvent  — strategy alpha signal ready for routing
//     OrderEvent   — order lifecycle: SUBMITTED → FILLED / CANCELLED / REJECTED
//     FillEvent    — confirmed execution details (price, qty, side)

#include "../lob/order.hpp"    // Side enum, uint64_t price ticks
#include <cstdint>
#include <variant>

namespace lob {

// ── SignalEvent ───────────────────────────────────────────────────────────────
// Emitted by a signal generator when it wants to express a directional view.
// score ∈ [-1, 1]: negative → bearish, positive → bullish.
struct SignalEvent {
    uint64_t signal_id;       // monotonic counter per signal source
    double   score;           // normalised signal strength [-1, +1]
    Side     direction;       // BID = bullish, ASK = bearish
    uint16_t stock_locate;    // ITCH stock_locate for the symbol
    uint64_t timestamp_ns;    // nanoseconds since midnight
};

// ── SimOrder ──────────────────────────────────────────────────────────────────
// An order submitted to the execution layer (OrderRouter → FillSimulator).
struct SimOrder {
    enum class Type : uint8_t { MARKET = 0, LIMIT = 1 };

    uint64_t id;              // globally unique, monotonically increasing
    uint64_t price;           // limit price (0 for MARKET orders)
    uint32_t quantity;
    Side     side;            // BID = buy, ASK = sell
    Type     type{Type::MARKET};
    uint16_t stock_locate;
    uint64_t timestamp_ns;
};

// ── OrderEvent ────────────────────────────────────────────────────────────────
// Emitted at every state transition of a SimOrder.
struct OrderEvent {
    enum class Status : uint8_t {
        SUBMITTED,
        FILLED,
        PARTIAL_FILL,
        CANCELLED,
        REJECTED
    };

    Status   status;
    SimOrder order;           // full order copy (always populated)
    uint64_t fill_price{0};   // set for FILLED / PARTIAL_FILL
    uint32_t fill_qty{0};     // set for FILLED / PARTIAL_FILL
    uint64_t timestamp_ns;
};

// ── FillEvent ─────────────────────────────────────────────────────────────────
// Emitted after a fill is confirmed; carries execution details independently of
// the full OrderEvent so downstream consumers can subscribe with less overhead.
struct FillEvent {
    uint64_t order_id;
    uint64_t fill_price;      // integer ticks (4 implied decimal places)
    uint32_t fill_qty;
    bool     is_buy;          // true if the filled order was a BID
    uint64_t timestamp_ns;
};

// ── SigmaEvent ────────────────────────────────────────────────────────────────
// The variant type dispatched by SigmaEventBus.
using SigmaEvent = std::variant<SignalEvent, OrderEvent, FillEvent>;

} // namespace lob
