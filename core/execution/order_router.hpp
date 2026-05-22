#pragma once
// OrderRouter: strategy-injected order submission with risk limits.
//
// Architecture note — why market data bypasses the bus:
//   In HFT, every nanosecond on the critical path matters.  The pub/sub
//   overhead is acceptable for execution events (rare) but not for every
//   top-of-book tick.  FeedHandler calls OrderRouter::on_tob directly.
//
// Strategy contract:
//   (TopOfBookUpdate, current_position, next_order_id) → optional<SimOrder>
//   Return nullopt to pass; SimOrder to trade (risk check applied after).

#include "../event_bus/event_bus.hpp"
#include "../feed_handler/feed_handler.hpp"  // TopOfBookUpdate, TradeEvent
#include <optional>
#include <functional>
#include <cstdint>

namespace lob {

// ── OrderRouterConfig: lifted outside the class so it can be used as a
//    default argument (GCC requires the type to be complete at that point).
struct OrderRouterConfig {
    int32_t  max_long_position{1'000};   // max long exposure in shares
    int32_t  max_short_position{-1'000}; // max short exposure (negative)
    uint32_t default_qty{100};           // shares per signal order
    uint32_t max_spread_ticks{100};      // skip signal if spread > this
};

// ── OrderRouter ───────────────────────────────────────────────────────────────
class OrderRouter {
public:
    using Config   = OrderRouterConfig;
    using Strategy = std::function<
        std::optional<SimOrder>(const TopOfBookUpdate&,
                                int32_t  position,
                                uint64_t next_id)>;

    // bus  : the execution event bus (subscribes to FillEvent here)
    // cfg  : risk/sizing parameters
    // strat: trading strategy (set later via set_strategy if omitted)
    explicit OrderRouter(SigmaEventBus& bus,
                         Config   cfg   = Config{},
                         Strategy strat = Strategy{});

    // ── Market-data entry point (called directly by FeedHandler bridge) ───────
    void on_tob(const TopOfBookUpdate& tob);

    // ── Direct order submission (tests / manual calls) ────────────────────────
    // Risk-checked; publishes OrderEvent{SUBMITTED} on acceptance.
    bool submit(SimOrder order);

    // ── Strategy ──────────────────────────────────────────────────────────────
    void set_strategy(Strategy s) { strategy_ = std::move(s); }

    // Built-in: enter long when flat + narrow spread; unwind when long.
    static Strategy make_mean_reversion_strategy(const Config& cfg);

    // ── Statistics ────────────────────────────────────────────────────────────
    int32_t  position()         const noexcept { return position_; }
    uint64_t orders_submitted() const noexcept { return submitted_; }
    uint64_t orders_rejected()  const noexcept { return rejected_;  }
    uint64_t fills_received()   const noexcept { return fills_;     }

    void reset_stats() noexcept;

private:
    SigmaEventBus& bus_;
    Config         cfg_;
    Strategy       strategy_;

    int32_t  position_{0};
    uint64_t order_seq_{0};
    uint64_t submitted_{0};
    uint64_t rejected_{0};
    uint64_t fills_{0};

    void on_fill(const FillEvent& fill);
    bool passes_risk(const SimOrder& order) const noexcept;
};

} // namespace lob
