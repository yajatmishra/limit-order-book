#pragma once
// FillSimulator: LOB-aware execution simulation with slippage and latency.
//
// Model:
//   • Subscribes to OrderEvent{SUBMITTED} on the SigmaEventBus.
//   • On receipt, computes fill price from the live LOB:
//       BUY  MARKET → best_ask + slippage_ticks
//       SELL MARKET → best_bid − slippage_ticks
//       LIMIT BUY   → fills if best_ask <= order.price
//       LIMIT SELL  → fills if best_bid >= order.price
//   • Enqueues a PendingFill due at (order.timestamp_ns + fill_latency_ns).
//   • tick(current_ns) fires fills whose due time has elapsed, publishing
//     OrderEvent{FILLED} then FillEvent on the bus.
//   • No opposing side → OrderEvent{REJECTED} published immediately.

#include "../event_bus/event_bus.hpp"
#include "../lob/limit_order_book.hpp"
#include <vector>
#include <algorithm>
#include <cstdint>

namespace lob {

// ── FillSimulatorConfig: lifted outside the class for the same reason as
//    OrderRouterConfig — GCC requires the nested type to be complete before
//    it can appear as a default argument inside the enclosing class.
struct FillSimulatorConfig {
    uint32_t slippage_ticks{1};       // market-impact ticks added/subtracted
    uint64_t fill_latency_ns{1'000};  // simulated exchange round-trip in ns
};

// ── FillSimulator ─────────────────────────────────────────────────────────────
class FillSimulator {
public:
    using Config = FillSimulatorConfig;

    // lob : live LOB consulted at order-arrival time (caller owns it)
    // bus : publishes FILLED / REJECTED OrderEvents and FillEvents
    FillSimulator(const LimitOrderBook& lob,
                  SigmaEventBus&        bus,
                  Config                cfg = Config{});

    // Advance simulation clock; fires all fills due by current_ns.
    // Returns number of fills fired this tick.
    uint32_t tick(uint64_t current_ns);

    // Process a SimOrder directly (bypasses bus; for unit tests).
    void submit_direct(const SimOrder& order, uint64_t timestamp_ns);

    // ── Statistics ────────────────────────────────────────────────────────────
    uint64_t total_fills()       const noexcept { return fills_;       }
    uint64_t total_filled_qty()  const noexcept { return filled_qty_;  }
    uint64_t total_rejections()  const noexcept { return rejections_;  }
    size_t   pending_count()     const noexcept { return pending_.size(); }

private:
    struct PendingFill {
        SimOrder order;
        uint64_t due_ns;        // fire when current_ns >= due_ns
        uint64_t fill_price;    // captured at submission time from LOB
    };

    const LimitOrderBook& lob_;
    SigmaEventBus&        bus_;
    Config                cfg_;

    std::vector<PendingFill> pending_;
    uint64_t fills_{0};
    uint64_t filled_qty_{0};
    uint64_t rejections_{0};

    uint64_t compute_fill_price(const SimOrder& order) const noexcept;
    void fire_fill(const PendingFill& pf, uint64_t current_ns);
    void reject(const SimOrder& order, uint64_t timestamp_ns);
    void on_order_submitted(const OrderEvent& ev);
};

} // namespace lob
