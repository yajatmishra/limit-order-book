#include "fill_simulator.hpp"

namespace lob {

FillSimulator::FillSimulator(const LimitOrderBook& lob,
                             SigmaEventBus& bus,
                             Config cfg)
    : lob_(lob), bus_(bus), cfg_(cfg)
{
    bus_.subscribe<OrderEvent>([this](const OrderEvent& ev) {
        on_order_submitted(ev);
    });
}

// ── on_order_submitted ────────────────────────────────────────────────────────

void FillSimulator::on_order_submitted(const OrderEvent& ev) {
    if (ev.status != OrderEvent::Status::SUBMITTED) return;
    submit_direct(ev.order, ev.timestamp_ns);
}

void FillSimulator::submit_direct(const SimOrder& order, uint64_t timestamp_ns) {
    const uint64_t fill_price = compute_fill_price(order);
    if (fill_price == 0) {
        reject(order, timestamp_ns);
        return;
    }
    pending_.push_back(PendingFill{order, timestamp_ns + cfg_.fill_latency_ns, fill_price});
}

// ── tick ──────────────────────────────────────────────────────────────────────

uint32_t FillSimulator::tick(uint64_t current_ns) {
    uint32_t fired = 0;
    // Scan forward; stable_partition preserves arrival order of remaining fills.
    auto new_end = std::stable_partition(
        pending_.begin(), pending_.end(),
        [current_ns](const PendingFill& pf) { return pf.due_ns > current_ns; }
    );

    for (auto it = new_end; it != pending_.end(); ++it) {
        fire_fill(*it, current_ns);
        ++fired;
    }
    pending_.erase(new_end, pending_.end());
    return fired;
}

// ── fire_fill ─────────────────────────────────────────────────────────────────

void FillSimulator::fire_fill(const PendingFill& pf, uint64_t current_ns) {
    ++fills_;
    filled_qty_ += pf.order.quantity;

    // OrderEvent{FILLED}
    OrderEvent oev;
    oev.status       = OrderEvent::Status::FILLED;
    oev.order        = pf.order;
    oev.fill_price   = pf.fill_price;
    oev.fill_qty     = pf.order.quantity;
    oev.timestamp_ns = current_ns;
    bus_.publish(oev);

    // FillEvent (lighter-weight; downstream consumers can subscribe to just this)
    FillEvent fev;
    fev.order_id     = pf.order.id;
    fev.fill_price   = pf.fill_price;
    fev.fill_qty     = pf.order.quantity;
    fev.is_buy       = (pf.order.side == Side::BID);
    fev.timestamp_ns = current_ns;
    bus_.publish(fev);
}

// ── reject ────────────────────────────────────────────────────────────────────

void FillSimulator::reject(const SimOrder& order, uint64_t timestamp_ns) {
    ++rejections_;
    OrderEvent ev;
    ev.status       = OrderEvent::Status::REJECTED;
    ev.order        = order;
    ev.timestamp_ns = timestamp_ns;
    bus_.publish(ev);
}

// ── compute_fill_price ────────────────────────────────────────────────────────

uint64_t FillSimulator::compute_fill_price(const SimOrder& order) const noexcept {
    if (order.type == SimOrder::Type::MARKET) {
        if (order.side == Side::BID) {
            // Buy: fill at best ask + slippage
            auto ask = lob_.best_ask();
            if (!ask) return 0;
            return *ask + cfg_.slippage_ticks;
        } else {
            // Sell: fill at best bid - slippage (clamp to 1 to avoid underflow)
            auto bid = lob_.best_bid();
            if (!bid) return 0;
            return (*bid > cfg_.slippage_ticks)
                       ? (*bid - cfg_.slippage_ticks)
                       : 1u;
        }
    } else {
        // LIMIT order
        if (order.side == Side::BID) {
            // Buy limit: fills if best_ask <= order.price
            auto ask = lob_.best_ask();
            if (!ask || *ask > order.price) return 0;
            return *ask + cfg_.slippage_ticks;
        } else {
            // Sell limit: fills if best_bid >= order.price
            auto bid = lob_.best_bid();
            if (!bid || *bid < order.price) return 0;
            return (*bid > cfg_.slippage_ticks)
                       ? (*bid - cfg_.slippage_ticks)
                       : 1u;
        }
    }
}

} // namespace lob
