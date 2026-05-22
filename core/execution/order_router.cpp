#include "order_router.hpp"

namespace lob {

OrderRouter::OrderRouter(SigmaEventBus& bus, Config cfg, Strategy strat)
    : bus_(bus), cfg_(cfg), strategy_(std::move(strat))
{
    if (!strategy_) strategy_ = make_mean_reversion_strategy(cfg_);

    // Subscribe to FillEvent on the bus to keep position current.
    bus_.subscribe<FillEvent>([this](const FillEvent& f) { on_fill(f); });
}

// ── market-data entry point ───────────────────────────────────────────────────

void OrderRouter::on_tob(const TopOfBookUpdate& tob) {
    if (!strategy_) return;

    const uint64_t next_id = order_seq_ + 1;
    auto maybe_order = strategy_(tob, position_, next_id);
    if (!maybe_order) return;

    submit(std::move(*maybe_order));
}

// ── submit ────────────────────────────────────────────────────────────────────

bool OrderRouter::submit(SimOrder order) {
    order.id = ++order_seq_;   // assign canonical id here

    if (!passes_risk(order)) {
        ++rejected_;
        OrderEvent ev;
        ev.status       = OrderEvent::Status::REJECTED;
        ev.order        = order;
        ev.timestamp_ns = order.timestamp_ns;
        bus_.publish(ev);
        return false;
    }

    ++submitted_;
    OrderEvent ev;
    ev.status       = OrderEvent::Status::SUBMITTED;
    ev.order        = order;
    ev.timestamp_ns = order.timestamp_ns;
    bus_.publish(ev);
    return true;
}

// ── on_fill ───────────────────────────────────────────────────────────────────

void OrderRouter::on_fill(const FillEvent& fill) {
    ++fills_;
    const int32_t delta = static_cast<int32_t>(fill.fill_qty);
    position_ += fill.is_buy ? delta : -delta;
}

// ── risk guard ────────────────────────────────────────────────────────────────

bool OrderRouter::passes_risk(const SimOrder& order) const noexcept {
    if (order.quantity == 0) return false;

    const int32_t delta = static_cast<int32_t>(order.quantity);
    const int32_t projected = position_ +
        (order.side == Side::BID ? delta : -delta);

    if (projected > cfg_.max_long_position)  return false;
    if (projected < cfg_.max_short_position) return false;

    return true;
}

// ── built-in mean-reversion strategy ─────────────────────────────────────────
// When flat: buy if ask exists and spread <= max_spread_ticks.
// When long: sell to unwind when bid exists.

OrderRouter::Strategy
OrderRouter::make_mean_reversion_strategy(const Config& cfg) {
    return [cfg](const TopOfBookUpdate& tob,
                 int32_t position,
                 uint64_t next_id) -> std::optional<SimOrder> {
        // Need both sides populated
        if (tob.bid_price == 0 || tob.ask_price == 0)
            return std::nullopt;

        const uint64_t spread = tob.ask_price - tob.bid_price;
        if (spread > cfg.max_spread_ticks) return std::nullopt;

        SimOrder o;
        o.id           = next_id;    // will be overwritten by submit()
        o.type         = SimOrder::Type::MARKET;
        o.quantity     = cfg.default_qty;
        o.stock_locate = tob.stock_locate;
        o.timestamp_ns = tob.timestamp_ns;

        if (position == 0) {
            // Enter long: buy at ask
            o.side  = Side::BID;
            o.price = tob.ask_price;
            return o;
        }
        if (position > 0) {
            // Unwind: sell at bid
            o.side     = Side::ASK;
            o.price    = tob.bid_price;
            o.quantity = static_cast<uint32_t>(
                std::min(static_cast<int32_t>(cfg.default_qty), position));
            return o;
        }
        return std::nullopt;
    };
}

void OrderRouter::reset_stats() noexcept {
    submitted_ = rejected_ = fills_ = 0;
}

} // namespace lob
