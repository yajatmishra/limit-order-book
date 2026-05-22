#include "limit_order_book.hpp"

namespace sigma_edge {

// ── add ──────────────────────────────────────────────────────────────────────

bool LimitOrderBook::add_order(const Order& order) {
    if (order_map_.count(order.id)) return false;

    if (order.side == Side::BID) {
        // emplace is a no-op on the key if it already exists; either way the
        // iterator points to the correct level.
        auto [it, _] = bids_.emplace(order.price, PriceLevel(order.price));
        it->second.add_order(order);
    } else {
        auto [it, _] = asks_.emplace(order.price, PriceLevel(order.price));
        it->second.add_order(order);
    }
    order_map_[order.id] = {order.side, order.price};
    return true;
}

// ── cancel ───────────────────────────────────────────────────────────────────

bool LimitOrderBook::cancel_order(uint64_t order_id) {
    auto mit = order_map_.find(order_id);
    if (mit == order_map_.end()) return false;

    const auto [side, price] = mit->second;

    if (side == Side::BID) {
        auto lit = bids_.find(price);
        if (lit != bids_.end()) {
            lit->second.cancel_order(order_id);
            if (lit->second.empty()) bids_.erase(lit);
        }
    } else {
        auto lit = asks_.find(price);
        if (lit != asks_.end()) {
            lit->second.cancel_order(order_id);
            if (lit->second.empty()) asks_.erase(lit);
        }
    }
    order_map_.erase(mit);
    return true;
}

// ── execute ──────────────────────────────────────────────────────────────────

bool LimitOrderBook::execute_order(uint64_t order_id, uint32_t qty) {
    auto mit = order_map_.find(order_id);
    if (mit == order_map_.end()) return false;

    const auto [side, price] = mit->second;

    if (side == Side::BID) {
        auto lit = bids_.find(price);
        if (lit == bids_.end()) return false;
        auto [found, fully_consumed] = lit->second.execute_order(order_id, qty);
        if (!found) return false;
        if (lit->second.empty()) bids_.erase(lit);
        if (fully_consumed) order_map_.erase(mit);
    } else {
        auto lit = asks_.find(price);
        if (lit == asks_.end()) return false;
        auto [found, fully_consumed] = lit->second.execute_order(order_id, qty);
        if (!found) return false;
        if (lit->second.empty()) asks_.erase(lit);
        if (fully_consumed) order_map_.erase(mit);
    }
    return true;
}

// ── replace ──────────────────────────────────────────────────────────────────

bool LimitOrderBook::replace_order(uint64_t old_id, const Order& new_order) {
    if (!order_map_.count(old_id))       return false;
    if (order_map_.count(new_order.id))  return false;  // new id already exists
    cancel_order(old_id);
    return add_order(new_order);
}

// ── top-of-book ──────────────────────────────────────────────────────────────

std::optional<uint64_t> LimitOrderBook::best_bid() const noexcept {
    if (bids_.empty()) return std::nullopt;
    return bids_.begin()->first;
}

std::optional<uint64_t> LimitOrderBook::best_ask() const noexcept {
    if (asks_.empty()) return std::nullopt;
    return asks_.begin()->first;
}

std::optional<double> LimitOrderBook::mid_price() const noexcept {
    auto bid = best_bid();
    auto ask = best_ask();
    if (!bid || !ask) return std::nullopt;
    return 0.5 * (static_cast<double>(*bid) + static_cast<double>(*ask));
}

std::optional<uint64_t> LimitOrderBook::spread() const noexcept {
    auto bid = best_bid();
    auto ask = best_ask();
    if (!bid || !ask) return std::nullopt;
    return *ask - *bid;
}

// ── depth snapshots ──────────────────────────────────────────────────────────

std::vector<DepthLevel> LimitOrderBook::bid_depth(size_t levels) const {
    std::vector<DepthLevel> depth;
    depth.reserve(levels);
    for (auto it = bids_.begin(); it != bids_.end() && depth.size() < levels; ++it) {
        depth.push_back(DepthLevel{it->first, it->second.total_qty(), it->second.order_count()});
    }
    return depth;
}

std::vector<DepthLevel> LimitOrderBook::ask_depth(size_t levels) const {
    std::vector<DepthLevel> depth;
    depth.reserve(levels);
    for (auto it = asks_.begin(); it != asks_.end() && depth.size() < levels; ++it) {
        depth.push_back(DepthLevel{it->first, it->second.total_qty(), it->second.order_count()});
    }
    return depth;
}

} // namespace sigma_edge
