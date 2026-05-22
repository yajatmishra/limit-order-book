#pragma once
#include "order.hpp"
#include <cstddef>
#include <cstdint>
#include <deque>
#include <utility>

namespace sigma_edge {

// All resting orders at a single price, maintained in FIFO arrival order.
// Supports O(1) add (push_back), O(n) cancel/execute by order_id.
// In practice n stays small (< ~50 orders per level on liquid equities).
class PriceLevel {
public:
    explicit PriceLevel(uint64_t price) noexcept;

    void add_order(const Order& order);

    // Remove the order with this id (cancel). Returns false if not found.
    bool cancel_order(uint64_t order_id);

    // Reduce qty on the given order. Removes the order when fully consumed.
    // Returns {found, fully_consumed}.
    std::pair<bool, bool> execute_order(uint64_t order_id, uint32_t qty);

    uint64_t price()       const noexcept { return price_; }
    uint64_t total_qty()   const noexcept { return total_qty_; }
    bool     empty()       const noexcept { return orders_.empty(); }
    size_t   order_count() const noexcept { return orders_.size(); }

    const std::deque<Order>& orders() const noexcept { return orders_; }

private:
    uint64_t          price_;
    uint64_t          total_qty_{0};
    std::deque<Order> orders_;
};

} // namespace sigma_edge
