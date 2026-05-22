#pragma once
#include "order.hpp"
#include "price_level.hpp"
#include <map>
#include <unordered_map>
#include <optional>
#include <functional>
#include <vector>
#include <cstdint>

namespace sigma_edge {

struct DepthLevel {
    uint64_t price;
    uint64_t quantity;
    size_t   order_count;
};

class LimitOrderBook {
public:
    LimitOrderBook() = default;

    // Returns false if order_id already exists (duplicate).
    bool add_order(const Order& order);

    // Returns false if order_id not found.
    bool cancel_order(uint64_t order_id);

    // Reduces qty on the named order. Removes it when fully consumed.
    // Returns false if order_id not found.
    bool execute_order(uint64_t order_id, uint32_t qty);

    // ITCH Order Replace: cancel old_id, insert new_order atomically.
    // new_order must carry its own new id. Returns false if old_id not found
    // or new_order.id already exists.
    bool replace_order(uint64_t old_id, const Order& new_order);

    // Top-of-book queries — empty if the respective side has no orders.
    std::optional<uint64_t> best_bid()  const noexcept;
    std::optional<uint64_t> best_ask()  const noexcept;
    std::optional<double>   mid_price() const noexcept;
    std::optional<uint64_t> spread()    const noexcept;

    // Depth snapshots: bids descending, asks ascending, up to `levels` entries.
    std::vector<DepthLevel> bid_depth(size_t levels = 10) const;
    std::vector<DepthLevel> ask_depth(size_t levels = 10) const;

    size_t total_orders()    const noexcept { return order_map_.size(); }
    size_t bid_level_count() const noexcept { return bids_.size(); }
    size_t ask_level_count() const noexcept { return asks_.size(); }
    bool   empty()           const noexcept { return order_map_.empty(); }

private:
    // Bids: highest price first (std::greater → descending map).
    std::map<uint64_t, PriceLevel, std::greater<uint64_t>> bids_;
    // Asks: lowest price first (default ascending map).
    std::map<uint64_t, PriceLevel>                         asks_;

    struct Loc { Side side; uint64_t price; };
    std::unordered_map<uint64_t, Loc> order_map_;  // order_id → (side, price)
};

} // namespace sigma_edge
