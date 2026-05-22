#include "price_level.hpp"
#include <algorithm>

namespace sigma_edge {

PriceLevel::PriceLevel(uint64_t price) noexcept : price_(price) {}

void PriceLevel::add_order(const Order& order) {
    orders_.push_back(order);
    total_qty_ += order.quantity;
}

bool PriceLevel::cancel_order(uint64_t order_id) {
    auto it = std::find_if(orders_.begin(), orders_.end(),
        [order_id](const Order& o) { return o.id == order_id; });
    if (it == orders_.end()) return false;
    total_qty_ -= it->quantity;
    orders_.erase(it);
    return true;
}

std::pair<bool, bool> PriceLevel::execute_order(uint64_t order_id, uint32_t qty) {
    auto it = std::find_if(orders_.begin(), orders_.end(),
        [order_id](const Order& o) { return o.id == order_id; });
    if (it == orders_.end()) return {false, false};

    uint32_t consumed  = std::min(qty, it->quantity);
    it->quantity      -= consumed;
    total_qty_        -= consumed;

    if (it->quantity == 0) {
        orders_.erase(it);
        return {true, true};   // found, fully consumed
    }
    return {true, false};      // found, partially consumed
}

} // namespace sigma_edge
