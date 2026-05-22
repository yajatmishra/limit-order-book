#pragma once
#include <cstdint>

namespace sigma_edge {

enum class Side : uint8_t { BID = 0, ASK = 1 };

// POD representing a single resting limit order in the book.
// price is stored as integer ticks (e.g. dollar_price * 10'000) to avoid
// floating-point comparisons as map keys.
struct Order {
    uint64_t id;         // unique order reference number (from ITCH)
    uint64_t price;      // integer ticks
    uint32_t quantity;   // shares remaining (decremented on partial fills)
    Side     side;
    uint64_t timestamp;  // nanoseconds since midnight
};

} // namespace sigma_edge
