#include "../../third_party/catch.hpp"
#include "../../core/lob/order.hpp"
#include "../../core/lob/price_level.hpp"
#include "../../core/lob/limit_order_book.hpp"

#include <algorithm>
#include <random>
#include <vector>
#include <numeric>

using namespace lob;

// ── helpers ──────────────────────────────────────────────────────────────────

static Order make_order(uint64_t id, uint64_t price, uint32_t qty, Side side,
                        uint64_t ts = 0) {
    return Order{id, price, qty, side, ts};
}

// ── PriceLevel unit tests ─────────────────────────────────────────────────────

TEST_CASE("PriceLevel - add orders accumulates total_qty", "[price_level]") {
    PriceLevel lvl(1000);
    REQUIRE(lvl.empty());
    REQUIRE(lvl.total_qty() == 0);

    lvl.add_order(make_order(1, 1000, 100, Side::BID));
    REQUIRE(lvl.total_qty() == 100);
    REQUIRE(lvl.order_count() == 1);

    lvl.add_order(make_order(2, 1000, 200, Side::BID));
    REQUIRE(lvl.total_qty() == 300);
    REQUIRE(lvl.order_count() == 2);
}

TEST_CASE("PriceLevel - cancel removes order and adjusts qty", "[price_level]") {
    PriceLevel lvl(1000);
    lvl.add_order(make_order(1, 1000, 100, Side::BID));
    lvl.add_order(make_order(2, 1000, 200, Side::BID));

    REQUIRE(lvl.cancel_order(1) == true);
    REQUIRE(lvl.total_qty() == 200);
    REQUIRE(lvl.order_count() == 1);

    REQUIRE(lvl.cancel_order(99) == false);  // non-existent
}

TEST_CASE("PriceLevel - cancel last order leaves level empty", "[price_level]") {
    PriceLevel lvl(1000);
    lvl.add_order(make_order(1, 1000, 50, Side::BID));
    lvl.cancel_order(1);
    REQUIRE(lvl.empty());
    REQUIRE(lvl.total_qty() == 0);
}

TEST_CASE("PriceLevel - partial execute reduces qty, order stays", "[price_level]") {
    PriceLevel lvl(1000);
    lvl.add_order(make_order(1, 1000, 100, Side::BID));

    auto [found, fully] = lvl.execute_order(1, 40);
    REQUIRE(found == true);
    REQUIRE(fully == false);
    REQUIRE(lvl.total_qty() == 60);
    REQUIRE(lvl.order_count() == 1);
    REQUIRE(lvl.orders().front().quantity == 60);
}

TEST_CASE("PriceLevel - full execute removes order", "[price_level]") {
    PriceLevel lvl(1000);
    lvl.add_order(make_order(1, 1000, 100, Side::BID));

    auto [found, fully] = lvl.execute_order(1, 100);
    REQUIRE(found == true);
    REQUIRE(fully == true);
    REQUIRE(lvl.empty());
    REQUIRE(lvl.total_qty() == 0);
}

TEST_CASE("PriceLevel - execute unknown order returns not-found", "[price_level]") {
    PriceLevel lvl(1000);
    lvl.add_order(make_order(1, 1000, 100, Side::BID));

    auto [found, fully] = lvl.execute_order(999, 50);
    REQUIRE(found == false);
    REQUIRE(fully == false);
    REQUIRE(lvl.total_qty() == 100);  // unchanged
}

TEST_CASE("PriceLevel - FIFO: orders preserved in insertion order", "[price_level]") {
    PriceLevel lvl(1000);
    lvl.add_order(make_order(10, 1000, 100, Side::BID, 1));
    lvl.add_order(make_order(20, 1000, 200, Side::BID, 2));
    lvl.add_order(make_order(30, 1000, 300, Side::BID, 3));

    const auto& orders = lvl.orders();
    REQUIRE(orders.size() == 3);
    REQUIRE(orders[0].id == 10);
    REQUIRE(orders[1].id == 20);
    REQUIRE(orders[2].id == 30);
}

// ── LimitOrderBook — empty book ───────────────────────────────────────────────

TEST_CASE("LOB - empty book has no best bid or ask", "[lob]") {
    LimitOrderBook book;
    REQUIRE(book.empty());
    REQUIRE_FALSE(book.best_bid().has_value());
    REQUIRE_FALSE(book.best_ask().has_value());
    REQUIRE_FALSE(book.mid_price().has_value());
    REQUIRE_FALSE(book.spread().has_value());
    REQUIRE(book.total_orders() == 0);
}

// ── LimitOrderBook — add ──────────────────────────────────────────────────────

TEST_CASE("LOB - add single bid", "[lob]") {
    LimitOrderBook book;
    REQUIRE(book.add_order(make_order(1, 10000, 100, Side::BID)));
    REQUIRE(book.best_bid() == 10000u);
    REQUIRE_FALSE(book.best_ask().has_value());
    REQUIRE(book.total_orders() == 1);
}

TEST_CASE("LOB - add single ask", "[lob]") {
    LimitOrderBook book;
    REQUIRE(book.add_order(make_order(1, 10050, 100, Side::ASK)));
    REQUIRE(book.best_ask() == 10050u);
    REQUIRE_FALSE(book.best_bid().has_value());
}

TEST_CASE("LOB - best bid is highest bid price", "[lob]") {
    LimitOrderBook book;
    book.add_order(make_order(1, 9900, 100, Side::BID));
    book.add_order(make_order(2, 10000, 100, Side::BID));
    book.add_order(make_order(3, 9950, 100, Side::BID));
    REQUIRE(book.best_bid() == 10000u);
}

TEST_CASE("LOB - best ask is lowest ask price", "[lob]") {
    LimitOrderBook book;
    book.add_order(make_order(1, 10100, 100, Side::ASK));
    book.add_order(make_order(2, 10050, 100, Side::ASK));
    book.add_order(make_order(3, 10200, 100, Side::ASK));
    REQUIRE(book.best_ask() == 10050u);
}

TEST_CASE("LOB - mid price and spread computed correctly", "[lob]") {
    LimitOrderBook book;
    book.add_order(make_order(1, 10000, 100, Side::BID));
    book.add_order(make_order(2, 10050, 100, Side::ASK));

    REQUIRE(book.mid_price().has_value());
    REQUIRE(book.mid_price().value() == 10025.0);
    REQUIRE(book.spread() == 50u);
}

TEST_CASE("LOB - duplicate order id rejected", "[lob]") {
    LimitOrderBook book;
    REQUIRE(book.add_order(make_order(1, 10000, 100, Side::BID)));
    REQUIRE_FALSE(book.add_order(make_order(1, 10010, 200, Side::BID)));
    REQUIRE(book.total_orders() == 1);
}

// ── LimitOrderBook — cancel ───────────────────────────────────────────────────

TEST_CASE("LOB - cancel reduces order count", "[lob]") {
    LimitOrderBook book;
    book.add_order(make_order(1, 10000, 100, Side::BID));
    book.add_order(make_order(2, 10000, 200, Side::BID));

    REQUIRE(book.cancel_order(1));
    REQUIRE(book.total_orders() == 1);
    REQUIRE(book.bid_level_count() == 1);   // level still there
}

TEST_CASE("LOB - cancel last order at a level removes that level", "[lob]") {
    LimitOrderBook book;
    book.add_order(make_order(1, 10000, 100, Side::BID));
    book.add_order(make_order(2, 9990,  100, Side::BID));

    book.cancel_order(1);
    REQUIRE(book.bid_level_count() == 1);
    REQUIRE(book.best_bid() == 9990u);
}

TEST_CASE("LOB - cancel adjusts best bid when top level removed", "[lob]") {
    LimitOrderBook book;
    book.add_order(make_order(1, 10000, 100, Side::BID));
    book.add_order(make_order(2, 9990,  100, Side::BID));

    book.cancel_order(1);  // removes the top bid level
    REQUIRE(book.best_bid() == 9990u);
}

TEST_CASE("LOB - cancel non-existent order returns false", "[lob]") {
    LimitOrderBook book;
    REQUIRE_FALSE(book.cancel_order(999));
}

TEST_CASE("LOB - cancel then re-add same id succeeds", "[lob]") {
    LimitOrderBook book;
    book.add_order(make_order(1, 10000, 100, Side::BID));
    book.cancel_order(1);
    REQUIRE(book.add_order(make_order(1, 10000, 200, Side::BID)));
    REQUIRE(book.total_orders() == 1);
}

// ── LimitOrderBook — execute ──────────────────────────────────────────────────

TEST_CASE("LOB - partial execute reduces quantity, order stays", "[lob]") {
    LimitOrderBook book;
    book.add_order(make_order(1, 10000, 100, Side::BID));

    REQUIRE(book.execute_order(1, 40));
    REQUIRE(book.total_orders() == 1);   // order still tracked
    REQUIRE(book.bid_level_count() == 1);

    auto depth = book.bid_depth(1);
    REQUIRE(depth[0].quantity == 60);
}

TEST_CASE("LOB - full execute removes order and level when last", "[lob]") {
    LimitOrderBook book;
    book.add_order(make_order(1, 10000, 100, Side::BID));

    REQUIRE(book.execute_order(1, 100));
    REQUIRE(book.empty());
    REQUIRE_FALSE(book.best_bid().has_value());
}

TEST_CASE("LOB - full execute on one order leaves others in level", "[lob]") {
    LimitOrderBook book;
    book.add_order(make_order(1, 10000, 100, Side::BID));
    book.add_order(make_order(2, 10000, 200, Side::BID));

    REQUIRE(book.execute_order(1, 100));
    REQUIRE(book.total_orders() == 1);
    REQUIRE(book.bid_level_count() == 1);

    auto depth = book.bid_depth(1);
    REQUIRE(depth[0].quantity == 200);
}

TEST_CASE("LOB - execute non-existent order returns false", "[lob]") {
    LimitOrderBook book;
    REQUIRE_FALSE(book.execute_order(999, 100));
}

TEST_CASE("LOB - execute on ask side works correctly", "[lob]") {
    LimitOrderBook book;
    book.add_order(make_order(1, 10050, 100, Side::ASK));

    REQUIRE(book.execute_order(1, 100));
    REQUIRE(book.empty());
    REQUIRE_FALSE(book.best_ask().has_value());
}

// ── LimitOrderBook — replace ──────────────────────────────────────────────────

TEST_CASE("LOB - replace changes price and inserts new order", "[lob]") {
    LimitOrderBook book;
    book.add_order(make_order(1, 10000, 100, Side::BID));

    Order replacement = make_order(2, 9980, 150, Side::BID);
    REQUIRE(book.replace_order(1, replacement));

    REQUIRE(book.total_orders() == 1);
    REQUIRE(book.best_bid() == 9980u);
}

TEST_CASE("LOB - replace non-existent old id returns false", "[lob]") {
    LimitOrderBook book;
    REQUIRE_FALSE(book.replace_order(999, make_order(2, 10000, 100, Side::BID)));
}

TEST_CASE("LOB - replace rejects collision with existing new id", "[lob]") {
    LimitOrderBook book;
    book.add_order(make_order(1, 10000, 100, Side::BID));
    book.add_order(make_order(2, 9990,  100, Side::BID));

    // Try to replace order 1 with new id=2, which already exists
    REQUIRE_FALSE(book.replace_order(1, make_order(2, 9980, 150, Side::BID)));
    REQUIRE(book.total_orders() == 2);  // unchanged
}

TEST_CASE("LOB - replace updates best bid when price improves", "[lob]") {
    LimitOrderBook book;
    book.add_order(make_order(1, 10000, 100, Side::BID));
    book.add_order(make_order(2, 9990,  100, Side::BID));

    // Replace the 9990 order at a better price
    REQUIRE(book.replace_order(2, make_order(3, 10010, 100, Side::BID)));
    REQUIRE(book.best_bid() == 10010u);
}

// ── Depth snapshots ───────────────────────────────────────────────────────────

TEST_CASE("LOB - bid depth is in descending price order", "[lob]") {
    LimitOrderBook book;
    book.add_order(make_order(1, 9900,  100, Side::BID));
    book.add_order(make_order(2, 10000, 200, Side::BID));
    book.add_order(make_order(3, 9950,  300, Side::BID));

    auto depth = book.bid_depth(10);
    REQUIRE(depth.size() == 3);
    REQUIRE(depth[0].price == 10000);
    REQUIRE(depth[1].price == 9950);
    REQUIRE(depth[2].price == 9900);
}

TEST_CASE("LOB - ask depth is in ascending price order", "[lob]") {
    LimitOrderBook book;
    book.add_order(make_order(1, 10200, 100, Side::ASK));
    book.add_order(make_order(2, 10050, 200, Side::ASK));
    book.add_order(make_order(3, 10100, 300, Side::ASK));

    auto depth = book.ask_depth(10);
    REQUIRE(depth.size() == 3);
    REQUIRE(depth[0].price == 10050);
    REQUIRE(depth[1].price == 10100);
    REQUIRE(depth[2].price == 10200);
}

TEST_CASE("LOB - depth respects levels cap", "[lob]") {
    LimitOrderBook book;
    for (uint64_t i = 0; i < 20; ++i)
        book.add_order(make_order(i, 10000 - i * 10, 100, Side::BID));

    REQUIRE(book.bid_depth(5).size() == 5);
    REQUIRE(book.bid_depth(10).size() == 10);
    REQUIRE(book.bid_depth(30).size() == 20);   // capped at actual level count
}

TEST_CASE("LOB - depth levels carry correct aggregate qty", "[lob]") {
    LimitOrderBook book;
    book.add_order(make_order(1, 10000, 100, Side::BID));
    book.add_order(make_order(2, 10000, 200, Side::BID));  // same price level
    book.add_order(make_order(3, 9990,  150, Side::BID));

    auto depth = book.bid_depth(10);
    REQUIRE(depth[0].price    == 10000);
    REQUIRE(depth[0].quantity == 300);           // 100 + 200
    REQUIRE(depth[0].order_count == 2);
    REQUIRE(depth[1].quantity == 150);
}

// ── Multi-level interaction ───────────────────────────────────────────────────

TEST_CASE("LOB - mixed bid/ask operations maintain correct spread", "[lob]") {
    LimitOrderBook book;
    book.add_order(make_order(1, 9990, 100, Side::BID));
    book.add_order(make_order(2, 9995, 100, Side::BID));
    book.add_order(make_order(3, 10005, 100, Side::ASK));
    book.add_order(make_order(4, 10010, 100, Side::ASK));

    REQUIRE(book.best_bid() == 9995u);
    REQUIRE(book.best_ask() == 10005u);
    REQUIRE(book.spread() == 10u);

    book.cancel_order(2);  // remove top bid
    REQUIRE(book.best_bid() == 9990u);
    REQUIRE(book.spread() == 15u);
}

// ── Stress test ───────────────────────────────────────────────────────────────

TEST_CASE("LOB - stress: 1000 orders, cancel half, execute rest", "[lob][stress]") {
    LimitOrderBook book;
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint64_t> price_dist(9900, 10100);
    std::uniform_int_distribution<uint32_t> qty_dist(1, 1000);

    const int N = 1000;
    std::vector<uint64_t> ids(N);
    std::iota(ids.begin(), ids.end(), 1);

    // Add N orders alternating bid/ask around the spread
    for (int i = 0; i < N; ++i) {
        Side side = (i % 2 == 0) ? Side::BID : Side::ASK;
        uint64_t px = price_dist(rng);
        // Keep bid < ask spread: clamp bids below 10000, asks above
        if (side == Side::BID && px >= 10000) px = 9999;
        if (side == Side::ASK && px <= 10000) px = 10001;
        book.add_order(make_order(ids[i], px, qty_dist(rng), side));
    }
    REQUIRE(book.total_orders() == N);

    // Cancel the first 500
    for (int i = 0; i < N / 2; ++i) {
        REQUIRE(book.cancel_order(ids[i]));
    }
    REQUIRE(book.total_orders() == N / 2);

    // Execute the remaining 500 (full fills)
    for (int i = N / 2; i < N; ++i) {
        REQUIRE(book.execute_order(ids[i], 1'000'000));  // qty > any order qty → full fill
    }
    REQUIRE(book.empty());

    // Book must be fully drained
    REQUIRE_FALSE(book.best_bid().has_value());
    REQUIRE_FALSE(book.best_ask().has_value());
}

TEST_CASE("LOB - stress: book never crossed after random ops", "[lob][stress]") {
    LimitOrderBook book;
    std::mt19937 rng(7);
    std::uniform_int_distribution<uint64_t> price_bid(9950, 9999);
    std::uniform_int_distribution<uint64_t> price_ask(10001, 10050);
    std::uniform_int_distribution<uint32_t> qty_dist(1, 500);

    const int ROUNDS = 500;
    uint64_t next_id = 1;
    std::vector<uint64_t> live_bids, live_asks;

    for (int r = 0; r < ROUNDS; ++r) {
        // Add a bid and an ask
        uint64_t bid_id = next_id++;
        uint64_t ask_id = next_id++;
        book.add_order(make_order(bid_id, price_bid(rng), qty_dist(rng), Side::BID));
        book.add_order(make_order(ask_id, price_ask(rng), qty_dist(rng), Side::ASK));
        live_bids.push_back(bid_id);
        live_asks.push_back(ask_id);

        // Randomly cancel one from each side when we have enough
        if (live_bids.size() > 10) {
            std::uniform_int_distribution<size_t> idx(0, live_bids.size() - 1);
            size_t i = idx(rng);
            book.cancel_order(live_bids[i]);
            live_bids.erase(live_bids.begin() + static_cast<long>(i));
        }
        if (live_asks.size() > 10) {
            std::uniform_int_distribution<size_t> idx(0, live_asks.size() - 1);
            size_t i = idx(rng);
            book.cancel_order(live_asks[i]);
            live_asks.erase(live_asks.begin() + static_cast<long>(i));
        }

        // Invariant: best_bid < best_ask (no crossed book)
        if (book.best_bid().has_value() && book.best_ask().has_value()) {
            REQUIRE(book.best_bid().value() < book.best_ask().value());
        }
    }
}
