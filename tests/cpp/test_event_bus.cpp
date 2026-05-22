// Tests for TypedEventBus, OrderRouter, FillSimulator, and the full pipeline.
#include "../../third_party/catch.hpp"
#include "../../core/event_bus/event_bus.hpp"
#include "../../core/execution/order_router.hpp"
#include "../../core/execution/fill_simulator.hpp"
#include "../../core/lob/limit_order_book.hpp"
#include <vector>
#include <string>
#include <cstring>

using namespace sigma_edge;

// ═══════════════════════════════════════════════════════════════════════════════
//  TypedEventBus — basic dispatch
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("EventBus - publish reaches single subscriber", "[bus]") {
    SigmaEventBus bus;
    std::vector<FillEvent> received;
    bus.subscribe<FillEvent>([&](const FillEvent& f) { received.push_back(f); });

    bus.publish(FillEvent{42, 1500000, 100, true, 999});

    REQUIRE(received.size()     == 1);
    REQUIRE(received[0].order_id   == 42);
    REQUIRE(received[0].fill_price == 1500000);
    REQUIRE(received[0].fill_qty   == 100);
    REQUIRE(received[0].is_buy     == true);
    REQUIRE(received[0].timestamp_ns == 999);
}

TEST_CASE("EventBus - multiple subscribers to same type all fire", "[bus]") {
    SigmaEventBus bus;
    int a = 0, b = 0, c = 0;
    bus.subscribe<FillEvent>([&](const FillEvent&) { ++a; });
    bus.subscribe<FillEvent>([&](const FillEvent&) { ++b; });
    bus.subscribe<FillEvent>([&](const FillEvent&) { ++c; });

    bus.publish(FillEvent{1, 0, 0, false, 0});

    REQUIRE(a == 1); REQUIRE(b == 1); REQUIRE(c == 1);
}

TEST_CASE("EventBus - subscribers fire in registration order", "[bus]") {
    SigmaEventBus bus;
    std::vector<int> order;
    bus.subscribe<FillEvent>([&](const FillEvent&) { order.push_back(1); });
    bus.subscribe<FillEvent>([&](const FillEvent&) { order.push_back(2); });
    bus.subscribe<FillEvent>([&](const FillEvent&) { order.push_back(3); });

    bus.publish(FillEvent{});

    const std::vector<int> expected_order{1, 2, 3};
    REQUIRE(order == expected_order);
}

TEST_CASE("EventBus - wrong event type does not fire unrelated subscribers", "[bus]") {
    SigmaEventBus bus;
    int fill_count  = 0;
    int order_count = 0;

    bus.subscribe<FillEvent>([&](const FillEvent&)   { ++fill_count;  });
    bus.subscribe<OrderEvent>([&](const OrderEvent&) { ++order_count; });

    bus.publish(FillEvent{});

    REQUIRE(fill_count  == 1);
    REQUIRE(order_count == 0);   // not touched
}

TEST_CASE("EventBus - variant publish dispatches to correct handler", "[bus]") {
    SigmaEventBus bus;
    int signal_count = 0, order_count = 0, fill_count = 0;

    bus.subscribe<SignalEvent>([&](const SignalEvent&) { ++signal_count; });
    bus.subscribe<OrderEvent>([&](const OrderEvent&)  { ++order_count;  });
    bus.subscribe<FillEvent>([&](const FillEvent&)    { ++fill_count;   });

    SigmaEvent ev = SignalEvent{1, 0.7, Side::BID, 0, 0};
    bus.publish(ev);

    REQUIRE(signal_count == 1);
    REQUIRE(order_count  == 0);
    REQUIRE(fill_count   == 0);
}

TEST_CASE("EventBus - publish with no subscribers is a no-op", "[bus]") {
    SigmaEventBus bus;
    // Should not crash or throw
    bus.publish(FillEvent{});
    bus.publish(OrderEvent{});
    bus.publish(SignalEvent{});
    REQUIRE(true);
}

TEST_CASE("EventBus - unsubscribe removes handler", "[bus]") {
    SigmaEventBus bus;
    int count = 0;
    auto h = bus.subscribe<FillEvent>([&](const FillEvent&) { ++count; });

    bus.publish(FillEvent{});
    REQUIRE(count == 1);

    bus.unsubscribe(h);
    bus.publish(FillEvent{});
    REQUIRE(count == 1);   // handler no longer firing
}

TEST_CASE("EventBus - unsubscribe of invalid handle is a no-op", "[bus]") {
    SigmaEventBus bus;
    SubscriptionHandle invalid{0};
    bus.unsubscribe(invalid);   // must not crash
    REQUIRE(true);
}

TEST_CASE("EventBus - unsubscribe one of several keeps others active", "[bus]") {
    SigmaEventBus bus;
    int a = 0, b = 0;
    auto ha = bus.subscribe<FillEvent>([&](const FillEvent&) { ++a; });
    bus.subscribe<FillEvent>([&](const FillEvent&) { ++b; });

    bus.unsubscribe(ha);
    bus.publish(FillEvent{});

    REQUIRE(a == 0);
    REQUIRE(b == 1);
}

TEST_CASE("EventBus - handler_count tracks per-type subscriptions", "[bus]") {
    SigmaEventBus bus;
    REQUIRE(bus.handler_count<FillEvent>()  == 0);
    REQUIRE(bus.total_handler_count()       == 0);

    bus.subscribe<FillEvent>([](const FillEvent&) {});
    bus.subscribe<FillEvent>([](const FillEvent&) {});
    bus.subscribe<OrderEvent>([](const OrderEvent&) {});

    REQUIRE(bus.handler_count<FillEvent>()  == 2);
    REQUIRE(bus.handler_count<OrderEvent>() == 1);
    REQUIRE(bus.total_handler_count()       == 3);
}

TEST_CASE("EventBus - clear removes all subscriptions", "[bus]") {
    SigmaEventBus bus;
    int count = 0;
    bus.subscribe<FillEvent>([&](const FillEvent&) { ++count; });
    bus.subscribe<OrderEvent>([&](const OrderEvent&) { ++count; });

    bus.clear();
    bus.publish(FillEvent{});
    bus.publish(OrderEvent{});

    REQUIRE(count == 0);
    REQUIRE(bus.total_handler_count() == 0);
}

TEST_CASE("EventBus - direct publish<T> dispatches correctly", "[bus]") {
    SigmaEventBus bus;
    uint64_t seen_id = 0;
    bus.subscribe<FillEvent>([&](const FillEvent& f) { seen_id = f.order_id; });

    FillEvent fe{77, 2000000, 50, true, 0};
    bus.publish(fe);   // direct T overload

    REQUIRE(seen_id == 77);
}

TEST_CASE("EventBus - SignalEvent fields preserved through dispatch", "[bus]") {
    SigmaEventBus bus;
    SignalEvent received{};
    bus.subscribe<SignalEvent>([&](const SignalEvent& s) { received = s; });

    bus.publish(SignalEvent{10, 0.85, Side::BID, 3, 123456789});

    REQUIRE(received.signal_id  == 10);
    REQUIRE(received.score      == 0.85);
    REQUIRE(received.direction  == Side::BID);
    REQUIRE(received.stock_locate == 3);
    REQUIRE(received.timestamp_ns == 123456789);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  OrderRouter
// ═══════════════════════════════════════════════════════════════════════════════

// Helper: build a TopOfBookUpdate
static TopOfBookUpdate make_tob(uint64_t bid, uint64_t ask, uint64_t ts = 0) {
    return TopOfBookUpdate{1, bid, ask, ts};
}

TEST_CASE("OrderRouter - default strategy buys when flat and ask exists", "[router]") {
    SigmaEventBus bus;
    OrderRouter::Config cfg;
    cfg.max_spread_ticks = 1000;
    cfg.default_qty      = 100;
    OrderRouter router(bus, cfg);

    std::vector<OrderEvent> orders;
    bus.subscribe<OrderEvent>([&](const OrderEvent& ev) { orders.push_back(ev); });

    router.on_tob(make_tob(1499500, 1500000));  // spread = 500 ticks < max_spread_ticks(1000)

    REQUIRE(orders.size()          == 1);
    REQUIRE(orders[0].status       == OrderEvent::Status::SUBMITTED);
    REQUIRE(orders[0].order.side   == Side::BID);
    REQUIRE(orders[0].order.quantity == 100);
}

TEST_CASE("OrderRouter - no order when only one side of book exists", "[router]") {
    SigmaEventBus bus;
    OrderRouter router(bus);

    std::vector<OrderEvent> orders;
    bus.subscribe<OrderEvent>([&](const OrderEvent& ev) { orders.push_back(ev); });

    router.on_tob(make_tob(0, 1500000));       // no bid
    router.on_tob(make_tob(1490000, 0));       // no ask
    router.on_tob(make_tob(0, 0));            // empty book

    REQUIRE(orders.empty());
}

TEST_CASE("OrderRouter - no order when spread exceeds threshold", "[router]") {
    SigmaEventBus bus;
    OrderRouter::Config cfg;
    cfg.max_spread_ticks = 10;
    OrderRouter router(bus, cfg);

    std::vector<OrderEvent> orders;
    bus.subscribe<OrderEvent>([&](const OrderEvent& ev) { orders.push_back(ev); });

    router.on_tob(make_tob(1490000, 1500000));  // spread = 10000 > 10

    REQUIRE(orders.empty());
}

TEST_CASE("OrderRouter - long position limit blocks further buys", "[router]") {
    SigmaEventBus bus;
    OrderRouter::Config cfg;
    cfg.max_long_position = 100;
    cfg.default_qty       = 100;
    cfg.max_spread_ticks  = 1'000'000;
    OrderRouter router(bus, cfg);

    std::vector<OrderEvent> orders;
    bus.subscribe<OrderEvent>([&](const OrderEvent& ev) { orders.push_back(ev); });

    // First buy: accepted (0 + 100 = 100 == max)
    router.on_tob(make_tob(1490000, 1500000));
    REQUIRE(orders.back().status == OrderEvent::Status::SUBMITTED);

    // Simulate the fill so position updates
    bus.publish(FillEvent{orders.back().order.id, 1500000, 100, true, 0});
    REQUIRE(router.position() == 100);

    // Second buy would exceed limit: rejected
    orders.clear();
    router.on_tob(make_tob(1490000, 1500000));
    // At position 100 with max 100, default strategy switches to sell — check
    // it emits a SELL (unwind), not a second buy.
    // (If it emits a sell SUBMITTED, that's fine — it's unwinding, not buying)
    // For this test: just verify no rejected event was emitted for a buy
    for (const auto& ev : orders) {
        if (ev.status == OrderEvent::Status::SUBMITTED)
            REQUIRE(ev.order.side == Side::ASK);  // must be a sell (unwind)
    }
}

TEST_CASE("OrderRouter - custom strategy injected correctly", "[router]") {
    SigmaEventBus bus;
    int strategy_calls = 0;

    OrderRouter router(bus, OrderRouter::Config{}, [&](const TopOfBookUpdate& tob,
                                     int32_t pos, uint64_t id) -> std::optional<SimOrder> {
        ++strategy_calls;
        (void)pos; (void)id;
        // Only trade when ask == 9999999
        if (tob.ask_price != 9999999u) return std::nullopt;
        SimOrder o;
        o.id = id; o.price = tob.ask_price; o.quantity = 50;
        o.side = Side::BID; o.type = SimOrder::Type::MARKET;
        o.timestamp_ns = tob.timestamp_ns; o.stock_locate = tob.stock_locate;
        return o;
    });

    router.on_tob(make_tob(9990000, 1000000));    // won't trigger
    router.on_tob(make_tob(9990000, 9999999));    // triggers

    REQUIRE(strategy_calls == 2);

    std::vector<OrderEvent> orders;
    bus.subscribe<OrderEvent>([&](const OrderEvent& ev) { orders.push_back(ev); });
    router.on_tob(make_tob(9990000, 9999999));

    REQUIRE(orders.size() == 1);
    REQUIRE(orders[0].order.quantity == 50);
}

TEST_CASE("OrderRouter - order IDs are monotonically increasing", "[router]") {
    SigmaEventBus bus;
    OrderRouter::Config cfg;
    cfg.max_spread_ticks  = 1'000'000;
    cfg.default_qty       = 10;
    cfg.max_long_position = 10'000;
    OrderRouter router(bus, cfg);

    std::vector<uint64_t> ids;
    bus.subscribe<OrderEvent>([&](const OrderEvent& ev) {
        if (ev.status == OrderEvent::Status::SUBMITTED)
            ids.push_back(ev.order.id);
    });

    for (int i = 0; i < 5; ++i)
        router.on_tob(make_tob(1490000, 1500000));

    REQUIRE(ids.size() >= 2);
    for (size_t i = 1; i < ids.size(); ++i)
        REQUIRE(ids[i] > ids[i-1]);
}

TEST_CASE("OrderRouter - fill updates position correctly", "[router]") {
    SigmaEventBus bus;
    OrderRouter::Config cfg;
    cfg.default_qty      = 200;
    cfg.max_spread_ticks = 1'000'000;
    cfg.max_long_position = 10'000;
    OrderRouter router(bus, cfg);

    std::vector<OrderEvent> submitted;
    bus.subscribe<OrderEvent>([&](const OrderEvent& ev) {
        if (ev.status == OrderEvent::Status::SUBMITTED) submitted.push_back(ev);
    });

    router.on_tob(make_tob(1490000, 1500000));
    REQUIRE(!submitted.empty());
    const uint64_t oid = submitted.back().order.id;

    REQUIRE(router.position() == 0);   // not yet filled

    // Simulate fill arriving back
    bus.publish(FillEvent{oid, 1500001, 200, true, 100});
    REQUIRE(router.position()      == 200);
    REQUIRE(router.fills_received() == 1);
}

TEST_CASE("OrderRouter - reject publishes OrderEvent{REJECTED}", "[router]") {
    SigmaEventBus bus;
    OrderRouter::Config cfg;
    cfg.max_long_position = 0;   // nothing allowed
    cfg.default_qty       = 100;
    cfg.max_spread_ticks  = 1'000'000;
    OrderRouter router(bus, cfg);

    std::vector<OrderEvent> events;
    bus.subscribe<OrderEvent>([&](const OrderEvent& ev) { events.push_back(ev); });

    router.on_tob(make_tob(1490000, 1500000));

    bool found_reject = false;
    for (const auto& ev : events)
        if (ev.status == OrderEvent::Status::REJECTED) found_reject = true;
    REQUIRE(found_reject);
    REQUIRE(router.orders_rejected() >= 1);
}

TEST_CASE("OrderRouter - reset_stats clears counters", "[router]") {
    SigmaEventBus bus;
    OrderRouter::Config cfg;
    cfg.max_spread_ticks  = 1'000'000;
    cfg.max_long_position = 10'000;
    OrderRouter router(bus, cfg);

    router.on_tob(make_tob(1490000, 1500000));
    REQUIRE(router.orders_submitted() >= 1);

    router.reset_stats();
    REQUIRE(router.orders_submitted() == 0);
    REQUIRE(router.orders_rejected()  == 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  FillSimulator
// ═══════════════════════════════════════════════════════════════════════════════

// Helper: populate a LOB with a bid and ask
static LimitOrderBook make_book(uint64_t bid_price, uint32_t bid_qty,
                                 uint64_t ask_price, uint32_t ask_qty) {
    LimitOrderBook lob;
    if (bid_price)
        lob.add_order({1, bid_price, bid_qty, Side::BID, 0});
    if (ask_price)
        lob.add_order({2, ask_price, ask_qty, Side::ASK, 0});
    return lob;
}

// Helper: make a market buy SimOrder
static SimOrder make_mkt_buy(uint64_t qty, uint64_t ts = 0) {
    SimOrder o;
    o.id = 100; o.price = 0; o.quantity = static_cast<uint32_t>(qty);
    o.side = Side::BID; o.type = SimOrder::Type::MARKET;
    o.stock_locate = 1; o.timestamp_ns = ts;
    return o;
}
static SimOrder make_mkt_sell(uint64_t qty, uint64_t ts = 0) {
    SimOrder o;
    o.id = 101; o.price = 0; o.quantity = static_cast<uint32_t>(qty);
    o.side = Side::ASK; o.type = SimOrder::Type::MARKET;
    o.stock_locate = 1; o.timestamp_ns = ts;
    return o;
}

TEST_CASE("FillSimulator - market buy fills at best_ask + slippage", "[fill]") {
    auto lob = make_book(1490000, 100, 1500000, 200);
    SigmaEventBus bus;
    FillSimulator::Config cfg;
    cfg.slippage_ticks  = 5;
    cfg.fill_latency_ns = 0;   // immediate for this test
    FillSimulator sim(lob, bus, cfg);

    std::vector<FillEvent> fills;
    bus.subscribe<FillEvent>([&](const FillEvent& f) { fills.push_back(f); });

    sim.submit_direct(make_mkt_buy(100), 0);
    sim.tick(0);   // latency = 0 → fire immediately

    REQUIRE(fills.size()          == 1);
    REQUIRE(fills[0].fill_price   == 1500005u);   // 1500000 + 5 slippage
    REQUIRE(fills[0].fill_qty     == 100u);
    REQUIRE(fills[0].is_buy       == true);
}

TEST_CASE("FillSimulator - market sell fills at best_bid - slippage", "[fill]") {
    auto lob = make_book(1490000, 100, 1500000, 200);
    SigmaEventBus bus;
    FillSimulator::Config cfg;
    cfg.slippage_ticks  = 3;
    cfg.fill_latency_ns = 0;
    FillSimulator sim(lob, bus, cfg);

    std::vector<FillEvent> fills;
    bus.subscribe<FillEvent>([&](const FillEvent& f) { fills.push_back(f); });

    sim.submit_direct(make_mkt_sell(50), 0);
    sim.tick(0);

    REQUIRE(fills.size()        == 1);
    REQUIRE(fills[0].fill_price == 1489997u);   // 1490000 - 3
    REQUIRE(fills[0].is_buy     == false);
}

TEST_CASE("FillSimulator - no fill when no opposing side", "[fill]") {
    LimitOrderBook empty_lob;
    SigmaEventBus bus;
    FillSimulator::Config cfg;
    cfg.fill_latency_ns = 0;
    FillSimulator sim(empty_lob, bus, cfg);

    std::vector<OrderEvent> events;
    bus.subscribe<OrderEvent>([&](const OrderEvent& ev) { events.push_back(ev); });

    sim.submit_direct(make_mkt_buy(100), 0);
    sim.tick(0);

    // Should emit REJECTED, not FILLED
    REQUIRE(events.size() == 1);
    REQUIRE(events[0].status == OrderEvent::Status::REJECTED);
    REQUIRE(sim.total_rejections() == 1);
    REQUIRE(sim.total_fills()      == 0);
}

TEST_CASE("FillSimulator - fill latency delays execution", "[fill]") {
    auto lob = make_book(1490000, 100, 1500000, 200);
    SigmaEventBus bus;
    FillSimulator::Config cfg;
    cfg.fill_latency_ns = 1000;
    FillSimulator sim(lob, bus, cfg);

    std::vector<FillEvent> fills;
    bus.subscribe<FillEvent>([&](const FillEvent& f) { fills.push_back(f); });

    sim.submit_direct(make_mkt_buy(100), /*ts=*/0);

    sim.tick(500);    // too early
    REQUIRE(fills.empty());
    REQUIRE(sim.pending_count() == 1);

    sim.tick(999);    // still too early
    REQUIRE(fills.empty());

    sim.tick(1000);   // exactly on time
    REQUIRE(fills.size()      == 1);
    REQUIRE(sim.pending_count() == 0);
}

TEST_CASE("FillSimulator - multiple pending fills fire at correct times", "[fill]") {
    auto lob = make_book(1490000, 1000, 1500000, 1000);
    SigmaEventBus bus;
    FillSimulator::Config cfg;
    cfg.fill_latency_ns = 100;
    cfg.slippage_ticks  = 0;
    FillSimulator sim(lob, bus, cfg);

    std::vector<uint64_t> fill_times;
    bus.subscribe<FillEvent>([&](const FillEvent& f) {
        fill_times.push_back(f.timestamp_ns);
    });

    // Submit at t=0, 50, 200
    SimOrder o1 = make_mkt_buy(10, 0);   o1.id = 1;
    SimOrder o2 = make_mkt_buy(10, 50);  o2.id = 2;
    SimOrder o3 = make_mkt_buy(10, 200); o3.id = 3;
    sim.submit_direct(o1, 0);
    sim.submit_direct(o2, 50);
    sim.submit_direct(o3, 200);

    sim.tick(100);  // fires o1 (due=100), o2 not yet (due=150)
    REQUIRE(fill_times.size() == 1);
    REQUIRE(fill_times[0]     == 100);

    sim.tick(150);  // fires o2
    REQUIRE(fill_times.size() == 2);

    sim.tick(300);  // fires o3
    REQUIRE(fill_times.size() == 3);
}

TEST_CASE("FillSimulator - limit buy fills when ask crosses price", "[fill]") {
    auto lob = make_book(1490000, 100, 1500000, 100);
    SigmaEventBus bus;
    FillSimulator::Config cfg;
    cfg.fill_latency_ns = 0;
    cfg.slippage_ticks  = 0;
    FillSimulator sim(lob, bus, cfg);

    std::vector<FillEvent> fills;
    bus.subscribe<FillEvent>([&](const FillEvent& f) { fills.push_back(f); });

    // Limit buy at 1500000: best_ask=1500000 <= 1500000 → fills
    SimOrder limit_buy;
    limit_buy.id = 200; limit_buy.price = 1500000; limit_buy.quantity = 50;
    limit_buy.side = Side::BID; limit_buy.type = SimOrder::Type::LIMIT;
    limit_buy.stock_locate = 1; limit_buy.timestamp_ns = 0;
    sim.submit_direct(limit_buy, 0);
    sim.tick(0);

    REQUIRE(fills.size() == 1);
}

TEST_CASE("FillSimulator - limit buy rejected when ask above limit price", "[fill]") {
    auto lob = make_book(1490000, 100, 1510000, 100);
    SigmaEventBus bus;
    FillSimulator::Config cfg;
    cfg.fill_latency_ns = 0;
    FillSimulator sim(lob, bus, cfg);

    std::vector<OrderEvent> events;
    bus.subscribe<OrderEvent>([&](const OrderEvent& ev) { events.push_back(ev); });

    SimOrder limit_buy;
    limit_buy.id = 201; limit_buy.price = 1500000; limit_buy.quantity = 50;
    limit_buy.side = Side::BID; limit_buy.type = SimOrder::Type::LIMIT;
    limit_buy.timestamp_ns = 0;
    sim.submit_direct(limit_buy, 0);
    sim.tick(0);

    // best_ask=1510000 > limit 1500000 → rejected
    REQUIRE(events.back().status == OrderEvent::Status::REJECTED);
    REQUIRE(sim.total_rejections() == 1);
}

TEST_CASE("FillSimulator - both OrderEvent{FILLED} and FillEvent published", "[fill]") {
    auto lob = make_book(1490000, 100, 1500000, 100);
    SigmaEventBus bus;
    FillSimulator::Config cfg;
    cfg.fill_latency_ns = 0;
    FillSimulator sim(lob, bus, cfg);

    std::vector<OrderEvent::Status> statuses;
    int fill_events = 0;
    bus.subscribe<OrderEvent>([&](const OrderEvent& ev) {
        statuses.push_back(ev.status);
    });
    bus.subscribe<FillEvent>([&](const FillEvent&) { ++fill_events; });

    sim.submit_direct(make_mkt_buy(100), 0);
    sim.tick(0);

    REQUIRE(std::count(statuses.begin(), statuses.end(),
                       OrderEvent::Status::FILLED) == 1);
    REQUIRE(fill_events == 1);
}

TEST_CASE("FillSimulator - bus-driven OrderEvent{SUBMITTED} triggers fill", "[fill]") {
    auto lob = make_book(1490000, 100, 1500000, 200);
    SigmaEventBus bus;
    FillSimulator::Config cfg;
    cfg.fill_latency_ns = 500;
    FillSimulator sim(lob, bus, cfg);

    std::vector<FillEvent> fills;
    bus.subscribe<FillEvent>([&](const FillEvent& f) { fills.push_back(f); });

    // Publish a SUBMITTED order event as if from OrderRouter
    SimOrder o = make_mkt_buy(75, 100);
    OrderEvent ev;
    ev.status       = OrderEvent::Status::SUBMITTED;
    ev.order        = o;
    ev.timestamp_ns = 100;
    bus.publish(ev);

    sim.tick(599);   // too early
    REQUIRE(fills.empty());

    sim.tick(600);   // 100 + 500 = due at 600
    REQUIRE(fills.size()    == 1);
    REQUIRE(fills[0].fill_qty == 75);
}

TEST_CASE("FillSimulator - statistics accumulate correctly", "[fill]") {
    auto lob = make_book(1490000, 1000, 1500000, 1000);
    SigmaEventBus bus;
    FillSimulator::Config cfg;
    cfg.fill_latency_ns = 0;
    FillSimulator sim(lob, bus, cfg);

    for (int i = 0; i < 10; ++i) {
        SimOrder o = make_mkt_buy(50, 0);
        o.id = static_cast<uint64_t>(i);
        sim.submit_direct(o, 0);
    }
    sim.tick(0);

    REQUIRE(sim.total_fills()     == 10);
    REQUIRE(sim.total_filled_qty() == 500);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  End-to-end integration: FeedHandler → LOB → OrderRouter → FillSimulator
// ═══════════════════════════════════════════════════════════════════════════════

#include "../../core/feed_handler/feed_handler.hpp"

TEST_CASE("Integration - FeedHandler TOB wires to OrderRouter via bridge", "[e2e]") {
    LimitOrderBook lob;
    SigmaEventBus  bus;

    OrderRouter::Config cfg;
    cfg.max_spread_ticks  = 500'000;
    cfg.default_qty       = 100;
    cfg.max_long_position = 10'000;
    OrderRouter router(bus, cfg);

    FillSimulator::Config fcfg;
    fcfg.fill_latency_ns = 1000;
    fcfg.slippage_ticks  = 1;
    FillSimulator sim(lob, bus, fcfg);

    // Collect fill events
    std::vector<FillEvent> fills;
    bus.subscribe<FillEvent>([&](const FillEvent& f) { fills.push_back(f); });

    // Wire FeedHandler to OrderRouter via bridge callback
    FeedHandler fh(lob, [&](const MarketEvent& ev) {
        if (auto* tob = std::get_if<TopOfBookUpdate>(&ev))
            router.on_tob(*tob);
    });

    // Feed two orders into LOB via FeedHandler (raw ITCH bytes)
    auto make_add = [](uint64_t ref, char side, uint32_t qty,
                       uint32_t price, uint16_t locate) -> std::vector<uint8_t> {
        std::vector<uint8_t> buf(36, 0);
        buf[0]  = 'A';
        buf[1]  = uint8_t(locate >> 8); buf[2] = uint8_t(locate);
        buf[19] = uint8_t(side);
        auto p32 = [&buf](size_t off, uint32_t v) {
            buf[off]=(v>>24)&0xff; buf[off+1]=(v>>16)&0xff;
            buf[off+2]=(v>>8)&0xff; buf[off+3]=v&0xff;
        };
        auto p64 = [&buf](size_t off, uint64_t v) {
            for (int i=7;i>=0;--i){buf[off+i]=v&0xff;v>>=8;}
        };
        p64(11, ref);
        p32(20, qty);
        std::memset(buf.data()+24, ' ', 8);
        std::memcpy(buf.data()+24, "AAPL", 4);
        p32(32, price);
        return buf;
    };

    // Add a bid and ask to the LOB via the feed handler
    auto add_bid = make_add(1, 'B', 500, 1490000, 1);
    auto add_ask = make_add(2, 'S', 500, 1500000, 1);
    fh.process(*itch::ItchParser::parse(add_bid.data(), 36));
    fh.process(*itch::ItchParser::parse(add_ask.data(), 36));

    // LOB should now have best bid and ask
    REQUIRE(lob.best_bid() == 1490000u);
    REQUIRE(lob.best_ask() == 1500000u);

    // OrderRouter should have received a TOB update and submitted a buy
    REQUIRE(router.orders_submitted() >= 1);

    // Advance simulation clock to trigger fill
    sim.tick(2000);

    REQUIRE(fills.size()      >= 1);
    REQUIRE(fills[0].fill_qty == 100);
    REQUIRE(fills[0].is_buy   == true);
    REQUIRE(router.position() == 100);
}

TEST_CASE("Integration - position builds up then unwinds via strategy", "[e2e]") {
    LimitOrderBook lob;
    lob.add_order({1, 1490000, 10000, Side::BID, 0});
    lob.add_order({2, 1500000, 10000, Side::ASK, 0});

    SigmaEventBus bus;
    OrderRouter::Config cfg;
    cfg.default_qty       = 200;
    cfg.max_long_position = 200;
    cfg.max_spread_ticks  = 1'000'000;
    OrderRouter router(bus, cfg);

    FillSimulator::Config fcfg;
    fcfg.fill_latency_ns = 0;
    fcfg.slippage_ticks  = 0;
    FillSimulator sim(lob, bus, fcfg);

    // Force a TOB → buy
    router.on_tob(TopOfBookUpdate{1, 1490000, 1500000, 100});
    sim.tick(100);    // immediate fill
    REQUIRE(router.position() == 200);   // long 200

    // Force another TOB → strategy should now unwind (sell)
    std::vector<OrderEvent> events;
    bus.subscribe<OrderEvent>([&](const OrderEvent& ev) { events.push_back(ev); });

    router.on_tob(TopOfBookUpdate{1, 1490000, 1500000, 200});
    sim.tick(200);

    // A sell should have been submitted and filled
    bool found_sell = false;
    for (const auto& ev : events)
        if (ev.status == OrderEvent::Status::SUBMITTED &&
            ev.order.side == Side::ASK)
            found_sell = true;
    REQUIRE(found_sell);
    REQUIRE(router.position() == 0);   // back to flat
}

TEST_CASE("Integration - signal event round-trips through bus", "[e2e]") {
    SigmaEventBus bus;
    std::vector<SignalEvent> signals;
    bus.subscribe<SignalEvent>([&](const SignalEvent& s) { signals.push_back(s); });

    bus.publish(SignalEvent{1, 0.9, Side::BID, 5, 12345});
    bus.publish(SignalEvent{2, -0.5, Side::ASK, 5, 67890});

    REQUIRE(signals.size()    == 2);
    REQUIRE(signals[0].score  == 0.9);
    REQUIRE(signals[1].score  == -0.5);
}
