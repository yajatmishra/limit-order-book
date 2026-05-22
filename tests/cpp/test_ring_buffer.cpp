// Tests for SPSCRingBuffer, FeedHandler, and ShmWriter.
#include "../../third_party/catch.hpp"
#include "../../core/shared_memory/ring_buffer.hpp"
#include "../../core/shared_memory/shm_writer.hpp"
#include "../../core/feed_handler/feed_handler.hpp"
#include "../../core/feed_handler/itch_parser.hpp"
#include "../../core/lob/limit_order_book.hpp"
#include <thread>
#include <vector>
#include <numeric>
#include <cstring>

using namespace lob;
using namespace lob::itch;

// ═══════════════════════════════════════════════════════════════════════════════
//  SPSCRingBuffer — single-threaded correctness
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("RingBuffer - empty on construction", "[ring]") {
    SPSCRingBuffer<int, 8> rb;
    REQUIRE(rb.empty());
    REQUIRE(!rb.full());
    REQUIRE(rb.size() == 0);
    REQUIRE(rb.capacity() == 8);
}

TEST_CASE("RingBuffer - try_pop on empty returns false", "[ring]") {
    SPSCRingBuffer<int, 4> rb;
    int v = 0;
    REQUIRE(!rb.try_pop(v));
}

TEST_CASE("RingBuffer - single push/pop roundtrip", "[ring]") {
    SPSCRingBuffer<int, 4> rb;
    REQUIRE(rb.try_push(42));
    REQUIRE(rb.size() == 1);
    REQUIRE(!rb.empty());

    int v = 0;
    REQUIRE(rb.try_pop(v));
    REQUIRE(v == 42);
    REQUIRE(rb.empty());
}

TEST_CASE("RingBuffer - fill to capacity", "[ring]") {
    SPSCRingBuffer<int, 4> rb;
    REQUIRE(rb.try_push(1));
    REQUIRE(rb.try_push(2));
    REQUIRE(rb.try_push(3));
    REQUIRE(rb.try_push(4));
    REQUIRE(rb.full());
    REQUIRE(rb.size() == 4);

    // One more push must fail
    REQUIRE(!rb.try_push(5));
}

TEST_CASE("RingBuffer - FIFO ordering preserved", "[ring]") {
    SPSCRingBuffer<int, 8> rb;
    for (int i = 0; i < 5; ++i) REQUIRE(rb.try_push(i * 10));

    for (int i = 0; i < 5; ++i) {
        int v = -1;
        REQUIRE(rb.try_pop(v));
        REQUIRE(v == i * 10);
    }
    REQUIRE(rb.empty());
}

TEST_CASE("RingBuffer - wraparound across index boundary", "[ring]") {
    // Push 3, pop 3, push 3 more — indices wrap around the power-of-two boundary.
    SPSCRingBuffer<int, 4> rb;
    REQUIRE(rb.try_push(10));
    REQUIRE(rb.try_push(20));
    REQUIRE(rb.try_push(30));

    int v = 0;
    REQUIRE(rb.try_pop(v)); REQUIRE(v == 10);
    REQUIRE(rb.try_pop(v)); REQUIRE(v == 20);
    REQUIRE(rb.try_pop(v)); REQUIRE(v == 30);

    REQUIRE(rb.try_push(40));
    REQUIRE(rb.try_push(50));
    REQUIRE(rb.try_push(60));

    REQUIRE(rb.try_pop(v)); REQUIRE(v == 40);
    REQUIRE(rb.try_pop(v)); REQUIRE(v == 50);
    REQUIRE(rb.try_pop(v)); REQUIRE(v == 60);
    REQUIRE(rb.empty());
}

TEST_CASE("RingBuffer - interleaved push/pop at capacity", "[ring]") {
    SPSCRingBuffer<uint32_t, 8> rb;
    // Fill it
    for (uint32_t i = 0; i < 8; ++i) REQUIRE(rb.try_push(i));
    REQUIRE(rb.full());

    // Pop one, push one — must succeed repeatedly
    for (uint32_t i = 8; i < 64; ++i) {
        uint32_t v = 0;
        REQUIRE(rb.try_pop(v));
        REQUIRE(v == i - 8);
        REQUIRE(rb.try_push(i));
    }
}

TEST_CASE("RingBuffer - works with struct types", "[ring]") {
    struct Msg { uint64_t id; uint32_t price; };
    SPSCRingBuffer<Msg, 16> rb;
    REQUIRE(rb.try_push({1, 100}));
    REQUIRE(rb.try_push({2, 200}));

    Msg m{};
    REQUIRE(rb.try_pop(m)); REQUIRE(m.id == 1); REQUIRE(m.price == 100);
    REQUIRE(rb.try_pop(m)); REQUIRE(m.id == 2); REQUIRE(m.price == 200);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SPSCRingBuffer — concurrent producer/consumer
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("RingBuffer - concurrent producer consumer", "[ring][threaded]") {
    static constexpr size_t N = 100'000;
    SPSCRingBuffer<uint64_t, 256> rb;

    std::vector<uint64_t> received;
    received.reserve(N);

    // Consumer thread: drain until it has seen N items
    std::thread consumer([&]() {
        uint64_t count = 0;
        uint64_t v = 0;
        while (count < N) {
            if (rb.try_pop(v)) {
                received.push_back(v);
                ++count;
            }
        }
    });

    // Producer: push 0..N-1
    for (uint64_t i = 0; i < N; ++i) {
        while (!rb.try_push(i)) { /* spin if full */ }
    }

    consumer.join();

    REQUIRE(received.size() == N);
    for (uint64_t i = 0; i < N; ++i)
        REQUIRE(received[i] == i);
}

TEST_CASE("RingBuffer - no items lost under concurrent pressure", "[ring][threaded]") {
    static constexpr size_t N = 50'000;
    SPSCRingBuffer<uint32_t, 512> rb;

    std::atomic<uint64_t> sum_sent{0}, sum_recv{0};

    std::thread producer([&]() {
        for (uint32_t i = 1; i <= N; ++i) {
            while (!rb.try_push(i)) {}
            sum_sent.fetch_add(i, std::memory_order_relaxed);
        }
    });

    std::thread consumer([&]() {
        uint32_t count = 0;
        uint32_t v = 0;
        while (count < N) {
            if (rb.try_pop(v)) {
                sum_recv.fetch_add(v, std::memory_order_relaxed);
                ++count;
            }
        }
    });

    producer.join();
    consumer.join();

    // Sum of 1..N = N*(N+1)/2
    const uint64_t expected = static_cast<uint64_t>(N) * (N + 1) / 2;
    REQUIRE(sum_sent.load() == expected);
    REQUIRE(sum_recv.load() == expected);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  FeedHandler — ITCH message processing
// ═══════════════════════════════════════════════════════════════════════════════

// ── Buffer helpers (shared with test_itch_parser.cpp logic) ──────────────────
static void fh_put16(uint8_t* p, uint16_t v) {
    p[0] = uint8_t(v >> 8); p[1] = uint8_t(v);
}
static void fh_put32(uint8_t* p, uint32_t v) {
    p[0] = uint8_t(v>>24); p[1] = uint8_t(v>>16);
    p[2] = uint8_t(v>>8);  p[3] = uint8_t(v);
}
static void fh_put64(uint8_t* p, uint64_t v) {
    for (int i=7;i>=0;--i){p[i]=uint8_t(v&0xff);v>>=8;}
}
static void fh_put48(uint8_t* p, uint64_t v) {
    p[0]=uint8_t((v>>40)&0xff); p[1]=uint8_t((v>>32)&0xff);
    p[2]=uint8_t((v>>24)&0xff); p[3]=uint8_t((v>>16)&0xff);
    p[4]=uint8_t((v>>8)&0xff);  p[5]=uint8_t(v&0xff);
}
static void fh_put_stock(uint8_t* p, const char* s) {
    std::memset(p, ' ', 8);
    size_t n = std::strlen(s); if (n>8) n=8;
    std::memcpy(p, s, n);
}
static void fh_hdr(uint8_t* b, uint8_t t, uint16_t loc, uint64_t ts) {
    b[0]=t; fh_put16(b+1,loc); fh_put16(b+3,0); fh_put48(b+5,ts);
}

// Build decoded AddOrder from raw bytes
static AddOrder make_ao(uint64_t ref, char side, uint32_t qty,
                         const char* sym, uint32_t price, uint64_t ts=0)
{
    uint8_t buf[36]={};
    fh_hdr(buf,'A',1,ts);
    fh_put64(buf+11,ref); buf[19]=uint8_t(side);
    fh_put32(buf+20,qty); fh_put_stock(buf+24,sym);
    fh_put32(buf+32,price);
    return std::get<AddOrder>(*ItchParser::parse(buf,36));
}
static OrderExecuted make_oe(uint64_t ref, uint32_t qty, uint64_t ts=0) {
    uint8_t buf[31]={};
    fh_hdr(buf,'E',1,ts);
    fh_put64(buf+11,ref); fh_put32(buf+19,qty); fh_put64(buf+23,0);
    return std::get<OrderExecuted>(*ItchParser::parse(buf,31));
}
static OrderCancel make_oc(uint64_t ref, uint32_t qty, uint64_t ts=0) {
    uint8_t buf[23]={};
    fh_hdr(buf,'X',1,ts);
    fh_put64(buf+11,ref); fh_put32(buf+19,qty);
    return std::get<OrderCancel>(*ItchParser::parse(buf,23));
}
static OrderDelete make_od(uint64_t ref, uint64_t ts=0) {
    uint8_t buf[19]={};
    fh_hdr(buf,'D',1,ts);
    fh_put64(buf+11,ref);
    return std::get<OrderDelete>(*ItchParser::parse(buf,19));
}
static OrderReplace make_or(uint64_t old_ref, uint64_t new_ref,
                             uint32_t qty, uint32_t price, uint64_t ts=0) {
    uint8_t buf[35]={};
    fh_hdr(buf,'U',1,ts);
    fh_put64(buf+11,old_ref); fh_put64(buf+19,new_ref);
    fh_put32(buf+27,qty); fh_put32(buf+31,price);
    return std::get<OrderReplace>(*ItchParser::parse(buf,35));
}
static OrderExecutedPrice make_oep(uint64_t ref, uint32_t qty,
                                    uint32_t exec_price, uint64_t ts=0) {
    uint8_t buf[36]={};
    fh_hdr(buf,'C',1,ts);
    fh_put64(buf+11,ref); fh_put32(buf+19,qty); fh_put64(buf+23,0);
    buf[31]='Y'; fh_put32(buf+32,exec_price);
    return std::get<OrderExecutedPrice>(*ItchParser::parse(buf,36));
}

// ── FeedHandler tests ─────────────────────────────────────────────────────────

TEST_CASE("FeedHandler - AddOrder updates LOB and fires TOB callback", "[fh]") {
    LimitOrderBook lob;
    std::vector<TopOfBookUpdate> tobs;

    FeedHandler fh(lob, [&](const MarketEvent& ev) {
        if (std::holds_alternative<TopOfBookUpdate>(ev))
            tobs.push_back(std::get<TopOfBookUpdate>(ev));
    });

    fh.process(make_ao(1, 'B', 100, "AAPL", 1500000u, 111));

    REQUIRE(lob.total_orders() == 1);
    REQUIRE(lob.best_bid()     == 1500000u);
    REQUIRE(tobs.size()        == 1);
    REQUIRE(tobs[0].bid_price  == 1500000u);
    REQUIRE(tobs[0].ask_price  == 0u);
    REQUIRE(tobs[0].timestamp_ns == 111u);
    REQUIRE(fh.messages_processed() == 1);
    REQUIRE(fh.tob_updates()        == 1);
}

TEST_CASE("FeedHandler - duplicate TOB does not fire second callback", "[fh]") {
    LimitOrderBook lob;
    size_t cb_count = 0;

    FeedHandler fh(lob, [&](const MarketEvent&) { ++cb_count; });

    // Two bids at the same price → TOB unchanged after second add
    fh.process(make_ao(1, 'B', 100, "AAPL", 1500000u));
    fh.process(make_ao(2, 'B', 200, "AAPL", 1500000u));  // same best bid

    REQUIRE(cb_count == 1);  // only first add changes TOB
}

TEST_CASE("FeedHandler - AddOrder bid + ask fires two TOB updates", "[fh]") {
    LimitOrderBook lob;
    std::vector<TopOfBookUpdate> tobs;
    FeedHandler fh(lob, [&](const MarketEvent& ev) {
        if (auto* t = std::get_if<TopOfBookUpdate>(&ev)) tobs.push_back(*t);
    });

    fh.process(make_ao(1, 'B', 100, "AAPL", 1500000u));
    fh.process(make_ao(2, 'S', 100, "AAPL", 1510000u));

    REQUIRE(tobs.size() == 2);
    REQUIRE(tobs[1].bid_price == 1500000u);
    REQUIRE(tobs[1].ask_price == 1510000u);
    REQUIRE(lob.spread()      == 10000u);
}

TEST_CASE("FeedHandler - OrderExecuted (E) reduces LOB qty and fires trade", "[fh]") {
    LimitOrderBook lob;
    std::vector<TradeEvent> trades;
    FeedHandler fh(lob, [&](const MarketEvent& ev) {
        if (auto* t = std::get_if<TradeEvent>(&ev)) trades.push_back(*t);
    });

    fh.process(make_ao(10, 'B', 500, "NVDA", 5000000u));
    fh.process(make_oe(10, 200));  // partial fill

    REQUIRE(trades.size()     == 1);
    REQUIRE(trades[0].shares  == 200u);
    REQUIRE(lob.total_orders() == 1);   // still in book (300 remaining)
    REQUIRE(fh.trade_events()  == 1);
}

TEST_CASE("FeedHandler - OrderExecuted full fill removes order", "[fh]") {
    LimitOrderBook lob;
    FeedHandler fh(lob);

    fh.process(make_ao(5, 'B', 100, "SPY", 4000000u));
    fh.process(make_oe(5, 100));  // full fill

    REQUIRE(lob.total_orders() == 0);
    REQUIRE(!lob.best_bid().has_value());
}

TEST_CASE("FeedHandler - OrderExecutedPrice (C) emits trade with price", "[fh]") {
    LimitOrderBook lob;
    std::vector<TradeEvent> trades;
    FeedHandler fh(lob, [&](const MarketEvent& ev) {
        if (auto* t = std::get_if<TradeEvent>(&ev)) trades.push_back(*t);
    });

    fh.process(make_ao(7, 'S', 200, "QQQ", 3500000u));
    fh.process(make_oep(7, 100, 3490000u));  // exec at different price

    REQUIRE(trades.size()         == 1);
    REQUIRE(trades[0].price       == 3490000u);
    REQUIRE(trades[0].shares      == 100u);
    REQUIRE(lob.total_orders()    == 1);   // 100 remaining
}

TEST_CASE("FeedHandler - OrderCancel (X) partially reduces qty", "[fh]") {
    LimitOrderBook lob;
    FeedHandler fh(lob);

    fh.process(make_ao(3, 'B', 1000, "TSLA", 2000000u));
    fh.process(make_oc(3, 400));  // cancel 400 of 1000

    // 600 shares should remain
    REQUIRE(lob.total_orders() == 1);
    const auto bids = lob.bid_depth(1);
    REQUIRE(bids[0].quantity   == 600u);
}

TEST_CASE("FeedHandler - OrderCancel full quantity removes order", "[fh]") {
    LimitOrderBook lob;
    FeedHandler fh(lob);

    fh.process(make_ao(4, 'B', 200, "AMZN", 1800000u));
    fh.process(make_oc(4, 200));

    REQUIRE(lob.total_orders() == 0);
    REQUIRE(!lob.best_bid().has_value());
}

TEST_CASE("FeedHandler - OrderDelete (D) removes entire order", "[fh]") {
    LimitOrderBook lob;
    std::vector<TopOfBookUpdate> tobs;
    FeedHandler fh(lob, [&](const MarketEvent& ev) {
        if (auto* t = std::get_if<TopOfBookUpdate>(&ev)) tobs.push_back(*t);
    });

    fh.process(make_ao(8, 'B', 300, "META", 3000000u));
    fh.process(make_od(8));

    REQUIRE(lob.total_orders() == 0);
    REQUIRE(tobs.back().bid_price == 0u);  // TOB cleared
}

TEST_CASE("FeedHandler - OrderReplace changes price in LOB", "[fh]") {
    LimitOrderBook lob;
    FeedHandler fh(lob);

    fh.process(make_ao(100, 'B', 500, "GOOG", 1400000u));
    fh.process(make_or(100, 101, 500, 1405000u));  // replace to better price

    // old order gone, new order at 1405000
    REQUIRE(lob.total_orders() == 1);
    REQUIRE(lob.best_bid()     == 1405000u);
    REQUIRE(!lob.cancel_order(100));  // old id no longer in book
    REQUIRE(lob.cancel_order(101));   // new id present
}

TEST_CASE("FeedHandler - OrderReplace on unknown id increments error count", "[fh]") {
    LimitOrderBook lob;
    FeedHandler fh(lob);

    fh.process(make_or(999, 1000, 100, 1000000u));
    REQUIRE(fh.lob_errors() == 1);
}

TEST_CASE("FeedHandler - informational messages do not mutate LOB", "[fh]") {
    LimitOrderBook lob;
    size_t cb_count = 0;
    FeedHandler fh(lob, [&](const MarketEvent&) { ++cb_count; });

    // System Event, NOII, RPII — none should touch the LOB or fire callbacks
    uint8_t sys_buf[12]={};
    fh_hdr(sys_buf,'S',0,0); sys_buf[11]='O';
    fh.process(*ItchParser::parse(sys_buf,12));

    uint8_t noii_buf[50]={};
    fh_hdr(noii_buf,'I',0,0);
    fh.process(*ItchParser::parse(noii_buf,50));

    REQUIRE(lob.empty());
    REQUIRE(cb_count == 0);
    REQUIRE(fh.messages_processed() == 2);
}

TEST_CASE("FeedHandler - no callback registered does not crash", "[fh]") {
    LimitOrderBook lob;
    FeedHandler fh(lob);  // no callback
    fh.process(make_ao(1, 'B', 100, "AAPL", 1500000u));
    fh.process(make_oe(1, 100));
    REQUIRE(lob.empty());
}

TEST_CASE("FeedHandler - stats reset works", "[fh]") {
    LimitOrderBook lob;
    FeedHandler fh(lob);
    fh.process(make_ao(1, 'B', 100, "AAPL", 1500000u));
    REQUIRE(fh.messages_processed() == 1);
    fh.reset_stats();
    REQUIRE(fh.messages_processed() == 0);
    REQUIRE(fh.tob_updates()        == 0);
}

TEST_CASE("FeedHandler - stress: 500 adds, 250 cancels, 250 deletes", "[fh][stress]") {
    LimitOrderBook lob;
    FeedHandler fh(lob);

    for (uint64_t i = 1; i <= 500; ++i)
        fh.process(make_ao(i, (i%2==0)?'B':'S', 100, "AAPL",
                            (i%2==0) ? 1490000u + uint32_t(i) * 10u
                                     : 1510000u + uint32_t(i) * 10u));

    REQUIRE(lob.total_orders() == 500);

    // Partially cancel 250 orders (halve their qty)
    for (uint64_t i = 1; i <= 250; ++i)
        fh.process(make_oc(i, 50));

    // Delete remaining 250
    for (uint64_t i = 251; i <= 500; ++i)
        fh.process(make_od(i));

    REQUIRE(lob.total_orders() == 250); // the partially-cancelled ones remain

    // Execute the remainder
    for (uint64_t i = 1; i <= 250; ++i)
        fh.process(make_oe(i, 50));   // consume the remaining 50

    REQUIRE(lob.total_orders() == 0);
    REQUIRE(fh.messages_processed() == 500 + 250 + 250 + 250);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  FeedHandler + SPSCRingBuffer: end-to-end event pipeline
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("FeedHandler feeds MarketEvents into ring buffer", "[fh][ring]") {
    LimitOrderBook lob;
    SPSCRingBuffer<MarketEvent, 64> ring;

    FeedHandler fh(lob, [&](const MarketEvent& ev) {
        ring.try_push(ev);   // producer
    });

    fh.process(make_ao(1, 'B', 100, "AAPL", 1500000u));  // TOB: bid appears
    fh.process(make_ao(2, 'S', 100, "AAPL", 1510000u));  // TOB: ask appears
    fh.process(make_oe(1, 100));                          // TradeEvent + TOB: bid disappears

    // Consumer: drain ring
    std::vector<MarketEvent> events;
    MarketEvent ev;
    while (ring.try_pop(ev)) events.push_back(ev);

    // Events in order:
    //   [0] TopOfBookUpdate — bid added (1500000, ask=0)
    //   [1] TopOfBookUpdate — ask added (bid=1500000, ask=1510000)
    //   [2] TradeEvent      — order 1 fully executed
    //   [3] TopOfBookUpdate — bid cleared after full fill (bid=0, ask=1510000)
    REQUIRE(events.size() == 4);
    REQUIRE(std::holds_alternative<TopOfBookUpdate>(events[0]));
    REQUIRE(std::holds_alternative<TopOfBookUpdate>(events[1]));
    REQUIRE(std::holds_alternative<TradeEvent>(events[2]));
    REQUIRE(std::holds_alternative<TopOfBookUpdate>(events[3]));

    const auto& tob0 = std::get<TopOfBookUpdate>(events[0]);
    REQUIRE(tob0.bid_price == 1500000u);
    REQUIRE(tob0.ask_price == 0u);

    const auto& tob1 = std::get<TopOfBookUpdate>(events[1]);
    REQUIRE(tob1.bid_price == 1500000u);
    REQUIRE(tob1.ask_price == 1510000u);

    const auto& trade = std::get<TradeEvent>(events[2]);
    REQUIRE(trade.shares == 100u);

    const auto& tob3 = std::get<TopOfBookUpdate>(events[3]);
    REQUIRE(tob3.bid_price == 0u);       // bid cleared
    REQUIRE(tob3.ask_price == 1510000u); // ask still standing
}

TEST_CASE("FeedHandler ring overflow is handled gracefully", "[fh][ring]") {
    // Ring capacity = 4; push 10 events. try_push silently fails when full.
    LimitOrderBook lob;
    SPSCRingBuffer<MarketEvent, 4> ring;

    FeedHandler fh(lob, [&](const MarketEvent& ev) {
        ring.try_push(ev);
    });

    // 10 different prices → 10 TOB updates
    for (uint64_t i = 1; i <= 10; ++i)
        fh.process(make_ao(i, 'B', 100, "AAPL",
                            static_cast<uint32_t>(1500000u + i * 1000u)));

    // Ring should be at capacity (4) — events beyond 4 were dropped
    REQUIRE(ring.size() == 4);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  ShmWriter — seqlock snapshot correctness
// ═══════════════════════════════════════════════════════════════════════════════

// Allocate an aligned buffer on the stack/heap large enough for ShmLayout.
static constexpr size_t SHM_BUF_SIZE = sizeof(ShmLayout) + 64;
alignas(64) static uint8_t g_shm_buf[SHM_BUF_SIZE];

TEST_CASE("ShmWriter - write_snapshot stores magic and version", "[shm]") {
    std::memset(g_shm_buf, 0, SHM_BUF_SIZE);
    ShmWriter writer(g_shm_buf, SHM_BUF_SIZE);

    LimitOrderBook lob;
    writer.write_snapshot(lob, 999ULL);

    const ShmLayout* layout = writer.layout();
    REQUIRE(layout->header.magic   == kShmMagic);
    REQUIRE(layout->header.version == kShmVersion);
    REQUIRE(layout->header.seqlock.load() % 2 == 0);  // stable (even)
}

TEST_CASE("ShmWriter - empty LOB produces zero bid/ask counts", "[shm]") {
    std::memset(g_shm_buf, 0, SHM_BUF_SIZE);
    ShmWriter writer(g_shm_buf, SHM_BUF_SIZE);

    LimitOrderBook lob;
    writer.write_snapshot(lob, 0ULL);

    REQUIRE(writer.layout()->bid_count == 0);
    REQUIRE(writer.layout()->ask_count == 0);
}

TEST_CASE("ShmWriter - snapshot reflects LOB depth correctly", "[shm]") {
    std::memset(g_shm_buf, 0, SHM_BUF_SIZE);
    ShmWriter writer(g_shm_buf, SHM_BUF_SIZE);

    LimitOrderBook lob;
    // Add 3 bid levels and 2 ask levels
    auto add = [&](uint64_t id, Side s, uint64_t price, uint32_t qty) {
        lob.add_order({id, price, qty, s, 0});
    };
    add(1, Side::BID, 1500000, 100);
    add(2, Side::BID, 1490000, 200);
    add(3, Side::BID, 1480000, 300);
    add(4, Side::ASK, 1510000, 150);
    add(5, Side::ASK, 1520000, 250);

    writer.write_snapshot(lob, 12345ULL);
    const ShmLayout* lay = writer.layout();

    REQUIRE(lay->timestamp_ns == 12345ULL);
    REQUIRE(lay->bid_count    == 3u);
    REQUIRE(lay->ask_count    == 2u);

    // Bids descending
    REQUIRE(lay->bids[0].price == 1500000u);
    REQUIRE(lay->bids[1].price == 1490000u);
    REQUIRE(lay->bids[2].price == 1480000u);

    // Asks ascending
    REQUIRE(lay->asks[0].price == 1510000u);
    REQUIRE(lay->asks[1].price == 1520000u);
}

TEST_CASE("ShmWriter - seqlock even after write", "[shm]") {
    std::memset(g_shm_buf, 0, SHM_BUF_SIZE);
    ShmWriter writer(g_shm_buf, SHM_BUF_SIZE);

    LimitOrderBook lob;
    lob.add_order({1, 1500000, 100, Side::BID, 0});

    writer.write_snapshot(lob, 0ULL);
    REQUIRE(writer.layout()->header.seqlock.load() == 2u); // two fetch_adds
    writer.write_snapshot(lob, 0ULL);
    REQUIRE(writer.layout()->header.seqlock.load() == 4u);
}

TEST_CASE("ShmWriter - shm_read_snapshot returns consistent data", "[shm]") {
    std::memset(g_shm_buf, 0, SHM_BUF_SIZE);
    ShmWriter writer(g_shm_buf, SHM_BUF_SIZE);

    LimitOrderBook lob;
    lob.add_order({1, 1500000, 100, Side::BID, 0});
    lob.add_order({2, 1510000, 200, Side::ASK, 0});

    writer.write_snapshot(lob, 777ULL);

    ShmSnapshot snap{};
    REQUIRE(shm_read_snapshot(writer.layout(), snap));
    REQUIRE(snap.timestamp_ns == 777ULL);
    REQUIRE(snap.bid_count    == 1u);
    REQUIRE(snap.ask_count    == 1u);
    REQUIRE(snap.bids[0].price == 1500000u);
    REQUIRE(snap.asks[0].price == 1510000u);
}

TEST_CASE("ShmWriter - shm_read_snapshot returns false on null layout", "[shm]") {
    ShmSnapshot snap{};
    REQUIRE(!shm_read_snapshot(nullptr, snap));
}

TEST_CASE("ShmWriter - buffer too small throws", "[shm]") {
    uint8_t tiny[10];
    bool threw = false;
    try { ShmWriter w(tiny, sizeof(tiny)); }
    catch (const std::invalid_argument&) { threw = true; }
    REQUIRE(threw);
}

TEST_CASE("ShmWriter - multiple snapshots overwrite correctly", "[shm]") {
    std::memset(g_shm_buf, 0, SHM_BUF_SIZE);
    ShmWriter writer(g_shm_buf, SHM_BUF_SIZE);

    LimitOrderBook lob;
    lob.add_order({1, 1500000, 100, Side::BID, 0});
    writer.write_snapshot(lob, 1ULL);

    lob.cancel_order(1);
    lob.add_order({2, 1490000, 200, Side::BID, 0});
    writer.write_snapshot(lob, 2ULL);

    ShmSnapshot snap{};
    shm_read_snapshot(writer.layout(), snap);
    REQUIRE(snap.timestamp_ns    == 2ULL);
    REQUIRE(snap.bids[0].price   == 1490000u);
    REQUIRE(snap.bids[0].quantity== 200u);
}
