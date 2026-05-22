CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wpedantic -I.

# ── LOB library sources ───────────────────────────────────────────────────────
LOB_SRCS := core/lob/price_level.cpp \
            core/lob/limit_order_book.cpp

# ── Feed handler sources ──────────────────────────────────────────────────────
FEED_SRCS := core/feed_handler/itch_parser.cpp \
             core/feed_handler/pcap_replayer.cpp \
             core/feed_handler/feed_handler.cpp

# ── Shared memory sources ─────────────────────────────────────────────────────
SHM_SRCS  := core/shared_memory/shm_writer.cpp

# ── Event bus + execution sources ────────────────────────────────────────────
BUS_SRCS  := core/event_bus/event_bus.cpp
EXEC_SRCS := core/execution/order_router.cpp \
             core/execution/fill_simulator.cpp

# ── All shared library sources ────────────────────────────────────────────────
LIB_SRCS := $(LOB_SRCS) $(FEED_SRCS) $(SHM_SRCS) $(BUS_SRCS) $(EXEC_SRCS)

PYTEST := $(shell command -v pytest 2>/dev/null || echo python -m pytest)

.PHONY: all test test-cpp test-py clean

all: build/test_lob build/test_itch_parser build/test_ring_buffer build/test_event_bus

# ── LOB tests ─────────────────────────────────────────────────────────────────
build/test_lob: $(LOB_SRCS) tests/cpp/test_lob.cpp | build
	$(CXX) $(CXXFLAGS) $(LOB_SRCS) tests/cpp/test_lob.cpp -o $@

# ── ITCH parser + pcap replayer tests ────────────────────────────────────────
build/test_itch_parser: $(LOB_SRCS) $(FEED_SRCS) tests/cpp/test_itch_parser.cpp | build
	$(CXX) $(CXXFLAGS) $(LOB_SRCS) $(FEED_SRCS) tests/cpp/test_itch_parser.cpp -o $@

# ── Ring buffer, FeedHandler, ShmWriter tests ─────────────────────────────────
build/test_ring_buffer: $(LOB_SRCS) $(FEED_SRCS) $(SHM_SRCS) tests/cpp/test_ring_buffer.cpp | build
	$(CXX) $(CXXFLAGS) $(LOB_SRCS) $(FEED_SRCS) $(SHM_SRCS) \
	    tests/cpp/test_ring_buffer.cpp -o $@ -lpthread

# ── Event bus, OrderRouter, FillSimulator tests ───────────────────────────────
build/test_event_bus: $(LIB_SRCS) tests/cpp/test_event_bus.cpp | build
	$(CXX) $(CXXFLAGS) $(LIB_SRCS) tests/cpp/test_event_bus.cpp -o $@

build:
	mkdir -p build

test: test-cpp test-py

test-cpp: all
	@echo "=== LOB tests ==="
	./build/test_lob
	@echo ""
	@echo "=== ITCH parser tests ==="
	./build/test_itch_parser
	@echo ""
	@echo "=== Ring buffer / FeedHandler / ShmWriter tests ==="
	./build/test_ring_buffer
	@echo ""
	@echo "=== EventBus / OrderRouter / FillSimulator tests ==="
	./build/test_event_bus

test-py:
	@echo ""
	@echo "=== Python microstructure tests ==="
	$(PYTEST) tests/python/ -q

clean:
	rm -rf build __pycache__ .pytest_cache
	find python tests/python -name '__pycache__' -type d | xargs rm -rf 2>/dev/null || true
