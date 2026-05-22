#pragma once
// Seqlock-protected shared-memory snapshot writer.
//
// Layout written to the shared region (ShmLayout):
//
//   ┌─────────────────────────────────────────┐
//   │ ShmHeader  (magic, version, seqlock)    │  64 bytes
//   ├─────────────────────────────────────────┤
//   │ bid_count  (uint32)                     │
//   │ ask_count  (uint32)                     │
//   │ timestamp_ns (uint64)                   │
//   │ bids[MAX_DEPTH] DepthLevel×             │
//   │ asks[MAX_DEPTH] DepthLevel×             │
//   └─────────────────────────────────────────┘
//
// Seqlock protocol (readers and writers share this counter):
//   Writer: seqlock.fetch_add(1, release)   → counter becomes odd (write in progress)
//           write data
//           seqlock.fetch_add(1, release)   → counter becomes even (snapshot stable)
//   Reader: do {
//              seq = seqlock.load(acquire)
//              if (seq & 1) { spin_pause(); continue; }
//              // copy data
//           } while (seqlock.load(acquire) != seq);
//
// ShmWriter has two construction modes:
//   1. In-memory  : ShmWriter(void* buf, size_t buf_size)   ← used in tests
//   2. POSIX shm  : ShmWriter(const std::string& shm_name)  ← production

#include "../lob/limit_order_book.hpp"
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <stdexcept>

namespace lob {

// ── On-wire constants ─────────────────────────────────────────────────────────
inline constexpr uint64_t kShmMagic   = 0x5349474D45444745ULL; // "SIGMEDGE"
inline constexpr uint32_t kShmVersion = 1;
inline constexpr size_t   kShmMaxDepth = 10;

// ── ShmHeader (64 bytes, cache-line aligned) ──────────────────────────────────
struct alignas(64) ShmHeader {
    uint64_t             magic{0};
    uint32_t             version{0};
    uint32_t             _pad0{0};
    std::atomic<uint64_t> seqlock{0};   // even = stable, odd = write in progress
    uint8_t              _pad1[64 - sizeof(uint64_t) * 2
                                  - sizeof(uint32_t) * 2
                                  - sizeof(std::atomic<uint64_t>)]{};
};
static_assert(sizeof(ShmHeader) == 64, "ShmHeader must be 64 bytes");

// ── ShmLayout: the complete shared-memory region ──────────────────────────────
struct ShmLayout {
    ShmHeader header;
    uint64_t  timestamp_ns{0};
    uint32_t  bid_count{0};
    uint32_t  ask_count{0};
    DepthLevel bids[kShmMaxDepth]{};
    DepthLevel asks[kShmMaxDepth]{};
};

// Minimum buffer size required.
inline constexpr size_t kShmLayoutSize = sizeof(ShmLayout);

// ── ShmWriter ─────────────────────────────────────────────────────────────────
class ShmWriter {
public:
    // ── Test / in-memory constructor ──────────────────────────────────────────
    // buf must point to at least kShmLayoutSize bytes and remain valid for the
    // lifetime of this object.
    ShmWriter(void* buf, size_t buf_size, size_t depth = kShmMaxDepth);

    // ── Production POSIX shm constructor ─────────────────────────────────────
    // Creates or opens a POSIX shared memory object at /shm_name (e.g.
    // "/lob").  Throws std::runtime_error on failure.
    explicit ShmWriter(const std::string& shm_name,
                       size_t depth = kShmMaxDepth);

    ~ShmWriter();

    // Write a seqlock-protected snapshot of the top-N LOB levels.
    // Thread-safe with respect to concurrent ShmReader readers, but must only
    // be called from the single writer thread.
    void write_snapshot(const LimitOrderBook& lob, uint64_t ts_ns);

    // Pointer to the raw shared region (for test inspection).
    const ShmLayout* layout() const noexcept { return layout_; }

private:
    ShmLayout*  layout_{nullptr};
    size_t      depth_{kShmMaxDepth};
    bool        owns_shm_{false};   // true → destructor calls munmap+shm_unlink
    std::string shm_name_;
    int         shm_fd_{-1};
    size_t      region_size_{0};
};

// ── ShmReader: lock-free snapshot reader ─────────────────────────────────────
// Reads a consistent snapshot from the ShmLayout region.  Safe to call from
// any number of reader threads concurrently.
struct ShmSnapshot {
    uint64_t   timestamp_ns;
    uint32_t   bid_count;
    uint32_t   ask_count;
    DepthLevel bids[kShmMaxDepth];
    DepthLevel asks[kShmMaxDepth];
};

// Returns false if the region has an invalid magic / version.
inline bool shm_read_snapshot(const ShmLayout* layout, ShmSnapshot& out) noexcept {
    if (!layout) return false;
    while (true) {
        const uint64_t seq0 = layout->header.seqlock.load(std::memory_order_acquire);
        if (seq0 & 1) {   // write in progress — spin
#if defined(__x86_64__) || defined(__i386__)
            __asm__ volatile("pause" ::: "memory");
#endif
            continue;
        }
        out.timestamp_ns = layout->timestamp_ns;
        out.bid_count    = layout->bid_count;
        out.ask_count    = layout->ask_count;
        std::memcpy(out.bids, layout->bids, sizeof(DepthLevel) * kShmMaxDepth);
        std::memcpy(out.asks, layout->asks, sizeof(DepthLevel) * kShmMaxDepth);
        const uint64_t seq1 = layout->header.seqlock.load(std::memory_order_acquire);
        if (seq0 == seq1) break;   // no concurrent write; snapshot is stable
    }
    return (layout->header.magic == kShmMagic);
}

} // namespace lob
