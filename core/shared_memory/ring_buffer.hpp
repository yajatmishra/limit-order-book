#pragma once
// Lock-free Single-Producer / Single-Consumer ring buffer.
//
// Design goals (HFT-grade):
//   • Zero dynamic allocation — slots live in the object itself.
//   • No locks, no syscalls on the hot path.
//   • Cache-line isolation between head_ (producer) and tail_ (consumer)
//     to eliminate false sharing on multi-core processors.
//   • Acquire / release memory ordering: the producer's store to head_
//     publishes the written slot; the consumer's store to tail_ releases
//     the consumed slot back.
//
// Constraints:
//   • Capacity must be a compile-time power of two (static_assert enforced).
//   • T must be trivially copyable for correct lock-free semantics.
//   • Only one thread may call try_push() and only one other thread may
//     call try_pop() concurrently.
//
// Usage:
//   SPSCRingBuffer<MarketEvent, 1024> ring;
//   // producer thread:
//   while (!ring.try_push(event)) { /* back-pressure */ }
//   // consumer thread:
//   MarketEvent ev;
//   if (ring.try_pop(ev)) { process(ev); }

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <new>   // std::hardware_destructive_interference_size (C++17)

namespace lob {

// Portable cache-line size constant.
#ifdef __cpp_lib_hardware_interference_size
    inline constexpr size_t kCacheLineSize =
        std::hardware_destructive_interference_size;
#else
    inline constexpr size_t kCacheLineSize = 64;
#endif

template<typename T, size_t Capacity>
class SPSCRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "SPSCRingBuffer: Capacity must be a power of two");
    static_assert(std::is_trivially_copyable_v<T>,
                  "SPSCRingBuffer: T must be trivially copyable");

public:
    SPSCRingBuffer() noexcept = default;

    // Not copyable or movable (atomics + fixed storage).
    SPSCRingBuffer(const SPSCRingBuffer&)            = delete;
    SPSCRingBuffer& operator=(const SPSCRingBuffer&) = delete;

    // ── Producer API (call from exactly one thread) ───────────────────────────

    // Try to push item.  Returns true on success, false if the buffer is full.
    bool try_push(const T& item) noexcept {
        const size_t head = head_.v.load(std::memory_order_relaxed);
        const size_t next = head + 1;
        // Check capacity: full when head - tail == Capacity.
        if (next - tail_.v.load(std::memory_order_acquire) > Capacity)
            return false;
        slots_[head & kMask] = item;
        head_.v.store(next, std::memory_order_release);
        return true;
    }

    // ── Consumer API (call from exactly one thread) ───────────────────────────

    // Try to pop into item.  Returns true on success, false if the buffer is empty.
    bool try_pop(T& item) noexcept {
        const size_t tail = tail_.v.load(std::memory_order_relaxed);
        if (head_.v.load(std::memory_order_acquire) == tail)
            return false;   // empty
        item = slots_[tail & kMask];
        tail_.v.store(tail + 1, std::memory_order_release);
        return true;
    }

    // ── Query API (safe to call from either thread) ───────────────────────────

    // Approximate size — may be stale by the time caller uses it.
    size_t size() const noexcept {
        const size_t h = head_.v.load(std::memory_order_acquire);
        const size_t t = tail_.v.load(std::memory_order_acquire);
        return h - t;
    }

    bool empty() const noexcept { return size() == 0; }
    bool full()  const noexcept { return size() >= Capacity; }

    static constexpr size_t capacity() noexcept { return Capacity; }

private:
    // Pad each atomic to its own cache line to prevent false sharing.
    struct alignas(kCacheLineSize) PaddedAtomic {
        std::atomic<size_t> v{0};
        // Fill the rest of the cache line with padding.
        char _pad[kCacheLineSize - sizeof(std::atomic<size_t>)];
        PaddedAtomic() noexcept : v{0} { }
    };

    static constexpr size_t kMask = Capacity - 1;

    PaddedAtomic head_{};               // written by producer
    PaddedAtomic tail_{};               // written by consumer
    T            slots_[Capacity]{};    // ring storage
};

} // namespace lob
