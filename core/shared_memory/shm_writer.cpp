#include "shm_writer.hpp"
#include <cstring>
#include <stdexcept>
#include <algorithm>

// POSIX shm headers — only needed for the production constructor.
#if __has_include(<sys/mman.h>)
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <fcntl.h>
#  include <unistd.h>
#  define SIGMA_EDGE_HAVE_POSIX_SHM 1
#endif

namespace sigma_edge {

// ── In-memory constructor (tests / embedded use) ──────────────────────────────

ShmWriter::ShmWriter(void* buf, size_t buf_size, size_t depth)
    : depth_(std::min(depth, kShmMaxDepth)), owns_shm_(false)
{
    if (!buf || buf_size < kShmLayoutSize)
        throw std::invalid_argument(
            "ShmWriter: buffer too small (need " +
            std::to_string(kShmLayoutSize) + " bytes)");

    layout_ = new (buf) ShmLayout{};   // placement-new to zero-init atomics
    layout_->header.magic   = kShmMagic;
    layout_->header.version = kShmVersion;
    layout_->header.seqlock.store(0, std::memory_order_relaxed);
}

// ── POSIX shm constructor (production) ────────────────────────────────────────

ShmWriter::ShmWriter(const std::string& shm_name, size_t depth)
    : depth_(std::min(depth, kShmMaxDepth)),
      owns_shm_(true),
      shm_name_(shm_name),
      region_size_(kShmLayoutSize)
{
#if defined(SIGMA_EDGE_HAVE_POSIX_SHM)
    shm_fd_ = ::shm_open(shm_name.c_str(),
                          O_CREAT | O_RDWR,
                          S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (shm_fd_ == -1)
        throw std::runtime_error("ShmWriter: shm_open failed: " + shm_name);

    if (::ftruncate(shm_fd_, static_cast<off_t>(region_size_)) == -1) {
        ::close(shm_fd_);
        ::shm_unlink(shm_name.c_str());
        throw std::runtime_error("ShmWriter: ftruncate failed: " + shm_name);
    }

    void* ptr = ::mmap(nullptr, region_size_,
                       PROT_READ | PROT_WRITE, MAP_SHARED,
                       shm_fd_, 0);
    if (ptr == MAP_FAILED) {
        ::close(shm_fd_);
        ::shm_unlink(shm_name.c_str());
        throw std::runtime_error("ShmWriter: mmap failed: " + shm_name);
    }

    layout_ = new (ptr) ShmLayout{};
    layout_->header.magic   = kShmMagic;
    layout_->header.version = kShmVersion;
    layout_->header.seqlock.store(0, std::memory_order_relaxed);
#else
    throw std::runtime_error(
        "ShmWriter: POSIX shared memory not available on this platform");
#endif
}

// ── Destructor ────────────────────────────────────────────────────────────────

ShmWriter::~ShmWriter() {
#if defined(SIGMA_EDGE_HAVE_POSIX_SHM)
    if (owns_shm_ && layout_) {
        ::munmap(layout_, region_size_);
        layout_ = nullptr;
    }
    if (owns_shm_ && shm_fd_ != -1) {
        ::close(shm_fd_);
        ::shm_unlink(shm_name_.c_str());
    }
#endif
}

// ── write_snapshot ────────────────────────────────────────────────────────────

void ShmWriter::write_snapshot(const LimitOrderBook& lob, uint64_t ts_ns) {
    if (!layout_) return;

    auto& sl = layout_->header.seqlock;

    // Begin write: increment to odd → readers will spin.
    sl.fetch_add(1, std::memory_order_release);

    // Compiler and CPU fence — all prior writes complete before any slot write.
    std::atomic_thread_fence(std::memory_order_seq_cst);

    const auto bids = lob.bid_depth(depth_);
    const auto asks = lob.ask_depth(depth_);

    layout_->timestamp_ns = ts_ns;
    layout_->bid_count    = static_cast<uint32_t>(bids.size());
    layout_->ask_count    = static_cast<uint32_t>(asks.size());

    for (size_t i = 0; i < bids.size(); ++i) layout_->bids[i] = bids[i];
    for (size_t i = 0; i < asks.size(); ++i) layout_->asks[i] = asks[i];

    // End write: increment to even → snapshot is now stable for readers.
    sl.fetch_add(1, std::memory_order_release);
}

} // namespace sigma_edge
