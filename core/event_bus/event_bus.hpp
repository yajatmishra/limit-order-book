#pragma once
// TypedEventBus<EventTypes...>
//
// Compile-time, type-safe synchronous pub/sub for a fixed set of event types.
//
// Implementation notes:
//   • Each event type T gets its own handler vector via a tagged Slot<T> wrapper
//     inside a std::tuple<Slot<EventTypes>...>.  std::get<Slot<T>>(tuple_)
//     selects the right vector at zero run-time cost.
//   • Handlers are type-erased to std::function<void(const void*)> so the
//     tuple only needs one Slot template, keeping binary size compact.
//   • publish() uses std::visit to dispatch the variant to fire(); the correct
//     Slot<T> is selected at compile time inside the lambda.
//   • NOT thread-safe; intended for single-threaded event loops.

#include "event_types.hpp"
#include <functional>
#include <tuple>
#include <vector>
#include <variant>
#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace sigma_edge {

// ── SubscriptionHandle ────────────────────────────────────────────────────────
struct SubscriptionHandle {
    uint64_t id{0};
    bool valid() const noexcept { return id != 0; }
    explicit operator bool() const noexcept { return valid(); }
};
inline bool operator==(SubscriptionHandle a, SubscriptionHandle b) noexcept {
    return a.id == b.id;
}

namespace detail {

// ── type-erased handler entry ─────────────────────────────────────────────────
struct Entry {
    uint64_t                         id;
    std::function<void(const void*)> fn;
};

// ── Slot<T>: per-type handler vector, tagged to survive tuple lookup ──────────
template<typename T>
struct Slot {
    std::vector<Entry> entries;
};

} // namespace detail

// ── TypedEventBus ─────────────────────────────────────────────────────────────

template<typename... EventTypes>
class TypedEventBus {
public:
    using EventVariant = std::variant<EventTypes...>;

    TypedEventBus() = default;
    TypedEventBus(const TypedEventBus&) = delete;
    TypedEventBus& operator=(const TypedEventBus&) = delete;
    TypedEventBus(TypedEventBus&&) = default;

    // ── subscribe ─────────────────────────────────────────────────────────────
    template<typename T>
    SubscriptionHandle subscribe(std::function<void(const T&)> handler) {
        const uint64_t hid = ++next_id_;
        slot<T>().entries.push_back({hid,
            [h = std::move(handler)](const void* p) {
                h(*static_cast<const T*>(p));
            }
        });
        return SubscriptionHandle{hid};
    }

    template<typename T, typename Callable>
    SubscriptionHandle subscribe(Callable&& c) {
        return subscribe<T>(
            std::function<void(const T&)>(std::forward<Callable>(c)));
    }

    // ── unsubscribe ───────────────────────────────────────────────────────────
    void unsubscribe(SubscriptionHandle h) {
        if (!h) return;
        remove_from_all(h.id, std::index_sequence_for<EventTypes...>{});
    }

    // ── publish (variant) ─────────────────────────────────────────────────────
    void publish(const EventVariant& ev) {
        std::visit([this](const auto& m) { fire(m); }, ev);
    }

    // ── publish (direct, concrete type) ──────────────────────────────────────
    template<typename T,
             typename = std::enable_if_t<
                 !std::is_same_v<std::decay_t<T>, EventVariant>>>
    void publish(const T& ev) { fire(ev); }

    // ── query ─────────────────────────────────────────────────────────────────
    template<typename T>
    size_t handler_count() const noexcept {
        return slot<T>().entries.size();
    }

    size_t total_handler_count() const noexcept {
        size_t n = 0;
        count_all(n, std::index_sequence_for<EventTypes...>{});
        return n;
    }

    void clear() noexcept {
        clear_all(std::index_sequence_for<EventTypes...>{});
    }

private:
    // ── per-type slot storage ─────────────────────────────────────────────────
    // One Slot<T> per event type, selected at compile time via std::get<Slot<T>>.
    std::tuple<detail::Slot<EventTypes>...> slots_;
    uint64_t next_id_{0};

    template<typename T>
    detail::Slot<T>& slot() {
        return std::get<detail::Slot<T>>(slots_);
    }
    template<typename T>
    const detail::Slot<T>& slot() const {
        return std::get<detail::Slot<T>>(slots_);
    }

    // ── dispatch ──────────────────────────────────────────────────────────────
    template<typename T>
    void fire(const T& ev) {
        for (auto& entry : slot<T>().entries) entry.fn(&ev);
    }

    // ── index-sequence helpers ────────────────────────────────────────────────
    template<size_t... Is>
    void remove_from_all(uint64_t id, std::index_sequence<Is...>) {
        (remove_from(std::get<Is>(slots_).entries, id), ...);
    }
    static void remove_from(std::vector<detail::Entry>& v, uint64_t id) {
        v.erase(std::remove_if(v.begin(), v.end(),
                    [id](const detail::Entry& e){ return e.id == id; }),
                v.end());
    }

    template<size_t... Is>
    void count_all(size_t& n, std::index_sequence<Is...>) const {
        ((n += std::get<Is>(slots_).entries.size()), ...);
    }

    template<size_t... Is>
    void clear_all(std::index_sequence<Is...>) noexcept {
        (std::get<Is>(slots_).entries.clear(), ...);
    }
};

// ── SigmaEventBus ─────────────────────────────────────────────────────────────
using SigmaEventBus = TypedEventBus<SignalEvent, OrderEvent, FillEvent>;

// Global singleton (definition in event_bus.cpp).
SigmaEventBus& get_sigma_bus();

} // namespace sigma_edge
