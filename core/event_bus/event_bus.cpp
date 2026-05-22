#include "event_bus.hpp"

namespace sigma_edge {

// ── Global singleton SigmaEventBus ────────────────────────────────────────────
// Lazily initialised; lives for the duration of the process.
// In production, prefer constructing the bus on the stack of the main
// event loop and passing it by reference, to control lifetime explicitly.
SigmaEventBus& get_sigma_bus() {
    static SigmaEventBus instance;
    return instance;
}

} // namespace sigma_edge
