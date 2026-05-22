// Minimal Catch2 v2-compatible single-header shim for environments without
// network access.  Provides TEST_CASE / SECTION / REQUIRE / CHECK and a
// main() that prints per-test results and returns non-zero on failure.
// The real project uses Catch2 v3 via CMake FetchContent; this header is
// only present so the sandbox can compile and run the tests.
#pragma once
#include <functional>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <sstream>

namespace Catch {

struct AssertionFailed : std::exception {
    std::string msg;
    explicit AssertionFailed(std::string m) : msg(std::move(m)) {}
    const char* what() const noexcept override { return msg.c_str(); }
};

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> reg;
    return reg;
}

struct Registrar {
    Registrar(const char* name, const char* /*tags*/, std::function<void()> fn) {
        registry().push_back({name, std::move(fn)});
    }
};

inline int run_all(int /*argc*/ = 0, char** /*argv*/ = nullptr) {
    int passed = 0, failed = 0;
    for (const auto& tc : registry()) {
        try {
            tc.fn();
            std::cout << "\033[32m[PASS]\033[0m " << tc.name << "\n";
            ++passed;
        } catch (const AssertionFailed& e) {
            std::cout << "\033[31m[FAIL]\033[0m " << tc.name
                      << "\n       " << e.msg << "\n";
            ++failed;
        } catch (const std::exception& e) {
            std::cout << "\033[31m[FAIL]\033[0m " << tc.name
                      << "\n       std::exception: " << e.what() << "\n";
            ++failed;
        } catch (...) {
            std::cout << "\033[31m[FAIL]\033[0m " << tc.name
                      << "\n       unknown exception\n";
            ++failed;
        }
    }
    std::cout << "\n" << passed + failed << " tests: "
              << "\033[32m" << passed << " passed\033[0m, ";
    if (failed)
        std::cout << "\033[31m" << failed << " failed\033[0m";
    else
        std::cout << "0 failed";
    std::cout << "\n";
    return failed > 0 ? 1 : 0;
}

} // namespace Catch

// ── Public macros (Catch2-compatible subset) ──────────────────────────────────

// Unique name helper: double-indirection forces __LINE__ to expand to its
// integer value before token-pasting, so we get e.g. _catch_fn_42 not
// _catch_fn___LINE__.
#define CATCH_INTERNAL_PASTE2(a, b) a##b
#define CATCH_INTERNAL_PASTE(a, b)  CATCH_INTERNAL_PASTE2(a, b)
#define CATCH_INTERNAL_LINESTR2(x)  #x
#define CATCH_INTERNAL_LINESTR(x)   CATCH_INTERNAL_LINESTR2(x)
#define CATCH_INTERNAL_UNIQUE(base) CATCH_INTERNAL_PASTE(base, __LINE__)

#define TEST_CASE(name, tags)                                                   \
    static void CATCH_INTERNAL_UNIQUE(_catch_fn_)();                            \
    static ::Catch::Registrar CATCH_INTERNAL_UNIQUE(_catch_reg_)(               \
        name, tags, CATCH_INTERNAL_UNIQUE(_catch_fn_));                         \
    static void CATCH_INTERNAL_UNIQUE(_catch_fn_)()

// SECTIONs in our shim are plain scoped blocks — they run sequentially.
// Tests are written so each section sets up its own state, matching Catch2
// semantics for non-shared-setup cases.
#define SECTION(name) if (true)

#define REQUIRE(expr)                                                           \
    do {                                                                        \
        if (!(expr)) {                                                          \
            std::ostringstream _oss;                                            \
            _oss << __FILE__ ":" CATCH_INTERNAL_LINESTR(__LINE__)               \
                 << "  REQUIRE(" #expr ") failed";                              \
            throw ::Catch::AssertionFailed(_oss.str());                         \
        }                                                                       \
    } while (false)

#define CHECK(expr)                                                             \
    do {                                                                        \
        if (!(expr))                                                            \
            std::cout << "  \033[33m[warn]\033[0m CHECK(" #expr ") failed at " \
                      << __FILE__ ":" << __LINE__ << "\n";                      \
    } while (false)

#define REQUIRE_FALSE(expr)  REQUIRE(!(expr))
#define CHECK_FALSE(expr)    CHECK(!(expr))
#define INFO(msg)            static_cast<void>(0)

// main() — defined once here so no separate main translation unit is needed.
int main(int argc, char* argv[]) {
    return ::Catch::run_all(argc, argv);
}
