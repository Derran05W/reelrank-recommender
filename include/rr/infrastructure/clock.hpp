#pragma once

#include <chrono>
#include <cstdint>

namespace rr {

// Simulated time in seconds, advanced by the simulator's logical clock (design decision D9).
// Never derived from wall-clock time.
using Timestamp = uint64_t;

// Wall-clock elapsed-time measurement. This is the ONLY place std::chrono::steady_clock is used
// (D9); everything else takes a Timestamp parameter.
class Stopwatch {
  public:
    Stopwatch() : start_(std::chrono::steady_clock::now()) {}

    void reset() { start_ = std::chrono::steady_clock::now(); }

    double elapsedMs() const {
        const auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

  private:
    std::chrono::steady_clock::time_point start_;
};

} // namespace rr
