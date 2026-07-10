#include "rr/infrastructure/clock.hpp"

#include <gtest/gtest.h>

TEST(ClockTest, TimestampIsUint64) {
    rr::Timestamp t = 100;
    static_assert(sizeof(rr::Timestamp) == 8, "Timestamp must be 64-bit");
    EXPECT_EQ(t, 100u);
}

TEST(ClockTest, StopwatchNonNegativeAndMonotonic) {
    rr::Stopwatch sw;
    double first = sw.elapsedMs();
    EXPECT_GE(first, 0.0);
    // Burn a little wall time without sleeping.
    volatile double acc = 0.0;
    for (int i = 0; i < 1000000; ++i) {
        acc += i;
    }
    double second = sw.elapsedMs();
    EXPECT_GE(second, first);
}

TEST(ClockTest, ResetDropsElapsed) {
    rr::Stopwatch sw;
    volatile double acc = 0.0;
    for (int i = 0; i < 1000000; ++i) {
        acc += i;
    }
    double before = sw.elapsedMs();
    sw.reset();
    double after = sw.elapsedMs();
    EXPECT_LE(after, before);
}
