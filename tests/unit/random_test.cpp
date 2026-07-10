#include "rr/infrastructure/random.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

using rr::fnv1a64;
using rr::forkRng;
using rr::Rng;
using rr::splitmix64;

TEST(RngTest, SameSeedSameSequence) {
    Rng a(1234);
    Rng b(1234);
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(a.nextU64(), b.nextU64());
    }
    Rng c(7), d(7);
    for (int i = 0; i < 50; ++i) {
        EXPECT_DOUBLE_EQ(c.uniform01(), d.uniform01());
        EXPECT_DOUBLE_EQ(c.gaussian(), d.gaussian());
        EXPECT_EQ(c.uniformInt(1000), d.uniformInt(1000));
    }
}

TEST(RngTest, DifferentSeedsDiffer) {
    Rng a(1);
    Rng b(2);
    bool anyDiff = false;
    for (int i = 0; i < 20; ++i) {
        if (a.nextU64() != b.nextU64()) {
            anyDiff = true;
        }
    }
    EXPECT_TRUE(anyDiff);
}

TEST(RngTest, ForkStreamsDiffer) {
    Rng topics = forkRng(42, "topics");
    Rng users = forkRng(42, "users");
    bool anyDiff = false;
    for (int i = 0; i < 20; ++i) {
        if (topics.nextU64() != users.nextU64()) {
            anyDiff = true;
        }
    }
    EXPECT_TRUE(anyDiff);
}

TEST(RngTest, ForkStreamIndependence) {
    // Forking "a" must be unaffected by whether "b" was ever forked.
    Rng a1 = forkRng(42, "a");
    Rng b = forkRng(42, "b");
    (void)b.nextU64();
    Rng a2 = forkRng(42, "a");
    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(a1.nextU64(), a2.nextU64());
    }
}

TEST(RngTest, ForkSameSeedNameIdentical) {
    Rng a = forkRng(99, "stream");
    Rng b = forkRng(99, "stream");
    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(a.nextU64(), b.nextU64());
    }
}

TEST(RngTest, UniformRange) {
    Rng r(5);
    for (int i = 0; i < 10000; ++i) {
        double v = r.uniform(2.0, 5.0);
        EXPECT_GE(v, 2.0);
        EXPECT_LT(v, 5.0);
    }
}

TEST(RngTest, Uniform01Range) {
    Rng r(5);
    for (int i = 0; i < 10000; ++i) {
        double v = r.uniform01();
        EXPECT_GE(v, 0.0);
        EXPECT_LT(v, 1.0);
    }
}

TEST(RngTest, UniformIntRange) {
    Rng r(5);
    for (int i = 0; i < 10000; ++i) {
        uint64_t v = r.uniformInt(7);
        EXPECT_LT(v, 7u);
    }
}

TEST(RngTest, UniformIntZeroThrows) {
    Rng r(5);
    EXPECT_THROW(r.uniformInt(0), std::invalid_argument);
}

TEST(RngTest, BernoulliEdges) {
    Rng r(5);
    for (int i = 0; i < 1000; ++i) {
        EXPECT_FALSE(r.bernoulli(0.0));
        EXPECT_TRUE(r.bernoulli(1.0));
    }
}

TEST(RngTest, Moments) {
    Rng r(12345);
    const int n = 100000;
    double sumU = 0.0;
    for (int i = 0; i < n; ++i) {
        sumU += r.uniform01();
    }
    EXPECT_NEAR(sumU / n, 0.5, 0.01);

    Rng rg(54321);
    double sumG = 0.0, sumG2 = 0.0;
    for (int i = 0; i < n; ++i) {
        double g = rg.gaussian();
        sumG += g;
        sumG2 += g * g;
    }
    double meanG = sumG / n;
    double varG = sumG2 / n - meanG * meanG;
    EXPECT_NEAR(meanG, 0.0, 0.02);
    EXPECT_NEAR(std::sqrt(varG), 1.0, 0.02);

    Rng rb(777);
    int hits = 0;
    for (int i = 0; i < n; ++i) {
        if (rb.bernoulli(0.3)) {
            ++hits;
        }
    }
    EXPECT_NEAR(static_cast<double>(hits) / n, 0.3, 0.01);
}

TEST(RngTest, HashHelpers) {
    // FNV-1a offset basis of the empty string.
    EXPECT_EQ(fnv1a64(""), 14695981039346656037ULL);
    // Deterministic and content-sensitive.
    EXPECT_EQ(fnv1a64("topics"), fnv1a64("topics"));
    EXPECT_NE(fnv1a64("topics"), fnv1a64("users"));
    EXPECT_NE(splitmix64(0), splitmix64(1));
    EXPECT_EQ(splitmix64(42), splitmix64(42));
}

// --- Cross-platform determinism tripwire: golden values for seed 42. ------------------------
// If these ever fail on a new platform, rr::Rng is no longer bit-portable.
TEST(RngTest, GoldenNextU64Seed42) {
    Rng r(42);
    EXPECT_EQ(r.nextU64(), 2576493707698874361ULL);
    EXPECT_EQ(r.nextU64(), 17880808640956396325ULL);
    EXPECT_EQ(r.nextU64(), 17896956056310571724ULL);
}

TEST(RngTest, GoldenUniform01Seed42) {
    Rng r(42);
    EXPECT_DOUBLE_EQ(r.uniform01(), 0.13967200376411748);
}

TEST(RngTest, GoldenUniformIntSeed42) {
    Rng r(42);
    EXPECT_EQ(r.uniformInt(1000), 361u);
}

TEST(RngTest, GoldenGaussianSeed42) {
    Rng r(42);
    // libm (log/cos) may differ by ulps across platforms; tolerate that.
    EXPECT_NEAR(r.gaussian(), 1.9474165742871408, 1e-12);
}
