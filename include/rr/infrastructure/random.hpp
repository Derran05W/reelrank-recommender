#pragma once

#include <cstdint>
#include <random>
#include <string_view>

namespace rr {

// Deterministic, cross-platform-portable random source.
//
// The std::*_distribution templates produce implementation-defined output, so they are BANNED
// (design decision D8). Rng provides in-house samplers over std::mt19937_64, whose raw output
// sequence IS standard-mandated, guaranteeing bit-identical runs across compilers and platforms.
class Rng {
  public:
    explicit Rng(uint64_t seed);

    // Raw 64-bit draw straight from the engine.
    uint64_t nextU64();

    // Uniform double in [0, 1).
    double uniform01();

    // Uniform double in [lo, hi).
    double uniform(double lo, double hi);

    // Uniform unsigned integer in [0, n) via unbiased modulo rejection. Throws
    // std::invalid_argument if n == 0.
    uint64_t uniformInt(uint64_t n);

    // Standard-normal draw (mean 0, stddev 1) via Box-Muller with a cached spare.
    double gaussian();

    // True with probability p.
    bool bernoulli(double p);

  private:
    std::mt19937_64 engine_;
    bool hasSpare_ = false;
    double spare_ = 0.0;
};

// FNV-1a 64-bit hash (offset basis 14695981039346656037, prime 1099511628211).
uint64_t fnv1a64(std::string_view s);

// SplitMix64 finalizer used as a seed mixer.
uint64_t splitmix64(uint64_t x);

// Derive an independent named stream from a master seed. Adding a new consumer never perturbs
// the streams already in use.
Rng forkRng(uint64_t masterSeed, std::string_view streamName);

} // namespace rr
