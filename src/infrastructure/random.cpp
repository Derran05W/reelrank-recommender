#include "rr/infrastructure/random.hpp"

#include <cmath>
#include <stdexcept>

namespace rr {

uint64_t splitmix64(uint64_t x) {
    uint64_t z = x + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

uint64_t fnv1a64(std::string_view s) {
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char c : s) {
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

Rng::Rng(uint64_t seed) : engine_(splitmix64(seed)) {}

uint64_t Rng::nextU64() { return engine_(); }

double Rng::uniform01() {
    // 53 bits of mantissa scaled into [0, 1).
    return (nextU64() >> 11) * 0x1.0p-53;
}

double Rng::uniform(double lo, double hi) { return lo + uniform01() * (hi - lo); }

uint64_t Rng::uniformInt(uint64_t n) {
    if (n == 0) {
        throw std::invalid_argument("rr::Rng::uniformInt: n must be > 0");
    }
    // threshold = 2^64 mod n; reject draws below it to remove modulo bias.
    const uint64_t threshold = (0u - n) % n;
    uint64_t r;
    do {
        r = nextU64();
    } while (r < threshold);
    return r % n;
}

double Rng::gaussian() {
    if (hasSpare_) {
        hasSpare_ = false;
        return spare_;
    }
    double u1;
    do {
        u1 = uniform01();
    } while (u1 <= 0.0);
    const double u2 = uniform01();
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double theta = 2.0 * 3.14159265358979323846 * u2;
    spare_ = r * std::sin(theta);
    hasSpare_ = true;
    return r * std::cos(theta);
}

bool Rng::bernoulli(double p) { return uniform01() < p; }

Rng forkRng(uint64_t masterSeed, std::string_view streamName) {
    return Rng(splitmix64(masterSeed ^ fnv1a64(streamName)));
}

} // namespace rr
