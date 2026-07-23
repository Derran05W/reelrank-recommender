#include "rr/core/pinned_hash.hpp"

namespace rr {

// PINNED SplitMix64-finalizer hash (see pinned_hash.hpp). This is the exact arithmetic that was
// inlined in cohort_hash.cpp through Phase 21, lifted VERBATIM and parameterized on a raw uint64 so
// cohortHash01 and the Phase 22 log sampler share one golden-tripwired body. DO NOT change these
// constants — cohort membership and log sampling are cross-package contracts.
double pinnedHash01(uint64_t value) {
    uint64_t x = value + 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return static_cast<double>(x >> 11) * 0x1.0p-53; // [0, 1)
}

} // namespace rr
