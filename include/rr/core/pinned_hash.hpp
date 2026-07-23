#pragma once

#include <cstdint>

namespace rr {

// Deterministic, rng-free uint64 -> [0, 1) map, PINNED to the SplitMix64 finalizer (Phase 10;
// promoted to core/ in Phase 22). The body is exactly the finalizer rr::Rng seeds with — golden
// gamma add, the two multiply-xorshift rounds, the final xorshift — followed by the canonical
// 53-bit-mantissa scaling. It is self-contained (NOT a call into infrastructure/random) so the
// golden values cannot drift if that helper is ever refactored.
//
// DO NOT change these constants. This exact bit pattern is a cross-package, cross-platform
// contract with golden-tripwire tests:
//   - rr::cohortHash01(userId) delegates here byte-identically (drift_scheduler_test.cpp pins
//     userId 4 -> 0.43145581774497377, userId 0 -> ~0.88331, ...); niche-treasure satisfaction
//     and drift cohorts both gate on it.
//   - Phase 22 training-log request sampling is pinnedHash01(requestId ^ salt) < rate (the two
//     salts in learning_v2/training_log_schema.hpp), so it draws ZERO rng and is golden-tripwired
//     exactly like cohortHash01 (no simulation-stream perturbation, D19).
double pinnedHash01(uint64_t value);

} // namespace rr
