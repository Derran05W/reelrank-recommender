#include "rr/simulation/cohort_hash.hpp"

#include <cstdint>

#include "rr/core/pinned_hash.hpp"

namespace rr {

// PINNED SplitMix64-finalizer hash (Phase 10; see cohort_hash.hpp). Phase 22 lifted the finalizer
// body VERBATIM into the shared rr::pinnedHash01(uint64) in core/ so cohort membership and the
// training-log request sampler use ONE golden-tripwired hash; cohortHash01 now delegates
// byte-identically (the drift_scheduler_test.cpp golden values — userId 4 -> 0.43145581774497377,
// etc. — are unchanged). DO NOT change pinnedHash01's constants — cohort membership is a
// cross-package contract (drift cohorts split "drifted vs control" on it; niche-treasure reels gate
// satisfaction on it).
double cohortHash01(UserId userId) { return pinnedHash01(static_cast<uint64_t>(userId.value)); }

} // namespace rr
