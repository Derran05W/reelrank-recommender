#pragma once

#include "rr/core/embedding.hpp"
#include "rr/domain/user.hpp"

namespace rr {

// The query vector a recommender scores a user against. Since Phase 7, estimatedPreference IS
// the TDD 8.3 blend: OnlineUserStateUpdater maintains it as
// normalize(longTermWeight*longTermPreference + sessionWeight*sessionPreference) after every
// interaction (the cached effective preference), so this helper stays a const-ref return and
// every personalizing recommender plus the retrieval evaluator keep routing through it. With
// learning disabled (LearningConfig.enabled = false) it degenerates to the cold-start static
// estimate of TDD 11.1, exactly the pre-Phase-7 behaviour.
inline const Embedding &effectivePreference(const User &user) { return user.estimatedPreference; }

} // namespace rr
