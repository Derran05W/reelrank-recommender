#pragma once

#include "rr/core/embedding.hpp"
#include "rr/domain/user.hpp"

namespace rr {

// The query vector a recommender scores a user against. At Phases 4-6 this is simply the
// cold-start static estimate (TDD 11.1) - there are no online updates yet. Phase 7 will replace
// the body with the TDD 8.3 blend longTermWeight*longTermPreference +
// sessionWeight*sessionPreference; every personalizing recommender and the retrieval evaluator
// route through this one helper so that upgrade is a single edit.
inline const Embedding &effectivePreference(const User &user) { return user.estimatedPreference; }

} // namespace rr
