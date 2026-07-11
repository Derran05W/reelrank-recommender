#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "rr/domain/candidate.hpp"
#include "rr/domain/ids.hpp"
#include "rr/domain/recommendation.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/recommendation/candidate_generator.hpp"

namespace rr {

// TDD 12.6 creator-affinity candidate source: reels from creators the user previously enjoyed,
// exposed as a CandidateGenerator for the orchestrated pipeline (TDD 13). Reads the
// recommender-visible User::creatorAffinity estimate (populated by Simulator::step from observable
// outcomes, clamped to [0, 1]) — never any hidden state (D11).
//
// SCORE (documented choice): affinity * smoothedEngagementRate(reel), where the engagement rate is
// rr::smoothedPopularity(reel, /*priorMean=*/0.0) — Bayesian smoothing toward a ZERO prior with
// the same pseudo-impression constant as PopularityRecommender. A self-contained zero prior (vs.
// the global mean Popular uses) keeps this source O(reels from affine creators) instead of
// O(catalog). The product means a proven reel from a strongly-liked creator ranks highest; a brand
// new (zero-impression) reel from a liked creator scores 0 but can still be selected via the id
// tie-break when the pool is not yet full — covering the "recent OR strong-performing" intent of
// TDD 12.6. Order is a deterministic TOTAL order: score descending, ties by ascending ReelId.
//
// Only creators with affinity > 0 contribute. An empty/absent creatorAffinity map yields an EMPTY
// result. May return fewer than `count`. A creatorId -> reel-indices map is built ONCE in the ctor
// (the reel set is immutable in Phase 6).
//
// FILTERING: skips inactive reels and empty embeddings; dedup / seen / pool-cap are the
// Orchestrator's job (TDD 13). Each reel belongs to exactly one creator and each affinity key is
// distinct, so no reel is visited twice — no intra-source dedup is needed. CANDIDATE FIELDS are
// filled exactly like the vector sources (reelId, source=CreatorAffinity, real
// retrievalSimilarity = cos(effectivePreference(user), reel.embedding), D3-inverse distance) so
// these candidates are not starved at the Orchestrator's similarity-ordered pool cap and feed the
// ranker's similarity feature.
class CreatorAffinityCandidateSource final : public CandidateGenerator {
  public:
    CreatorAffinityCandidateSource(const std::vector<Reel> &reels, uint32_t count);

    std::vector<Candidate> generate(const User &user,
                                    const RecommendationRequest &request) override;

  private:
    struct Scored {
        double score;
        uint32_t index; // index into reels_
    };

    const std::vector<Reel> &reels_;
    uint32_t count_;
    std::unordered_map<CreatorId, std::vector<uint32_t>> reelsByCreator_;
    std::vector<Scored> scratch_; // reused across generate() calls (single-threaded core, D13)
};

} // namespace rr
