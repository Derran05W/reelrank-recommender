#include "rr/candidate_sources/creator_affinity_candidate_source.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

#include "rr/core/embedding.hpp"
#include "rr/recommendation/effective_preference.hpp"
#include "rr/recommendation/popularity_recommender.hpp"

namespace rr {

namespace {

// See PopularCandidateSource: fill a non-vector Candidate with the real cosine similarity (D3) and
// its D3-inverse distance so every source is comparable at the Orchestrator's pool cap.
Candidate makeCandidate(const Reel &reel, CandidateSource source, const Embedding &query) {
    Candidate candidate{};
    candidate.reelId = reel.id;
    candidate.source = source;
    const float sim = dot(query, reel.embedding);
    candidate.retrievalSimilarity = sim;
    candidate.retrievalDistance = std::sqrt(std::max(0.0f, 2.0f - 2.0f * sim));
    candidate.rankingScore = 0.0f;
    return candidate;
}

} // namespace

CreatorAffinityCandidateSource::CreatorAffinityCandidateSource(const std::vector<Reel> &reels,
                                                               uint32_t count)
    : reels_(reels), count_(count) {
    // Immutable reel set in Phase 6: build the creatorId -> reel-indices map once (TDD 12.6). All
    // reels are indexed here; active/embedding filtering happens per request in generate().
    for (std::size_t i = 0; i < reels_.size(); ++i) {
        reelsByCreator_[reels_[i].creatorId].push_back(static_cast<uint32_t>(i));
    }
}

std::vector<Candidate>
CreatorAffinityCandidateSource::generate(const User &user,
                                         const RecommendationRequest & /*request*/) {
    std::vector<Candidate> candidates;
    if (count_ == 0 || user.creatorAffinity.empty()) {
        return candidates;
    }

    // Score reels only from creators with positive affinity: affinity * smoothed engagement rate
    // (zero prior). Iterating the affinity map in unspecified order is fine — the final order is
    // fixed by the strict total-order comparator below (score desc, then unique ascending id).
    scratch_.clear();
    for (const auto &[creatorId, affinity] : user.creatorAffinity) {
        if (affinity <= 0.0f) {
            continue;
        }
        const auto it = reelsByCreator_.find(creatorId);
        if (it == reelsByCreator_.end()) {
            continue;
        }
        for (const uint32_t idx : it->second) {
            const Reel &reel = reels_[idx];
            if (!reel.active || reel.embedding.empty()) {
                continue;
            }
            const double score =
                static_cast<double>(affinity) * smoothedPopularity(reel, /*priorMean=*/0.0);
            scratch_.push_back(Scored{score, idx});
        }
    }

    const std::size_t n = std::min(static_cast<std::size_t>(count_), scratch_.size());
    std::partial_sort(scratch_.begin(), scratch_.begin() + static_cast<std::ptrdiff_t>(n),
                      scratch_.end(), [this](const Scored &a, const Scored &b) {
                          if (a.score != b.score) {
                              return a.score > b.score;
                          }
                          return reels_[a.index].id.value < reels_[b.index].id.value;
                      });

    const Embedding &query = effectivePreference(user);
    candidates.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        candidates.push_back(
            makeCandidate(reels_[scratch_[i].index], CandidateSource::CreatorAffinity, query));
    }
    return candidates;
}

} // namespace rr
