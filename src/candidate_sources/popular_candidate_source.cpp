#include "rr/candidate_sources/popular_candidate_source.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

#include "rr/core/embedding.hpp"
#include "rr/recommendation/effective_preference.hpp"
#include "rr/recommendation/popularity_recommender.hpp"
#include "rr/recommendation/scoring.hpp"

namespace rr {

namespace {

// Fill a Candidate for a non-vector source: real cosine similarity of the user's query against the
// reel (unit vectors, D3), with the D3-inverse Euclidean distance. rankingScore is left at 0 for
// the ranker to fill. Kept identical to the vector sources so the Orchestrator can order every
// source uniformly by retrievalSimilarity at the pool cap.
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

PopularCandidateSource::PopularCandidateSource(const std::vector<Reel> &reels, uint32_t count)
    : reels_(reels), count_(count) {}

std::vector<Candidate> PopularCandidateSource::generate(const User &user,
                                                        const RecommendationRequest & /*request*/) {
    std::vector<Candidate> candidates;
    if (count_ == 0) {
        return candidates;
    }

    // One prior-mean pass over the live catalog (TDD 12.3 Bayesian smoothing), then one scoring
    // pass over active, embeddable reels into the reused scratch buffer.
    const double priorMean = engagementPriorMean(reels_);
    scratch_.clear();
    for (std::size_t i = 0; i < reels_.size(); ++i) {
        const Reel &reel = reels_[i];
        if (!reel.active || reel.embedding.empty()) {
            continue;
        }
        scratch_.push_back(Scored{smoothedPopularity(reel, priorMean), static_cast<uint32_t>(i)});
    }

    // Deterministic total order: score descending, ties by ascending ReelId. partial_sort selects
    // just the top-n.
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
            makeCandidate(reels_[scratch_[i].index], CandidateSource::Popular, query));
    }
    return candidates;
}

} // namespace rr
