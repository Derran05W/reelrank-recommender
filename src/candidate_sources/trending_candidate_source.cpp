#include "rr/candidate_sources/trending_candidate_source.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

#include "rr/core/embedding.hpp"
#include "rr/recommendation/effective_preference.hpp"
#include "rr/recommendation/scoring.hpp"

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

TrendingCandidateSource::TrendingCandidateSource(const std::vector<Reel> &reels, uint32_t count,
                                                 double halfLifeSeconds)
    : reels_(reels), count_(count), halfLifeSeconds_(halfLifeSeconds) {}

std::vector<Candidate> TrendingCandidateSource::generate(const User &user,
                                                         const RecommendationRequest &request) {
    std::vector<Candidate> candidates;
    if (count_ == 0) {
        return candidates;
    }

    // One scoring pass; only reels with a positive decayed velocity qualify as "trending" (TDD
    // 12.4). request.requestTime is the logical `now` the accumulators are decayed forward to.
    scratch_.clear();
    for (std::size_t i = 0; i < reels_.size(); ++i) {
        const Reel &reel = reels_[i];
        if (!reel.active || reel.embedding.empty()) {
            continue;
        }
        const double score = trendingScore(reel, request.requestTime, halfLifeSeconds_);
        if (score > 0.0) {
            scratch_.push_back(Scored{score, static_cast<uint32_t>(i)});
        }
    }

    // Deterministic total order: score descending, ties by ascending ReelId. May yield fewer than
    // `count` (possibly zero) since only positive-velocity reels are included.
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
            makeCandidate(reels_[scratch_[i].index], CandidateSource::Trending, query));
    }
    return candidates;
}

} // namespace rr
