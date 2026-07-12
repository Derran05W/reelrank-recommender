#include "rr/recommendation/diversity_reranker.hpp"

#include <vector>

namespace rr {

DiversityReranker::DiversityReranker(const std::vector<Reel> &reels, const DiversityConfig &config)
    : config_(config), constraint_(reels, config), mmr_(reels, config.mmrLambda) {}

std::vector<RankedReel> DiversityReranker::rerank(const User &user,
                                                  const std::vector<Candidate> &rankedCandidates,
                                                  std::size_t feedSize) const {
    if (!config_.useMmr) {
        // Constraints only: the hard-selected set in relevance order with the consecutive-topic
        // swap applied — ConstraintReranker.rerank verbatim.
        return constraint_.rerank(user, rankedCandidates, feedSize);
    }
    // Constraints pick the SET (hard rules), MMR orders WITHIN it. selectFeed omits the cosmetic
    // consecutive-topic swap because MMR's diversity objective subsumes it (see header). MMR
    // reorders the whole set (feedSize = set size) and reassigns ranks 0..n-1.
    const std::vector<Candidate> set = constraint_.selectFeed(user, rankedCandidates, feedSize);
    return mmr_.rerank(user, set, set.size());
}

} // namespace rr
