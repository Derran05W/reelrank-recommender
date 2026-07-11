#include "rr/candidate_sources/hnsw_candidate_source.hpp"

#include <cstddef>
#include <utility>

#include "rr/core/embedding.hpp"
#include "rr/recommendation/effective_preference.hpp"
#include "rr/recommendation/vector_index.hpp"

namespace rr {

HNSWCandidateSource::HNSWCandidateSource(const HNSWVectorIndex &index) : index_(index) {}

std::vector<Candidate> HNSWCandidateSource::generate(const User &user,
                                                     const RecommendationRequest &request) {
    const Embedding &query = effectivePreference(user);
    const std::size_t k = static_cast<std::size_t>(request.candidateLimit);
    std::vector<Candidate> candidates;
    if (k == 0 || index_.size() == 0) {
        return candidates;
    }

    const std::vector<VectorSearchResult> results = index_.search(query, k);
    candidates.reserve(results.size());
    for (const VectorSearchResult &result : results) {
        Candidate candidate{};
        candidate.reelId = result.reelId;
        candidate.source = CandidateSource::VectorHNSW;
        candidate.retrievalDistance = result.distance;
        // D3 (overrides TDD 12.1's 1/(1+d)): for unit vectors similarity = 1 - d^2/2 = cos.
        candidate.retrievalSimilarity = similarityFromEuclidean(result.distance);
        candidate.rankingScore = 0.0f;
        candidates.push_back(std::move(candidate));
    }
    return candidates;
}

} // namespace rr
