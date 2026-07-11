#include "rr/recommendation/hnsw_recommender.hpp"

#include <utility>

#include "rr/core/embedding.hpp"
#include "rr/domain/candidate.hpp"
#include "rr/infrastructure/config.hpp"

namespace rr {

HNSWRecommender::HNSWRecommender(const RecommenderDeps &deps, Rng rng)
    : reels_(deps.reels), users_(deps.users),
      // D8 draw order: the constructor makes exactly ONE rng draw — the HNSW index seed — as the
      // first and only use of the forked "recommender" stream. index_ is the first member to be
      // initialized, so rng.nextU64() here is that single draw. Two same-seed recommenders build
      // byte-identical graphs and therefore serve identical feeds.
      index_(deps.config.simulation.dimensions, deps.config.hnsw, rng.nextU64()), source_(index_),
      orchestrator_({&source_}, reels_) {
    // Build the graph once over all active reels; embeddings are immutable (D2). insert validates
    // dimension/finiteness and throws std::invalid_argument on a bad embedding — a setup error
    // (D10) that correctly surfaces from the constructor. Mirrors ExactVectorRecommender exactly.
    for (const Reel &reel : reels_) {
        if (reel.active) {
            index_.insert(reel.id, reel.embedding);
        }
    }
}

RecommendationResponse HNSWRecommender::recommend(const RecommendationRequest &request) {
    const User &user = users_[request.userId.value];
    return orchestrator_.recommend(user, request);
}

std::string HNSWRecommender::name() const { return toString(RecommendationAlgorithm::Hnsw); }

const VectorIndex *HNSWRecommender::retrievalIndex() const { return &index_; }

} // namespace rr
