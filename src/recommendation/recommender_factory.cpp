#include "rr/recommendation/recommender_factory.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include "rr/recommendation/exact_vector_recommender.hpp"
#include "rr/recommendation/full_recommender.hpp"
#include "rr/recommendation/hnsw_exploration_recommender.hpp"
#include "rr/recommendation/hnsw_ranker_recommender.hpp"
#include "rr/recommendation/hnsw_recommender.hpp"
#include "rr/recommendation/popularity_recommender.hpp"
#include "rr/recommendation/random_recommender.hpp"

namespace rr {

namespace {

// Every baseline looks the user/reel up by id used directly as a vector index (deps.users[i],
// deps.reels[i]), so the dense-id invariant is load-bearing. Validate it here at construction
// (setup, so throwing is allowed by D10) rather than risk a silent out-of-bounds on the hot path.
void validateDenseIds(const RecommenderDeps &deps) {
    for (std::size_t i = 0; i < deps.reels.size(); ++i) {
        if (deps.reels[i].id.value != i) {
            throw std::invalid_argument("makeRecommender: reels are not densely indexed; reels[" +
                                        std::to_string(i) +
                                        "].id.value == " + std::to_string(deps.reels[i].id.value));
        }
    }
    for (std::size_t i = 0; i < deps.users.size(); ++i) {
        if (deps.users[i].id.value != i) {
            throw std::invalid_argument("makeRecommender: users are not densely indexed; users[" +
                                        std::to_string(i) +
                                        "].id.value == " + std::to_string(deps.users[i].id.value));
        }
    }
}

} // namespace

std::unique_ptr<Recommender> makeRecommender(RecommendationAlgorithm algorithm,
                                             const RecommenderDeps &deps, Rng rng) {
    validateDenseIds(deps);
    switch (algorithm) {
    case RecommendationAlgorithm::Random:
        return std::make_unique<RandomRecommender>(deps, std::move(rng));
    case RecommendationAlgorithm::Popularity:
        return std::make_unique<PopularityRecommender>(deps, std::move(rng));
    case RecommendationAlgorithm::ExactVector:
        return std::make_unique<ExactVectorRecommender>(deps, std::move(rng));
    case RecommendationAlgorithm::Hnsw:
        return std::make_unique<HNSWRecommender>(deps, std::move(rng));
    case RecommendationAlgorithm::HnswRanker:
        return std::make_unique<HNSWRankerRecommender>(deps, std::move(rng));
    case RecommendationAlgorithm::HnswRankerExploration:
        return std::make_unique<HNSWExplorationRecommender>(deps, std::move(rng));
    case RecommendationAlgorithm::HnswRankerDiversity:
        return std::make_unique<FullRecommender>(deps, std::move(rng));
    case RecommendationAlgorithm::OracleSatisfaction:
        // D18: the oracle arm reads hidden state and lives in evaluation/ — only the
        // ExperimentRunner may construct it. Rejecting it HERE keeps recommendation-side code
        // structurally unable to serve it as a policy.
        throw std::invalid_argument("makeRecommender: oracle_satisfaction is evaluation-only "
                                    "(constructed by the ExperimentRunner, never the factory)");
    case RecommendationAlgorithm::HnswLearnedRanker:
        // Phase 23: the learned-ranking arm needs the in-loop retrainer + a live handle to its
        // LearnedRanker for deterministic model hot-swaps, and it lives in rr_learning_v2 (which
        // depends on this recommendation layer). The EVENT RUNNER constructs it directly (mirroring
        // the OracleSatisfaction event-only precedent); rejecting it here keeps rr_recommend free
        // of any rr_learning_v2 dependency (no library cycle) and prevents a modelless learned
        // policy.
        throw std::invalid_argument("makeRecommender: hnsw_learned_ranker is constructed by the "
                                    "EventDrivenRunner (it wires the in-loop retrainer), never the "
                                    "factory");
    }
    throw std::invalid_argument("makeRecommender: unknown RecommendationAlgorithm value");
}

} // namespace rr
