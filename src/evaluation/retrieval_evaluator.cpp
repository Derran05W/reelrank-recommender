#include "rr/evaluation/retrieval_evaluator.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_set>

#include "rr/domain/ids.hpp"

namespace rr {

namespace {

// Recall@`prefix`: |ANN_prefix intersect Exact_prefix| / min(prefix, |exact|). Both top-lists are
// treated as SETS of reel ids (TDD 18.1). The denominator uses the exact list's effective length so
// a corpus smaller than `prefix` is scored against min(prefix, size) rather than `prefix`.
double recallAtPrefix(const std::vector<VectorSearchResult> &exact,
                      const std::vector<VectorSearchResult> &ann, std::size_t prefix) {
    const std::size_t denom = std::min(prefix, exact.size());
    if (denom == 0) {
        return 0.0;
    }
    std::unordered_set<ReelId> exactIds;
    exactIds.reserve(denom * 2);
    for (std::size_t i = 0; i < denom; ++i) {
        exactIds.insert(exact[i].reelId);
    }
    std::size_t hit = 0;
    const std::size_t annCount = std::min(prefix, ann.size());
    for (std::size_t i = 0; i < annCount; ++i) {
        if (exactIds.count(ann[i].reelId) > 0) {
            ++hit;
        }
    }
    return static_cast<double>(hit) / static_cast<double>(denom);
}

} // namespace

RetrievalEvaluator::RetrievalEvaluator(std::size_t dimensions, const std::vector<Reel> &reels)
    : ground_(dimensions) {
    // Insert every active reel; embeddings are immutable (D2). insert() validates dimension and
    // finiteness and throws std::invalid_argument on a bad embedding, which is a setup error (D10)
    // and correctly surfaces here.
    for (const Reel &reel : reels) {
        if (reel.active) {
            ground_.insert(reel.id, reel.embedding);
        }
    }
}

RetrievalSample RetrievalEvaluator::evaluate(const VectorIndex &annIndex,
                                             const Embedding &query) const {
    const std::vector<VectorSearchResult> exact = ground_.search(query, kEval);
    const std::vector<VectorSearchResult> ann = annIndex.search(query, kEval);

    RetrievalSample sample;
    sample.exactK = exact.size();
    sample.recallAt10 = recallAtPrefix(exact, ann, kPrefix);
    sample.recallAt50 = recallAtPrefix(exact, ann, kEval);

    // Positionwise distance error over the first P = min(kPrefix, |exact|, |ann|) results.
    const std::size_t p = std::min({kPrefix, exact.size(), ann.size()});
    if (p > 0) {
        double sum = 0.0;
        for (std::size_t i = 0; i < p; ++i) {
            sum += std::abs(static_cast<double>(ann[i].distance) -
                            static_cast<double>(exact[i].distance));
        }
        sample.distanceError = sum / static_cast<double>(p);
    }
    return sample;
}

} // namespace rr
