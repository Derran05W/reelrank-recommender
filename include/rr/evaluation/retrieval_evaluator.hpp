#pragma once

#include <cstddef>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/domain/reel.hpp"
#include "rr/recommendation/vector_index.hpp"
#include "rr/vindex/exact_vector_index.hpp"

namespace rr {

// One live retrieval-quality measurement for a single sampled request (TDD 18.1). Every value is
// derived by comparing a recommender's ANN index against an exact ground-truth index for the same
// query. recallAt10/recallAt50 are in [0, 1]; distanceError >= 0.
struct RetrievalSample {
    double recallAt10 = 0.0;
    double recallAt50 = 0.0;
    // Mean positionwise |d_ann,i - d_exact,i| over the first `kPrefix` results (see below).
    double distanceError = 0.0;
    // Effective corpus size seen by the exact search = min(kEval, ground-truth size). Exposed so
    // the caller can see the denominator that Recall@K used when the corpus is smaller than K.
    std::size_t exactK = 0;
};

// Owns an ExactVectorIndex ground truth over the active reels and scores an arbitrary ANN index
// against it on demand (TDD 18.1). Fully deterministic: it holds no rng and both underlying exact
// searches are order-stable (ascending distance, ties by ascending ReelId), so two runs with the
// same query produce identical samples. Built once per experiment (embeddings are immutable, D2)
// and reused for every sampled request.
//
// Metric definitions (TDD 18.1), all measured at k = kEval with a kPrefix "@10" window:
//   Recall@K = |ANN_K intersect Exact_K| / K, where ANN_K / Exact_K are the top-K reel-id SETS of
//     the ANN and exact searches. When the corpus has fewer than K items the denominator is
//     min(K, size) (documented). Recall@10 uses the first kPrefix results, Recall@50 the first
//     kEval results, both taken from the single k = kEval search.
//   Distance error = mean over positions i in [0, P) of |d_ann,i - d_exact,i|, where
//     P = min(kPrefix, |exact|, |ann|). This is the positionwise comparison of the i-th ANN result
//     distance against the i-th exact-neighbour distance ("compare ANN result distances with
//     exact-neighbour distances", TDD 18.1). For an exact ANN index it is exactly 0.
class RetrievalEvaluator {
  public:
    // k used for both the exact and ANN searches.
    static constexpr std::size_t kEval = 50;
    // Prefix window used for Recall@10 and the distance-error average.
    static constexpr std::size_t kPrefix = 10;

    // Build the ground-truth exact index from every ACTIVE reel (mirrors ExactVectorRecommender's
    // constructor). Embeddings are copied into the index; `reels` need not outlive the evaluator.
    RetrievalEvaluator(std::size_t dimensions, const std::vector<Reel> &reels);

    std::size_t groundTruthSize() const { return ground_.size(); }

    // Compare one ANN search against exact ground truth for `query`.
    RetrievalSample evaluate(const VectorIndex &annIndex, const Embedding &query) const;

  private:
    ExactVectorIndex ground_;
};

} // namespace rr
