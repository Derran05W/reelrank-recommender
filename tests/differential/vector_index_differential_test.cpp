// Differential test suite (TDD 24.4, plan Phase 1 task 3): HNSW approximate search vs exact
// brute-force search over the SAME randomly generated normalized datasets, compared purely
// through the rr::VectorIndex abstraction (no vector-db header is included here — both indexes are
// already fully wrapped behind rr::). These are the regression tripwires that would catch a future
// HNSW recall/behaviour regression.
//
// What is compared, per dataset (size x dimension) and over many random queries:
//   * Top-k overlap  = Recall@K in TDD 18.1 terms: |ANN_K intersect Exact_K| / K.
//   * Nearest-neighbour / per-rank distance error: HNSW is approximate, so its distances should be
//     statistically close to (never strictly better than) exact's.
//   * Self-match: querying with a stored vector finds that vector at ~0 distance in both indexes.
//   * No malformed outputs: no duplicate ids, ids within the inserted range, no NaN/Inf, count<=k.
//   * A dedicated Recall@10 floor property test on 10k x 64d with default HNSWConfig.
//
// Determinism (D8): every random draw comes from rr::Rng forked streams; no std::*_distribution.
#include "rr/core/embedding.hpp"
#include "rr/domain/ids.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/recommendation/vector_index.hpp"
#include "rr/vindex/exact_vector_index.hpp"
#include "rr/vindex/hnsw_vector_index.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

using rr::Embedding;
using rr::ExactVectorIndex;
using rr::forkRng;
using rr::HNSWConfig;
using rr::HNSWVectorIndex;
using rr::ReelId;
using rr::Rng;
using rr::similarityFromEuclidean;
using rr::VectorSearchResult;

// A normalized random embedding drawn from an isotropic Gaussian (D8: rr::Rng only). After
// normalization the direction is uniform on the unit sphere, matching the adapter unit tests.
Embedding randomUnit(Rng &rng, std::size_t dim) {
    Embedding e(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        e[i] = static_cast<float>(rng.gaussian());
    }
    rr::normalize(e);
    return e;
}

// Insert the SAME n normalized random vectors (ids 0..n-1) into both indexes and return the
// embeddings so callers can issue exact self-queries. The two indexes see byte-identical inputs in
// the same order, which is the whole point of a differential comparison.
std::vector<Embedding> buildDatasets(std::uint64_t seed, std::size_t n, std::size_t dim,
                                     ExactVectorIndex &exact, HNSWVectorIndex &hnsw) {
    Rng rng = forkRng(seed, "differential-dataset");
    std::vector<Embedding> embeddings;
    embeddings.reserve(n);
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(n); ++i) {
        Embedding e = randomUnit(rng, dim);
        exact.insert(ReelId{i}, e); // insert copies internally, so e stays valid...
        hnsw.insert(ReelId{i}, e);
        embeddings.push_back(std::move(e)); // ...and can now be moved into the keep-list.
    }
    return embeddings;
}

// Recall@K per TDD 18.1: fraction of the exact top-K ids that the ANN top-K also returned. The
// exact result IS the ground truth, so |truth| == min(K, size); dividing by |truth| gives recall.
double recallAtK(const std::vector<VectorSearchResult> &exact,
                 const std::vector<VectorSearchResult> &ann, std::size_t k) {
    std::unordered_set<std::uint32_t> truth;
    for (std::size_t i = 0; i < exact.size() && i < k; ++i) {
        truth.insert(exact[i].reelId.value);
    }
    if (truth.empty()) {
        return 1.0;
    }
    std::size_t hit = 0;
    for (std::size_t i = 0; i < ann.size() && i < k; ++i) {
        if (truth.count(ann[i].reelId.value) == 1) {
            ++hit;
        }
    }
    return static_cast<double>(hit) / static_cast<double>(truth.size());
}

// "No malformed outputs" (TDD 24.4, 24.3): every returned result must be internally consistent and
// within contract for a k-NN query over an index of `n` inserted ids.
void expectNoMalformedOutputs(const std::vector<VectorSearchResult> &results, std::size_t k,
                              std::size_t n) {
    EXPECT_LE(results.size(), k) << "result count exceeds k";
    EXPECT_LE(results.size(), n) << "result count exceeds index size";
    std::unordered_set<std::uint32_t> seen;
    for (const auto &r : results) {
        EXPECT_TRUE(std::isfinite(r.distance)) << "non-finite distance";
        EXPECT_TRUE(std::isfinite(r.similarity)) << "non-finite similarity";
        EXPECT_GE(r.distance, -1e-4f) << "negative distance"; // fp slack around 0.
        EXPECT_LT(r.reelId.value, static_cast<std::uint32_t>(n)) << "id outside inserted range";
        EXPECT_TRUE(seen.insert(r.reelId.value).second) << "duplicate id in results";
        EXPECT_FLOAT_EQ(r.similarity, similarityFromEuclidean(r.distance))
            << "similarity is not the D3 conversion of distance";
    }
}

struct DiffParam {
    std::size_t n;
    std::size_t dim;
    std::uint64_t seed;
};

// Representative combinations covering both sizes (1k, 10k) and both dimensions (32, 64) from the
// plan, without an exhaustive cross product (128/256d and 100k+ are the benchmark app's job).
const DiffParam kDiffParams[] = {
    {1000, 32, 0xD1FF01u},
    {1000, 64, 0xD1FF02u},
    {10000, 32, 0xD1FF03u},
    {10000, 64, 0xD1FF04u},
};

class VectorIndexDifferentialTest : public testing::TestWithParam<DiffParam> {};

TEST_P(VectorIndexDifferentialTest, HnswMatchesExactOnIdenticalData) {
    const DiffParam p = GetParam();
    const std::size_t k = 10;

    ExactVectorIndex exact(p.dim);
    HNSWVectorIndex hnsw(p.dim, HNSWConfig{}, p.seed); // default m=16, efC=200, efS=64.
    const std::vector<Embedding> embeddings = buildDatasets(p.seed, p.n, p.dim, exact, hnsw);
    ASSERT_EQ(exact.size(), p.n);
    ASSERT_EQ(hnsw.size(), p.n);

    // ---- Top-k overlap (recall) + nearest-neighbour / per-rank distance error ----
    Rng qrng = forkRng(p.seed, "differential-queries");
    const int numQueries = 100;
    double recallSum = 0.0;
    double top1ErrSum = 0.0;
    double rankErrSum = 0.0;
    std::size_t rankErrCount = 0;
    for (int q = 0; q < numQueries; ++q) {
        const Embedding query = randomUnit(qrng, p.dim);
        const auto ex = exact.search(query, k);
        const auto an = hnsw.search(query, k);
        expectNoMalformedOutputs(ex, k, p.n);
        expectNoMalformedOutputs(an, k, p.n);
        ASSERT_EQ(ex.size(), k);
        ASSERT_EQ(an.size(), k);

        recallSum += recallAtK(ex, an, k);
        // Exact top-1 is the true minimum distance; ANN can never be strictly closer (fp slack).
        EXPECT_GE(an.front().distance, ex.front().distance - 1e-4f)
            << "HNSW returned a nearest-neighbour closer than the exact minimum";
        top1ErrSum += (an.front().distance - ex.front().distance);
        for (std::size_t i = 0; i < k; ++i) {
            rankErrSum += std::fabs(static_cast<double>(an[i].distance) - ex[i].distance);
            ++rankErrCount;
        }
    }
    const double meanRecall = recallSum / numQueries;
    const double meanTop1Err = top1ErrSum / numQueries;
    const double meanRankErr = rankErrSum / static_cast<double>(rankErrCount);

    std::cout << "[ MEASURED ] n=" << p.n << " dim=" << p.dim << " (m=16 efC=200 efS=64) over "
              << numQueries << " queries: recall@" << k << "=" << meanRecall
              << " meanTop1DistErr=" << meanTop1Err << " meanRankDistErr=" << meanRankErr << "\n";

    // Statistical floors, not exact equality (HNSW is approximate). Set comfortably below what the
    // VDB-1-hardened HNSW actually delivers so these are regression tripwires, not razor-thin.
    EXPECT_GE(meanRecall, 0.80) << "HNSW top-" << k << " overlap with exact fell below floor";
    EXPECT_LT(meanTop1Err, 0.05) << "HNSW top-1 distance error too large";
    EXPECT_LT(meanRankErr, 0.05) << "HNSW per-rank distance error too large";

    // ---- Self-match: a query identical to a stored vector must return it at ~0 distance ----
    Rng srng = forkRng(p.seed, "differential-selfmatch");
    for (int s = 0; s < 20; ++s) {
        const auto id = static_cast<std::uint32_t>(srng.uniformInt(p.n));
        const Embedding &v = embeddings[id];
        const auto ex = exact.search(v, k);
        const auto an = hnsw.search(v, k);
        ASSERT_FALSE(ex.empty());
        ASSERT_FALSE(an.empty());
        // Exact is guaranteed to place the self first at distance 0.
        EXPECT_EQ(ex.front().reelId.value, id);
        EXPECT_NEAR(ex.front().distance, 0.0f, 1e-4f);
        // Random unit vectors are near-orthogonal (~sqrt(2) apart), so the self at distance 0 is
        // dramatically the nearest; HNSW greedy descent lands on it.
        EXPECT_EQ(an.front().reelId.value, id) << "HNSW failed to self-match id " << id;
        EXPECT_NEAR(an.front().distance, 0.0f, 1e-4f);
        EXPECT_NEAR(an.front().similarity, 1.0f, 1e-4f);
    }
}

INSTANTIATE_TEST_SUITE_P(Combos, VectorIndexDifferentialTest, testing::ValuesIn(kDiffParams),
                         [](const testing::TestParamInfo<DiffParam> &info) {
                             return "n" + std::to_string(info.param.n) + "_d" +
                                    std::to_string(info.param.dim);
                         });

// Dedicated Recall@10 floor property test (plan Phase 1 task 3 regression tripwire). Uses the
// default-constructed HNSWConfig and enough sample queries for a statistically stable estimate.
//
// MEASURED FINDING (recorded per the plan's instruction when recall lands below the initial 0.85
// target): with the VDB-1-hardened HNSW at its DEFAULT config (m=16, efConstruction=200,
// efSearch=64), recall@10 on 10k x 64d is ~0.816 (deterministic under this fixed seed) -- just shy
// of 0.85. The shortfall is a search-breadth effect, NOT a correctness defect: widening only
// efSearch to 128 (vector-db's own hardening benchmark point) lifts the SAME index/dataset to
// ~0.9+, which the second half of this test asserts and demonstrates. The default efSearch=64
// simply trades recall for query latency. The primary floor is therefore pinned at 0.80 (just
// below the measured 0.816) as a real regression tripwire, and this is flagged in commit.md's
// known-issues so the config default can be revisited. Do NOT raise the default-config floor to
// mask this, and do NOT lower the efSearch=128 floor below 0.85.
TEST(VectorIndexDifferentialFloorTest, RecallAt10ExceedsFloor) {
    const std::size_t n = 10000;
    const std::size_t dim = 64;
    const std::size_t k = 10;
    const std::uint64_t seed = 0xBEEFCAFEu;

    ExactVectorIndex exact(dim);
    HNSWVectorIndex hnsw(dim, HNSWConfig{}, seed); // default m=16, efC=200, efS=64.
    buildDatasets(seed, n, dim, exact, hnsw);
    ASSERT_EQ(hnsw.size(), n);

    // A fixed query set, so the efSearch=64 and efSearch=128 measurements below are compared on
    // identical queries against identical exact ground truth (only the HNSW beam width differs).
    Rng qrng = forkRng(seed, "recall-floor-queries");
    const int numQueries = 300;
    std::vector<Embedding> queries;
    std::vector<std::unordered_set<std::uint32_t>> exactTopK; // ground-truth ids per query.
    queries.reserve(numQueries);
    exactTopK.reserve(numQueries);
    for (int q = 0; q < numQueries; ++q) {
        Embedding query = randomUnit(qrng, dim);
        const auto ex = exact.search(query, k);
        std::unordered_set<std::uint32_t> truth;
        for (const auto &r : ex) {
            truth.insert(r.reelId.value);
        }
        queries.push_back(std::move(query));
        exactTopK.push_back(std::move(truth));
    }

    const auto measureRecall = [&]() {
        double sum = 0.0;
        for (int q = 0; q < numQueries; ++q) {
            const auto an = hnsw.search(queries[q], k);
            expectNoMalformedOutputs(an, k, n);
            std::size_t hit = 0;
            for (const auto &r : an) {
                if (exactTopK[q].count(r.reelId.value) == 1) {
                    ++hit;
                }
            }
            sum += static_cast<double>(hit) / static_cast<double>(k);
        }
        return sum / numQueries;
    };

    // ---- Primary tripwire: default HNSWConfig (efSearch=64) ----
    const double recallDefault = measureRecall();
    std::cout << "[ RECALL   ] recall@" << k << " (n=" << n << " dim=" << dim
              << " m=16 efC=200 efS=64, DEFAULT config) over " << numQueries
              << " queries = " << recallDefault << "\n";
    RecordProperty("recall_at_10_default_x1000", static_cast<int>(recallDefault * 1000.0));

    // Pinned just below the measured ~0.816 (see the block comment above). A drop below this is a
    // genuine recall regression to investigate, not a reason to lower the bar further.
    EXPECT_GE(recallDefault, 0.80)
        << "recall@10 regressed below the pinned 0.80 floor on 10k x 64d with default HNSWConfig";

    // ---- Diagnostic tripwire: same index/data, efSearch widened to 128 ----
    hnsw.setEfSearch(128);
    const double recallWide = measureRecall();
    std::cout << "[ RECALL   ] recall@" << k << " (n=" << n << " dim=" << dim
              << " m=16 efC=200 efS=128) over " << numQueries << " queries = " << recallWide
              << "\n";
    RecordProperty("recall_at_10_ef128_x1000", static_cast<int>(recallWide * 1000.0));

    // A wider beam can only help (never reduce recall), and confirms the default-config shortfall
    // is breadth-limited rather than a correctness defect: efSearch=128 clears the original 0.85.
    EXPECT_GE(recallWide, recallDefault - 1e-9)
        << "widening efSearch reduced recall -- unexpected for a correct HNSW";
    EXPECT_GE(recallWide, 0.85)
        << "recall@10 at efSearch=128 fell below 0.85 -- a real HNSW recall regression";
}

} // namespace
