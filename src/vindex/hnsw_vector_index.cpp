#include "rr/vindex/hnsw_vector_index.hpp"

// D2 containment: this is one of the few translation units permitted to include vector-db headers.
// vector-db symbols live in the GLOBAL namespace (::HNSWIndex, ::HNSWConfig, ::Vector), so they are
// always qualified with :: here to keep them distinct from rr::HNSWConfig et al.
#include "distance_metrics.hpp"
#include "hnsw_index.hpp"
#include "vector.hpp"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace rr {

namespace {

// Distance-computation counter for TDD 17.3 ("Distance computations per query, if feasible").
// vector-db's bench/counting_metric.hpp is NOT reachable from this vendored build (bench/ is off
// the include path and its headers assume a `src/`-rooted include prefix we don't provide), so we
// implement the decorator locally — the guidance's documented fallback. It wraps the EXACT same
// ::EuclideanDistance the index uses by default (D3), delegating every distance() to it, so values
// are bit-identical to the non-counting path; only a relaxed atomic increment is added. Relaxed is
// sufficient: single-threaded here, and even under concurrent readers the total stays exact.
class CountingEuclidean : public ::DistanceMetric {
  public:
    CountingEuclidean() : inner_(std::make_shared<::EuclideanDistance>()) {}

    float distance(const ::Vector &a, const ::Vector &b) const override {
        count_.fetch_add(1, std::memory_order_relaxed);
        return inner_->distance(a, b);
    }

    std::uint64_t count() const { return count_.load(std::memory_order_relaxed); }
    void reset() { count_.store(0, std::memory_order_relaxed); }

  private:
    std::shared_ptr<const ::EuclideanDistance> inner_;
    mutable std::atomic<std::uint64_t> count_{0};
};

} // namespace

// Holds the concrete vector-db index. Defined only in this .cpp so the public header stays free of
// vector-db symbols (pimpl, per the frozen contract).
struct HNSWVectorIndex::Impl {
    // `counter` is declared BEFORE `index` so it is constructed first and can be handed to the
    // index's ctor (it co-owns the metric via shared_ptr). Null when counting is disabled, in which
    // case the index builds its own ::EuclideanDistance (D3) — the historical default path.
    std::shared_ptr<CountingEuclidean> counter;
    ::HNSWIndex index;
    std::size_t dimensions;

    Impl(std::size_t dims, const HNSWConfig &config, std::uint64_t seed, bool countDistanceComps)
        // Translate rr::HNSWConfig field names to vector-db's (m->M,
        // efConstruction->ef_construction, efSearch->ef_search) and thread the seed through (D8).
        // The metric arg is `counter` (a counting EuclideanDistance) when enabled, else nullptr =>
        // vector-db makes its own EuclideanDistance. Either way the distance math is
        // EuclideanDistance (D3).
        : counter(countDistanceComps ? std::make_shared<CountingEuclidean>() : nullptr),
          index(dims,
                ::HNSWConfig{static_cast<std::size_t>(config.m),
                             static_cast<std::size_t>(config.efConstruction),
                             static_cast<std::size_t>(config.efSearch), seed},
                counter),
          dimensions(dims) {}
};

HNSWVectorIndex::HNSWVectorIndex(std::size_t dimensions, const HNSWConfig &config,
                                 std::uint64_t seed, bool countDistanceComps)
    : impl_(std::make_unique<Impl>(dimensions, config, seed, countDistanceComps)) {}

HNSWVectorIndex::~HNSWVectorIndex() = default;

void HNSWVectorIndex::insert(const ReelId &id, const Embedding &embedding) {
    // D2: validate dimension and finiteness ourselves BEFORE touching vector-db, so a
    // vector-db-side throw never occurs for these two cases in normal operation. Duplicate-key
    // detection is deliberately left to vector-db (adapters catch nothing on the hot path).
    if (embedding.size() != impl_->dimensions) {
        throw std::invalid_argument("HNSWVectorIndex::insert: embedding dimension " +
                                    std::to_string(embedding.size()) + " != index dimension " +
                                    std::to_string(impl_->dimensions));
    }
    for (float component : embedding) {
        if (!std::isfinite(component)) {
            throw std::invalid_argument(
                "HNSWVectorIndex::insert: embedding has a non-finite component (id " +
                std::to_string(id.value) + ")");
        }
    }

    // D4: ReelId -> decimal string key. Only the adapter touches string keys.
    ::Vector vec(embedding);
    impl_->index.insert(vec,
                        std::to_string(id.value)); // dup-key throw (if any) propagates untouched
}

std::vector<VectorSearchResult> HNSWVectorIndex::search(const Embedding &query, size_t k) const {
    ::Vector q(query);
    const auto raw = impl_->index.search(q, k); // ascending distance; k==0/empty index => {}

    std::vector<VectorSearchResult> results;
    results.reserve(raw.size());
    for (const auto &[key, distance] : raw) {
        // D4: parse the decimal string key back to a ReelId. D3: distance -> cosine similarity.
        const ReelId reelId{static_cast<std::uint32_t>(std::stoul(key))};
        results.push_back(VectorSearchResult{reelId, distance, similarityFromEuclidean(distance)});
    }
    return results;
}

size_t HNSWVectorIndex::size() const { return impl_->index.size(); }

void HNSWVectorIndex::setEfSearch(size_t ef) { impl_->index.setEfSearch(ef); }

std::vector<size_t> HNSWVectorIndex::getLevelDistribution() const {
    return impl_->index.getLevelDistribution();
}

HnswGraphStats HNSWVectorIndex::graphStats() const {
    const ::HNSWIndex::IndexStats s = impl_->index.getIndexStats();
    HnswGraphStats out;
    out.nodeCount = s.node_count;
    out.maxLevel = s.max_level;
    out.levelDistribution = s.level_distribution;
    out.degreeHistogramLevel0 = s.degree_histogram_level0;
    return out;
}

uint64_t HNSWVectorIndex::distanceComputations() const {
    return impl_->counter ? impl_->counter->count() : 0ull;
}

void HNSWVectorIndex::resetDistanceCounter() {
    if (impl_->counter) {
        impl_->counter->reset();
    }
}

} // namespace rr
