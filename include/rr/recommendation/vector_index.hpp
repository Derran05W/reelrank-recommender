#pragma once

#include <cstddef>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/domain/ids.hpp"

namespace rr {

struct VectorSearchResult {
    ReelId reelId;
    float distance;
    float similarity;
};

// TDD 23.1: ReelRank depends on this abstraction, not on HNSW directly. Only the src/vindex/
// adapters (Phase 1) implement it over vector-db.
class VectorIndex {
  public:
    virtual void insert(const ReelId &id, const Embedding &embedding) = 0;

    virtual std::vector<VectorSearchResult> search(const Embedding &query, size_t k) const = 0;

    virtual size_t size() const = 0;

    virtual ~VectorIndex() = default;
};

} // namespace rr
