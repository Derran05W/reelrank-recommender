// The ONLY test translation unit permitted to include vector-db headers (design decision D2).
// vector-db symbols live in the global namespace (no rr::). This proves the vendored dependency
// compiles, links, and answers a trivial query; deeper validation is Phase 1's job.
#include "hnsw_index.hpp"
#include "vector.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

Vector unitVector(std::vector<float> values) {
    double sumSq = 0.0;
    for (float v : values) {
        sumSq += static_cast<double>(v) * static_cast<double>(v);
    }
    const double norm = std::sqrt(sumSq);
    for (float &v : values) {
        v = static_cast<float>(v / norm);
    }
    return Vector(values);
}

} // namespace

TEST(VdbLinkTest, InsertAndSearch) {
    HNSWIndex index(8, 16, 200, 64);

    Vector v1 = unitVector({1, 0, 0, 0, 0, 0, 0, 0});
    Vector v2 = unitVector({0, 1, 0, 0, 0, 0, 0, 0});
    Vector v3 = unitVector({0, 0, 1, 0, 0, 0, 0, 0});
    index.insert(v1, "1");
    index.insert(v2, "2");
    index.insert(v3, "3");

    EXPECT_EQ(index.size(), 3u);

    auto results = index.search(v2, 1);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].first, "2");
    EXPECT_NEAR(results[0].second, 0.0f, 1e-4f);
}
