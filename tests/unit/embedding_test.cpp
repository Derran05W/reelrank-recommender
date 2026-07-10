#include "rr/core/embedding.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

using rr::dot;
using rr::Embedding;
using rr::isValid;
using rr::normalize;
using rr::similarityFromEuclidean;

TEST(EmbeddingTest, NormalizeYieldsValid) {
    Embedding e{3.0f, 4.0f};
    normalize(e);
    EXPECT_TRUE(isValid(e));
    EXPECT_NEAR(e[0], 0.6f, 1e-5f);
    EXPECT_NEAR(e[1], 0.8f, 1e-5f);
}

TEST(EmbeddingTest, NormalizeZeroThrows) {
    Embedding e{0.0f, 0.0f, 0.0f};
    EXPECT_THROW(normalize(e), std::invalid_argument);
}

TEST(EmbeddingTest, NormalizeEmptyThrows) {
    Embedding e;
    EXPECT_THROW(normalize(e), std::invalid_argument);
}

TEST(EmbeddingTest, NormalizeNonFiniteThrows) {
    Embedding nan{1.0f, std::numeric_limits<float>::quiet_NaN()};
    EXPECT_THROW(normalize(nan), std::invalid_argument);
    Embedding inf{1.0f, std::numeric_limits<float>::infinity()};
    EXPECT_THROW(normalize(inf), std::invalid_argument);
}

TEST(EmbeddingTest, AlreadyNormalizedUnchanged) {
    Embedding e{0.6f, 0.8f};
    Embedding copy = e;
    normalize(e);
    EXPECT_NEAR(e[0], copy[0], 1e-6f);
    EXPECT_NEAR(e[1], copy[1], 1e-6f);
}

TEST(EmbeddingTest, DotOrthogonalAndSelf) {
    Embedding a{1.0f, 0.0f};
    Embedding b{0.0f, 1.0f};
    EXPECT_NEAR(dot(a, b), 0.0f, 1e-6f);
    Embedding c{3.0f, 4.0f};
    normalize(c);
    EXPECT_NEAR(dot(c, c), 1.0f, 1e-5f);
}

TEST(EmbeddingTest, DotSizeMismatchThrows) {
    Embedding a{1.0f, 2.0f};
    Embedding b{1.0f, 2.0f, 3.0f};
    EXPECT_THROW(dot(a, b), std::invalid_argument);
}

TEST(EmbeddingTest, SimilarityFromEuclidean) {
    EXPECT_NEAR(similarityFromEuclidean(0.0f), 1.0f, 1e-6f);
    EXPECT_NEAR(similarityFromEuclidean(std::sqrt(2.0f)), 0.0f, 1e-6f);
    EXPECT_NEAR(similarityFromEuclidean(2.0f), -1.0f, 1e-6f);
}

TEST(EmbeddingTest, IsValidRejectsOffNorm) {
    Embedding good{0.6f, 0.8f};
    EXPECT_TRUE(isValid(good));
    Embedding bad{0.6f, 0.9f}; // norm ~1.08, off by > 1e-4
    EXPECT_FALSE(isValid(bad));
    Embedding empty;
    EXPECT_FALSE(isValid(empty));
    Embedding nan{std::numeric_limits<float>::quiet_NaN()};
    EXPECT_FALSE(isValid(nan));
}
