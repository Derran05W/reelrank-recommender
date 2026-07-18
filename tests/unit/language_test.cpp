#include "rr/simulation/language.hpp"

#include <numeric>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

using rr::languageWeights;

TEST(LanguageTest, ThrowsOnZeroLanguages) {
    EXPECT_THROW(languageWeights(0), std::invalid_argument);
}

TEST(LanguageTest, SingleLanguageIsCertain) {
    auto w = languageWeights(1);
    ASSERT_EQ(w.size(), 1u);
    EXPECT_DOUBLE_EQ(w[0], 1.0);
}

TEST(LanguageTest, WeightsAreNormalized) {
    for (uint32_t n : {1u, 2u, 3u, 5u, 8u, 20u, 100u}) {
        auto w = languageWeights(n);
        ASSERT_EQ(w.size(), n);
        const double sum = std::accumulate(w.begin(), w.end(), 0.0);
        EXPECT_NEAR(sum, 1.0, 1e-12) << "languages=" << n;
        for (double x : w) {
            EXPECT_GT(x, 0.0);
        }
    }
}

TEST(LanguageTest, WeightsAreStrictlyDecreasing) {
    auto w = languageWeights(12);
    for (std::size_t i = 0; i + 1 < w.size(); ++i) {
        EXPECT_GT(w[i], w[i + 1]) << "not decreasing at index " << i;
    }
    // Language 0 dominates (the skewed global distribution's most common language).
    EXPECT_EQ(std::distance(w.begin(), std::max_element(w.begin(), w.end())), 0);
}

TEST(LanguageTest, FollowsZipfSOne) {
    // weight_i proportional to 1/(i+1): the ratio w[i]/w[j] must equal (j+1)/(i+1).
    auto w = languageWeights(8);
    for (std::size_t i = 0; i < w.size(); ++i) {
        for (std::size_t j = 0; j < w.size(); ++j) {
            const double expected = static_cast<double>(j + 1) / static_cast<double>(i + 1);
            EXPECT_NEAR(w[i] / w[j], expected, 1e-9) << "i=" << i << " j=" << j;
        }
    }
}
