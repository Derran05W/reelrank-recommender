#include "rr/simulation/user_generator.hpp"

#include "rr/core/embedding.hpp"
#include "rr/domain/creator.hpp"
#include "rr/domain/hidden_user_state.hpp"
#include "rr/domain/ids.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace rr;

namespace {

// Self-contained topic factory (mirrors the unit test; no dependency on TopicGenerator).
std::vector<Topic> makeTopics(uint32_t count, uint32_t dim, uint64_t seed) {
    Rng rng(seed);
    std::vector<Topic> topics;
    topics.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        Embedding c(dim);
        for (uint32_t d = 0; d < dim; ++d) {
            c[d] = static_cast<float>(rng.gaussian());
        }
        normalize(c);
        topics.push_back(Topic{TopicId{i}, std::move(c)});
    }
    return topics;
}

SimulationConfig cfg(uint32_t users, uint32_t dim) {
    SimulationConfig c;
    c.users = users;
    c.dimensions = dim;
    return c;
}

// >= 20 distinct seeds for the determinism/validity sweep.
std::vector<uint64_t> sweepSeeds() {
    std::vector<uint64_t> seeds;
    for (int i = 0; i < 24; ++i) {
        seeds.push_back(1000ULL + static_cast<uint64_t>(i) * 7919ULL);
    }
    return seeds;
}

// Assert two generation runs are byte-identical: ids, latent preference (component-wise), preferred
// topics, and every behavioural trait exactly equal.
void expectIdentical(const GeneratedUsers &a, const GeneratedUsers &b) {
    ASSERT_EQ(a.users.size(), b.users.size());
    ASSERT_EQ(a.hiddenStates.size(), b.hiddenStates.size());
    for (size_t i = 0; i < a.hiddenStates.size(); ++i) {
        const HiddenUserState &x = a.hiddenStates[i];
        const HiddenUserState &y = b.hiddenStates[i];
        ASSERT_EQ(a.users[i].id, b.users[i].id);
        ASSERT_EQ(x.userId, y.userId);

        ASSERT_EQ(x.preferredTopics, y.preferredTopics);

        ASSERT_EQ(x.hiddenPreference.size(), y.hiddenPreference.size());
        for (size_t d = 0; d < x.hiddenPreference.size(); ++d) {
            ASSERT_EQ(x.hiddenPreference[d], y.hiddenPreference[d]) << "user " << i << " dim " << d;
        }

        ASSERT_EQ(x.preferenceConcentration, y.preferenceConcentration);
        ASSERT_EQ(x.exploreWillingness, y.exploreWillingness);
        ASSERT_EQ(x.avgSessionLength, y.avgSessionLength);
        ASSERT_EQ(x.likePropensity, y.likePropensity);
        ASSERT_EQ(x.sharePropensity, y.sharePropensity);
        ASSERT_EQ(x.durationTolerance, y.durationTolerance);
        ASSERT_EQ(x.preferenceStability, y.preferenceStability);
    }
}

} // namespace

// Same seed => byte-identical dataset, across >= 20 seeds. Also: every hiddenPreference is valid.
TEST(UserGeneratorProperty, DeterministicAndValidAcrossSeeds) {
    const uint32_t dim = 64;
    const uint32_t n = 200;
    auto topics = makeTopics(16, dim, 55);

    for (uint64_t seed : sweepSeeds()) {
        Rng r1(seed);
        Rng r2(seed);
        auto g1 = generateUsers(cfg(n, dim), topics, r1);
        auto g2 = generateUsers(cfg(n, dim), topics, r2);

        expectIdentical(g1, g2);
        for (const auto &h : g1.hiddenStates) {
            EXPECT_TRUE(isValid(h.hiddenPreference))
                << "seed " << seed << " user " << h.userId.value;
        }
    }
}

// Different seeds must actually flow randomness through: at least one user's hidden preference
// differs between two distinct seeds.
TEST(UserGeneratorProperty, DifferentSeedsProduceDifferentPreferences) {
    const uint32_t dim = 64;
    const uint32_t n = 200;
    auto topics = makeTopics(16, dim, 55);

    Rng ra(11);
    Rng rb(999983);
    auto ga = generateUsers(cfg(n, dim), topics, ra);
    auto gb = generateUsers(cfg(n, dim), topics, rb);

    bool anyDiff = false;
    for (uint32_t i = 0; i < n && !anyDiff; ++i) {
        const auto &pa = ga.hiddenStates[i].hiddenPreference;
        const auto &pb = gb.hiddenStates[i].hiddenPreference;
        for (size_t d = 0; d < pa.size(); ++d) {
            if (pa[d] != pb[d]) {
                anyDiff = true;
                break;
            }
        }
    }
    EXPECT_TRUE(anyDiff);
}

// Exit criterion: 10k users generate deterministically (same seed => identical).
TEST(UserGeneratorProperty, TenThousandUsersDeterministic) {
    const uint32_t dim = 64;
    const uint32_t n = 10000;
    auto topics = makeTopics(32, dim, 55);

    Rng r1(2024);
    Rng r2(2024);
    auto g1 = generateUsers(cfg(n, dim), topics, r1);
    auto g2 = generateUsers(cfg(n, dim), topics, r2);

    ASSERT_EQ(g1.hiddenStates.size(), n);
    expectIdentical(g1, g2);
    for (const auto &h : g1.hiddenStates) {
        ASSERT_TRUE(isValid(h.hiddenPreference));
    }
}
