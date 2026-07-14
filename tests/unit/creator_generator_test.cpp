#include "rr/simulation/creator_generator.hpp"

#include <gtest/gtest.h>

#include <algorithm>

#include "rr/core/embedding.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/simulation/topic_generator.hpp"

using rr::Creator;
using rr::forkRng;
using rr::generateCreators;
using rr::generateTopics;
using rr::isValid;
using rr::SimulationConfig;
using rr::Topic;

namespace {
SimulationConfig smallConfig() {
    SimulationConfig c;
    c.seed = 7;
    c.topics = 8;
    c.creators = 40;
    c.dimensions = 16;
    return c;
}

std::vector<Topic> makeTopics(const SimulationConfig &c) {
    rr::Rng rng = forkRng(c.seed, "topics");
    return generateTopics(c, rng);
}
} // namespace

TEST(CreatorGeneratorTest, CountAndDenseIds) {
    SimulationConfig c = smallConfig();
    std::vector<Topic> topics = makeTopics(c);
    rr::Rng rng = forkRng(c.seed, "creators");
    std::vector<Creator> creators = generateCreators(c, topics, rng);
    ASSERT_EQ(creators.size(), 40u);
    for (uint32_t i = 0; i < creators.size(); ++i) {
        EXPECT_EQ(creators[i].id.value, i);
    }
}

TEST(CreatorGeneratorTest, StyleEmbeddingsValid) {
    SimulationConfig c = smallConfig();
    std::vector<Topic> topics = makeTopics(c);
    rr::Rng rng = forkRng(c.seed, "creators");
    std::vector<Creator> creators = generateCreators(c, topics, rng);
    for (const Creator &cr : creators) {
        EXPECT_EQ(cr.styleEmbedding.size(), c.dimensions);
        EXPECT_TRUE(isValid(cr.styleEmbedding));
    }
}

TEST(CreatorGeneratorTest, SpecialtiesInRangeDistinctAndValid) {
    SimulationConfig c = smallConfig();
    std::vector<Topic> topics = makeTopics(c);
    rr::Rng rng = forkRng(c.seed, "creators");
    std::vector<Creator> creators = generateCreators(c, topics, rng);
    for (const Creator &cr : creators) {
        ASSERT_GE(cr.topicSpecialties.size(), 1u);
        ASSERT_LE(cr.topicSpecialties.size(), 3u);
        std::vector<uint32_t> seen;
        for (rr::TopicId t : cr.topicSpecialties) {
            EXPECT_LT(t.value, c.topics);                                // valid topic id
            EXPECT_EQ(std::count(seen.begin(), seen.end(), t.value), 0); // distinct
            seen.push_back(t.value);
        }
    }
}

TEST(CreatorGeneratorTest, BaseQualityInUnitInterval) {
    SimulationConfig c = smallConfig();
    std::vector<Topic> topics = makeTopics(c);
    rr::Rng rng = forkRng(c.seed, "creators");
    std::vector<Creator> creators = generateCreators(c, topics, rng);
    for (const Creator &cr : creators) {
        EXPECT_GE(cr.baseQuality, 0.0f);
        EXPECT_LE(cr.baseQuality, 1.0f);
    }
}

TEST(CreatorGeneratorTest, ZeroCreatorsIsEmpty) {
    SimulationConfig c = smallConfig();
    c.creators = 0;
    std::vector<Topic> topics = makeTopics(c);
    rr::Rng rng = forkRng(c.seed, "creators");
    std::vector<Creator> creators = generateCreators(c, topics, rng);
    EXPECT_TRUE(creators.empty());
}
