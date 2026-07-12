#include "rr/evaluation/diversity_metrics.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <unordered_set>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/domain/ids.hpp"
#include "rr/domain/reel.hpp"

using namespace rr;

namespace {

// Build a minimal Reel with just the fields the diversity calculator reads (id, creator, topic,
// embedding). Everything else is left default; the calculator never touches it.
Reel makeReel(uint32_t id, uint32_t creator, uint32_t topic, Embedding emb) {
    Reel r{};
    r.id = ReelId{id};
    r.creatorId = CreatorId{creator};
    r.primaryTopic = TopicId{topic};
    r.embedding = std::move(emb);
    return r;
}

// Two-dimensional unit vectors give exact hand-computable cosines: ex.ex = 1, ey.ey = 1, ex.ey = 0.
const Embedding ex{1.0F, 0.0F};
const Embedding ey{0.0F, 1.0F};

std::vector<ReelId> feed(std::initializer_list<uint32_t> ids) {
    std::vector<ReelId> v;
    for (uint32_t id : ids) {
        v.push_back(ReelId{id});
    }
    return v;
}

const std::unordered_set<ReelId> kNothingSeen{};

} // namespace

// A hand-built 4-item feed with every metric computed by hand.
//   ids      : 0 1 2 3
//   creators : 1 1 1 2  -> distinct {1,2} = 2 ; HHI = (3/4)^2 + (1/4)^2 = 0.5625 + 0.0625 = 0.625
//   topics   : 5 5 6 7  -> distinct {5,6,7} = 3 ; HHI = (2/4)^2 + (1/4)^2 + (1/4)^2 = 0.375
//   embeds   : ex ex ey ey -> pairwise dots: (0,1)=1 (0,2)=0 (0,3)=0 (1,2)=0 (1,3)=0 (2,3)=1
//              sum = 2 over 6 pairs -> mean = 2/6
TEST(DiversityMetricsTest, FourItemFeedAllSixMetrics) {
    std::vector<Reel> reels = {makeReel(0, 1, 5, ex), makeReel(1, 1, 5, ex), makeReel(2, 1, 6, ey),
                               makeReel(3, 2, 7, ey)};

    const FeedDiversity d = computeFeedDiversity(feed({0, 1, 2, 3}), reels, kNothingSeen);

    EXPECT_EQ(d.feedSize, 4u);
    EXPECT_EQ(d.uniqueTopics, 3u);
    EXPECT_EQ(d.uniqueCreators, 2u);
    EXPECT_NEAR(d.topicConcentration, 0.375, 1e-12);
    EXPECT_NEAR(d.creatorConcentration, 0.625, 1e-12);
    EXPECT_NEAR(d.intraListSimilarity, 2.0 / 6.0, 1e-9);
    EXPECT_EQ(d.repeatCount, 0u);
    EXPECT_NEAR(d.repetitionRate, 0.0, 1e-12);
}

// Non-orthogonal embedding: a 45-degree unit vector gives cosine sqrt(1/2) against ex.
TEST(DiversityMetricsTest, IntraListSimilarityNonOrthogonal) {
    Embedding diag{1.0F, 1.0F};
    normalize(diag); // -> (1/sqrt2, 1/sqrt2)
    std::vector<Reel> reels = {makeReel(0, 1, 1, ex), makeReel(1, 2, 2, diag)};

    const FeedDiversity d = computeFeedDiversity(feed({0, 1}), reels, kNothingSeen);
    // Single pair: ex . diag = 1/sqrt2.
    EXPECT_NEAR(d.intraListSimilarity, std::sqrt(0.5), 1e-6);
}

// Empty feed: everything zero, no pairs, no shares (documented edge case).
TEST(DiversityMetricsTest, EmptyFeed) {
    std::vector<Reel> reels = {makeReel(0, 1, 1, ex)};
    const FeedDiversity d = computeFeedDiversity(feed({}), reels, kNothingSeen);

    EXPECT_EQ(d.feedSize, 0u);
    EXPECT_EQ(d.uniqueTopics, 0u);
    EXPECT_EQ(d.uniqueCreators, 0u);
    EXPECT_NEAR(d.intraListSimilarity, 0.0, 1e-12);
    EXPECT_NEAR(d.topicConcentration, 0.0, 1e-12);
    EXPECT_NEAR(d.creatorConcentration, 0.0, 1e-12);
    EXPECT_EQ(d.repeatCount, 0u);
    EXPECT_NEAR(d.repetitionRate, 0.0, 1e-12);
}

// One-item feed: 1 topic / 1 creator, HHI = 1.0 for both, no pairs so similarity is the defined 0.
TEST(DiversityMetricsTest, SingleItemFeed) {
    std::vector<Reel> reels = {makeReel(0, 3, 9, ex)};
    const FeedDiversity d = computeFeedDiversity(feed({0}), reels, kNothingSeen);

    EXPECT_EQ(d.feedSize, 1u);
    EXPECT_EQ(d.uniqueTopics, 1u);
    EXPECT_EQ(d.uniqueCreators, 1u);
    EXPECT_NEAR(d.intraListSimilarity, 0.0, 1e-12);
    EXPECT_NEAR(d.topicConcentration, 1.0, 1e-12);
    EXPECT_NEAR(d.creatorConcentration, 1.0, 1e-12);
    EXPECT_EQ(d.repeatCount, 0u);
}

// All-same-topic feed: uniqueTopics 1, topic HHI = 1.0 (single share of 1.0).
TEST(DiversityMetricsTest, AllSameTopic) {
    std::vector<Reel> reels = {makeReel(0, 1, 4, ex), makeReel(1, 2, 4, ey), makeReel(2, 3, 4, ex),
                               makeReel(3, 4, 4, ey)};
    const FeedDiversity d = computeFeedDiversity(feed({0, 1, 2, 3}), reels, kNothingSeen);

    EXPECT_EQ(d.uniqueTopics, 1u);
    EXPECT_NEAR(d.topicConcentration, 1.0, 1e-12);
    EXPECT_EQ(d.uniqueCreators, 4u);
    EXPECT_NEAR(d.creatorConcentration, 0.25, 1e-12); // 4 * (1/4)^2
}

// All-distinct feed: k=4 distinct topics/creators, HHI = 1/4 for both (uniform lower bound).
TEST(DiversityMetricsTest, AllDistinctTopicsAndCreators) {
    std::vector<Reel> reels = {makeReel(0, 10, 20, ex), makeReel(1, 11, 21, ey),
                               makeReel(2, 12, 22, ex), makeReel(3, 13, 23, ey)};
    const FeedDiversity d = computeFeedDiversity(feed({0, 1, 2, 3}), reels, kNothingSeen);

    EXPECT_EQ(d.uniqueTopics, 4u);
    EXPECT_EQ(d.uniqueCreators, 4u);
    EXPECT_NEAR(d.topicConcentration, 0.25, 1e-12);
    EXPECT_NEAR(d.creatorConcentration, 0.25, 1e-12);
}

// Repetition via the seen-set: a feed item already shown to this user is a repeat.
TEST(DiversityMetricsTest, RepetitionFromSeenSet) {
    std::vector<Reel> reels = {makeReel(0, 1, 1, ex), makeReel(1, 2, 2, ey)};
    const std::unordered_set<ReelId> seen{ReelId{0}};

    const FeedDiversity d = computeFeedDiversity(feed({0, 1}), reels, seen);
    EXPECT_EQ(d.repeatCount, 1u); // reel 0 was seen
    EXPECT_NEAR(d.repetitionRate, 0.5, 1e-12);
}

// Repetition via a within-feed duplicate: the SECOND occurrence of an id is the repeat.
TEST(DiversityMetricsTest, RepetitionFromWithinFeedDuplicate) {
    std::vector<Reel> reels = {makeReel(0, 1, 1, ex), makeReel(1, 2, 2, ey)};

    const FeedDiversity d = computeFeedDiversity(feed({0, 0}), reels, kNothingSeen);
    EXPECT_EQ(d.repeatCount, 1u); // first 0 is fresh, second 0 duplicates it
    EXPECT_NEAR(d.repetitionRate, 0.5, 1e-12);
    // A duplicate collapses distinct topics/creators to 1 each.
    EXPECT_EQ(d.uniqueTopics, 1u);
    EXPECT_EQ(d.uniqueCreators, 1u);
}

// Both seen and duplicated: every item is a repeat -> rate 1.0.
TEST(DiversityMetricsTest, RepetitionSeenAndDuplicatedCountsEvery) {
    std::vector<Reel> reels = {makeReel(0, 1, 1, ex)};
    const std::unordered_set<ReelId> seen{ReelId{0}};

    const FeedDiversity d = computeFeedDiversity(feed({0, 0}), reels, seen);
    EXPECT_EQ(d.repeatCount, 2u); // both occurrences of the seen reel
    EXPECT_NEAR(d.repetitionRate, 1.0, 1e-12);
}

// Accumulator means over two feeds within one round.
TEST(DiversityMetricsTest, AccumulatorMeansOverFeeds) {
    std::vector<Reel> reels = {makeReel(0, 1, 5, ex), makeReel(1, 1, 5, ex), makeReel(2, 1, 6, ey),
                               makeReel(3, 2, 7, ey)};

    DiversityAccumulator acc(1);
    // Feed A: 4 distinct-ish items -> uniqueTopics 3, uniqueCreators 2 (from the case above).
    acc.add(0, feed({0, 1, 2, 3}), reels, kNothingSeen);
    // Feed B: single item -> uniqueTopics 1, uniqueCreators 1.
    acc.add(0, feed({0}), reels, kNothingSeen);

    const DiversitySummary s = acc.roundSummary(0);
    EXPECT_EQ(s.feeds, 2u);
    EXPECT_NEAR(s.meanUniqueTopics, (3.0 + 1.0) / 2.0, 1e-12);
    EXPECT_NEAR(s.meanUniqueCreators, (2.0 + 1.0) / 2.0, 1e-12);
    // topic HHI: (0.375 + 1.0) / 2 ; creator HHI: (0.625 + 1.0) / 2.
    EXPECT_NEAR(s.meanTopicConcentration, (0.375 + 1.0) / 2.0, 1e-12);
    EXPECT_NEAR(s.meanCreatorConcentration, (0.625 + 1.0) / 2.0, 1e-12);
    // intra-list similarity: (2/6 from feed A) + (0 from the single-item feed B), / 2 feeds.
    EXPECT_NEAR(s.meanIntraListSimilarity, (2.0 / 6.0 + 0.0) / 2.0, 1e-9);
    EXPECT_EQ(s.totalItems, 5u);
    EXPECT_EQ(s.totalRepeats, 0u);
    EXPECT_NEAR(s.repetitionRate, 0.0, 1e-12);
}

// overall() pools ALL feeds across rounds (grand mean, not mean-of-means); repetition is
// totalRepeats/totalItems over the whole run.
TEST(DiversityMetricsTest, AccumulatorOverallPoolsAcrossRounds) {
    std::vector<Reel> reels = {makeReel(0, 1, 5, ex), makeReel(1, 2, 6, ey)};
    const std::unordered_set<ReelId> seen{ReelId{0}};

    DiversityAccumulator acc(2);
    acc.add(0, feed({0, 1}), reels, kNothingSeen); // 2 items, 0 repeats
    acc.add(1, feed({0, 1}), reels, seen);         // 2 items, 1 repeat (reel 0 seen)

    const DiversitySummary r0 = acc.roundSummary(0);
    const DiversitySummary r1 = acc.roundSummary(1);
    EXPECT_NEAR(r0.repetitionRate, 0.0, 1e-12);
    EXPECT_NEAR(r1.repetitionRate, 0.5, 1e-12);

    const DiversitySummary all = acc.overall();
    EXPECT_EQ(all.feeds, 2u);
    EXPECT_EQ(all.totalItems, 4u);
    EXPECT_EQ(all.totalRepeats, 1u);
    EXPECT_NEAR(all.repetitionRate, 1.0 / 4.0, 1e-12); // pooled 1 repeat over 4 items
    EXPECT_NEAR(all.meanUniqueTopics, 2.0, 1e-12);     // both feeds have 2 distinct topics
}

// Empty round summary: no feeds -> all-zero, no divide-by-zero.
TEST(DiversityMetricsTest, EmptyRoundSummaryIsZero) {
    DiversityAccumulator acc(1);
    const DiversitySummary s = acc.roundSummary(0);
    EXPECT_EQ(s.feeds, 0u);
    EXPECT_NEAR(s.meanUniqueTopics, 0.0, 1e-12);
    EXPECT_NEAR(s.repetitionRate, 0.0, 1e-12);
}

// Determinism: the same inputs produce bit-identical results (rng-free, order-independent sums).
TEST(DiversityMetricsTest, Deterministic) {
    std::vector<Reel> reels = {makeReel(0, 1, 5, ex), makeReel(1, 1, 5, ey), makeReel(2, 2, 6, ex),
                               makeReel(3, 3, 7, ey)};

    const FeedDiversity a = computeFeedDiversity(feed({0, 1, 2, 3}), reels, kNothingSeen);
    const FeedDiversity b = computeFeedDiversity(feed({0, 1, 2, 3}), reels, kNothingSeen);
    EXPECT_EQ(a.uniqueTopics, b.uniqueTopics);
    EXPECT_EQ(a.uniqueCreators, b.uniqueCreators);
    EXPECT_EQ(a.topicConcentration, b.topicConcentration);
    EXPECT_EQ(a.creatorConcentration, b.creatorConcentration);
    EXPECT_EQ(a.intraListSimilarity, b.intraListSimilarity);
    EXPECT_EQ(a.repeatCount, b.repeatCount);
}
