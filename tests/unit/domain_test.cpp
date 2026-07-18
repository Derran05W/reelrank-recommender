#include "rr/domain/candidate.hpp"
#include "rr/domain/creator.hpp"
#include "rr/domain/ids.hpp"
#include "rr/domain/interaction.hpp"
#include "rr/domain/recommendation.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/simulation/hidden/hidden_user_state.hpp"

// Interface headers: included to prove each is self-contained and overridable.
#include "rr/learning/user_state_updater.hpp"
#include "rr/recommendation/candidate_generator.hpp"
#include "rr/recommendation/ranker.hpp"
#include "rr/recommendation/recommender.hpp"
#include "rr/recommendation/reranker.hpp"
#include "rr/recommendation/vector_index.hpp"

#include <gtest/gtest.h>

#include <type_traits>
#include <unordered_set>
#include <vector>

using namespace rr;

TEST(DomainTest, IdEqualityOrderingHash) {
    ReelId a{1};
    ReelId b{1};
    ReelId c{2};
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
    EXPECT_LT(a, c);

    std::unordered_set<ReelId> set;
    set.insert(a);
    set.insert(c);
    EXPECT_EQ(set.count(b), 1u); // a == b hash/equal
    EXPECT_EQ(set.size(), 2u);
}

TEST(DomainTest, DistinctIdTypesNotConvertible) {
    static_assert(!std::is_convertible_v<ReelId, UserId>);
    static_assert(!std::is_convertible_v<UserId, CreatorId>);
    SUCCEED();
}

TEST(DomainTest, AggregateConstruction) {
    Topic topic{TopicId{1}, Embedding{1.0f, 0.0f}};
    Creator creator{CreatorId{2}, Embedding{0.0f, 1.0f}, {TopicId{1}}, 0.8f};
    Reel reel{ReelId{3},
              CreatorId{2},
              Embedding{1.0f, 0.0f},
              0.9f,
              0.5f,
              30.0f,
              TopicId{1},
              {TopicId{4}},
              Timestamp{100},
              0,
              0,
              0,
              0,
              0,
              true};
    InteractionEvent ev{UserId{5}, ReelId{3}, CreatorId{2},   InteractionType::Like, 10.0f,
                        1.0f,      0.7f,      Timestamp{100}, SessionId{6}};
    Candidate cand{ReelId{3}, CandidateSource::VectorHNSW, 0.1f, 0.95f, 0.0f, {}};
    RankedReel ranked{ReelId{3}, 0.9f, 0, {CandidateSource::VectorHNSW}};
    RecommendationRequest req{UserId{5}, SessionId{6}, 10, 500, true, true, Timestamp{100}};
    HiddenUserState hidden{UserId{5}, Embedding{1.0f, 0.0f}};

    EXPECT_EQ(topic.id.value, 1u);
    EXPECT_EQ(creator.id.value, 2u);
    EXPECT_EQ(reel.id.value, 3u);
    EXPECT_EQ(ev.type, InteractionType::Like);
    EXPECT_EQ(cand.source, CandidateSource::VectorHNSW);
    EXPECT_EQ(ranked.sources.size(), 1u);
    EXPECT_TRUE(req.enableDiversity);
    EXPECT_EQ(hidden.userId.value, 5u);

    User user;
    user.id = UserId{5};
    user.seenReels.insert(ReelId{3});
    user.creatorAffinity[CreatorId{2}] = 0.5f;
    EXPECT_EQ(user.seenReels.count(ReelId{3}), 1u);
}

// Trivial mock proves VectorIndex is self-contained and overridable.
namespace {
class MockVectorIndex : public VectorIndex {
  public:
    void insert(const ReelId &, const Embedding &) override { ++count_; }
    std::vector<VectorSearchResult> search(const Embedding &, size_t) const override { return {}; }
    size_t size() const override { return count_; }

  private:
    size_t count_ = 0;
};
} // namespace

TEST(DomainTest, VectorIndexMockOverridable) {
    MockVectorIndex idx;
    idx.insert(ReelId{1}, Embedding{1.0f, 0.0f});
    idx.insert(ReelId{2}, Embedding{0.0f, 1.0f});
    EXPECT_EQ(idx.size(), 2u);
    EXPECT_TRUE(idx.search(Embedding{1.0f, 0.0f}, 5).empty());
    VectorIndex &base = idx;
    EXPECT_EQ(base.size(), 2u);
}
