#include "rr/learning/online_user_state_updater.hpp"

#include <gtest/gtest.h>

#include <deque>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/domain/ids.hpp"
#include "rr/domain/interaction.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/config.hpp"

// Unit tests for rr::OnlineUserStateUpdater (TDD 11.2, 11.3, 8.3). Every value here is
// hand-computed to float tolerance; the tests construct User::recentInteractions deques directly
// (rather than driving the simulator) so each update rule is exercised in isolation.

using namespace rr;

namespace {

// A minimal Reel carrying only the fields the updater reads (id + embedding). All other fields are
// irrelevant to preference learning and left default.
Reel makeReel(uint32_t id, Embedding embedding) {
    Reel r{};
    r.id = ReelId{id};
    r.creatorId = CreatorId{0};
    r.embedding = std::move(embedding);
    return r;
}

// A minimal InteractionEvent carrying the fields the updater reads: reelId (session-window
// embedding lookup), reward (long-term + session weighting), sessionId (current-session filter).
InteractionEvent makeEvent(uint32_t reelId, float reward, uint32_t sessionId) {
    InteractionEvent e{};
    e.userId = UserId{0};
    e.reelId = ReelId{reelId};
    e.creatorId = CreatorId{0};
    e.type = InteractionType::CompleteWatch;
    e.watchSeconds = 0.0f;
    e.watchRatio = 0.0f;
    e.reward = reward;
    e.timestamp = 0;
    e.sessionId = SessionId{sessionId};
    return e;
}

// A cold-started user: all three preference vectors set to `prior` (a unit vector).
User makeUser(const Embedding &prior) {
    User u{};
    u.id = UserId{0};
    u.estimatedPreference = prior;
    u.longTermPreference = prior;
    u.sessionPreference = prior;
    u.totalInteractions = 0;
    u.currentSessionLength = 0;
    return u;
}

LearningConfig defaultConfig() { return LearningConfig{}; }

constexpr float kTol = 1e-4f;

} // namespace

// --- Long-term update (TDD 11.2) --------------------------------------------------------------

// Positive reward moves the long-term vector toward the reel: cosine to the reel embedding rises.
// eta = 0.5, u = {1,0}, v = {0,1}, r = +1 => target = {0.5, 0.5} => normalize {sqrt(1/2)}^2.
TEST(OnlineUpdaterUnit, LongTermPositiveRewardIncreasesCosineToReel) {
    std::vector<Reel> reels{makeReel(0, {0.0f, 1.0f})};
    LearningConfig cfg = defaultConfig();
    cfg.longTermRate = 0.5;
    OnlineUserStateUpdater updater(reels, cfg);

    User user = makeUser({1.0f, 0.0f});
    const Embedding v = {0.0f, 1.0f};
    const float cosBefore = dot(user.longTermPreference, v); // = 0

    InteractionEvent e = makeEvent(/*reelId=*/0, /*reward=*/1.0f, /*sessionId=*/7);
    user.recentInteractions.push_back(e);
    updater.apply(user, reels[0], e);

    EXPECT_NEAR(user.longTermPreference[0], 0.70710678f, kTol);
    EXPECT_NEAR(user.longTermPreference[1], 0.70710678f, kTol);
    const float cosAfter = dot(user.longTermPreference, v);
    EXPECT_GT(cosAfter, cosBefore);
    EXPECT_TRUE(isValid(user.longTermPreference));
}

// Negative reward pushes the long-term vector away: cosine to the reel embedding drops below the
// starting value. Same setup, r = -1 => target = {0.5, -0.5} => normalize {sqrt(1/2), -sqrt(1/2)}.
TEST(OnlineUpdaterUnit, LongTermNegativeRewardDecreasesCosineToReel) {
    std::vector<Reel> reels{makeReel(0, {0.0f, 1.0f})};
    LearningConfig cfg = defaultConfig();
    cfg.longTermRate = 0.5;
    OnlineUserStateUpdater updater(reels, cfg);

    User user = makeUser({1.0f, 0.0f});
    const Embedding v = {0.0f, 1.0f};
    const float cosBefore = dot(user.longTermPreference, v); // = 0

    InteractionEvent e = makeEvent(0, -1.0f, 7);
    user.recentInteractions.push_back(e);
    updater.apply(user, reels[0], e);

    EXPECT_NEAR(user.longTermPreference[0], 0.70710678f, kTol);
    EXPECT_NEAR(user.longTermPreference[1], -0.70710678f, kTol);
    const float cosAfter = dot(user.longTermPreference, v);
    EXPECT_LT(cosAfter, cosBefore);
    EXPECT_TRUE(isValid(user.longTermPreference));
}

// --- Session recompute (TDD 11.3) -------------------------------------------------------------

// Hand-computed session vector on a 3-event current-session deque. lambda = 0.5.
// Chronological (oldest->newest): r0*v0, r1*v1, r2*v2 with the NEWEST getting weight lambda^0.
//   sum = 0.25*{1,0} + 0.5*{0,1} + 1*{1,0} = {1.25, 0.5} => normalize {0.928477, 0.371391}.
TEST(OnlineUpdaterUnit, SessionVectorLambdaDecayHandComputed) {
    std::vector<Reel> reels{makeReel(0, {1.0f, 0.0f}), makeReel(1, {0.0f, 1.0f}),
                            makeReel(2, {1.0f, 0.0f})};
    LearningConfig cfg = defaultConfig();
    cfg.sessionLambda = 0.5;
    OnlineUserStateUpdater updater(reels, cfg);

    User user = makeUser({1.0f, 0.0f});
    user.recentInteractions.push_back(makeEvent(0, 1.0f, 7)); // oldest -> weight lambda^2 = 0.25
    user.recentInteractions.push_back(makeEvent(1, 1.0f, 7)); //          -> weight lambda^1 = 0.5
    InteractionEvent newest = makeEvent(2, 1.0f, 7);          // newest -> weight lambda^0 = 1
    user.recentInteractions.push_back(newest);

    updater.apply(user, reels[2], newest);

    EXPECT_NEAR(user.sessionPreference[0], 0.928477f, kTol);
    EXPECT_NEAR(user.sessionPreference[1], 0.371391f, kTol);
    EXPECT_TRUE(isValid(user.sessionPreference));
}

// A single current-session event always gets weight lambda^0 = 1, so the session vector is exactly
// the normalized reel embedding (independent of lambda).
TEST(OnlineUpdaterUnit, SessionSingleEventEqualsNormalizedReel) {
    std::vector<Reel> reels{makeReel(0, {0.0f, 1.0f})};
    OnlineUserStateUpdater updater(reels, defaultConfig());

    User user = makeUser({1.0f, 0.0f});
    InteractionEvent only = makeEvent(0, 1.0f, 7);
    user.recentInteractions.push_back(only);

    updater.apply(user, reels[0], only);

    EXPECT_NEAR(user.sessionPreference[0], 0.0f, kTol);
    EXPECT_NEAR(user.sessionPreference[1], 1.0f, kTol);
}

// The session sum includes ONLY events whose sessionId matches the current interaction. A deque
// spanning two sessions must ignore the other-session event entirely (no exponent consumed).
// lambda = 0.5; current-session events {reelId1:v={0,1}, reelId2:v={1,0}} with the other-session
// reelId0:v={5,0} present but excluded.
//   sum = 0.5*{0,1} + 1*{1,0} = {1, 0.5} => normalize {0.894427, 0.447214}.
// If reelId0 leaked in (weight 0.25) the first component would be 1 + 1.25 = 2.25, not 1.
TEST(OnlineUpdaterUnit, SessionSumExcludesOtherSessionEvents) {
    std::vector<Reel> reels{makeReel(0, {5.0f, 0.0f}), makeReel(1, {0.0f, 1.0f}),
                            makeReel(2, {1.0f, 0.0f})};
    LearningConfig cfg = defaultConfig();
    cfg.sessionLambda = 0.5;
    OnlineUserStateUpdater updater(reels, cfg);

    User user = makeUser({1.0f, 0.0f});
    user.recentInteractions.push_back(makeEvent(0, 1.0f, /*sessionId=*/5)); // other session
    user.recentInteractions.push_back(makeEvent(1, 1.0f, /*sessionId=*/7)); // weight lambda^1
    InteractionEvent newest = makeEvent(2, 1.0f, /*sessionId=*/7);          // weight lambda^0
    user.recentInteractions.push_back(newest);

    updater.apply(user, reels[2], newest);

    EXPECT_NEAR(user.sessionPreference[0], 0.894427f, kTol);
    EXPECT_NEAR(user.sessionPreference[1], 0.447214f, kTol);
}

// --- Effective blend (TDD 8.3) ----------------------------------------------------------------

// Construct so that after apply longTerm = {1,0} (reward 0 leaves it) and session = {0,1}
// (one older current-session event with reward 1 pointing to {0,1}; the newest event is the
// reward-0 interaction). The blend uses the DEFAULT weights 0.65/0.35 => normalize({0.65, 0.35}).
TEST(OnlineUpdaterUnit, BlendUsesDefaultWeights) {
    std::vector<Reel> reels{makeReel(0, {0.0f, 1.0f}), makeReel(1, {1.0f, 0.0f})};
    LearningConfig cfg = defaultConfig();
    cfg.sessionLambda = 0.90;
    OnlineUserStateUpdater updater(reels, cfg);

    User user = makeUser({1.0f, 0.0f});
    user.recentInteractions.push_back(makeEvent(0, 1.0f, 7)); // older, weight lambda^1, v={0,1}
    InteractionEvent newest = makeEvent(1, 0.0f, 7);          // newest, reward 0 (no contribution)
    user.recentInteractions.push_back(newest);

    updater.apply(user, reels[1], newest);

    // longTerm unchanged by a reward-0 update.
    EXPECT_NEAR(user.longTermPreference[0], 1.0f, kTol);
    EXPECT_NEAR(user.longTermPreference[1], 0.0f, kTol);
    // session is the (normalized) reward-1 event embedding.
    EXPECT_NEAR(user.sessionPreference[0], 0.0f, kTol);
    EXPECT_NEAR(user.sessionPreference[1], 1.0f, kTol);

    Embedding expected = {0.65f, 0.35f};
    normalize(expected);
    EXPECT_NEAR(user.estimatedPreference[0], expected[0], kTol);
    EXPECT_NEAR(user.estimatedPreference[1], expected[1], kTol);
    EXPECT_TRUE(isValid(user.estimatedPreference));
}

// Same construction, but custom equal weights 0.5/0.5 => normalize({0.5, 0.5}) = {sqrt(1/2)}^2.
TEST(OnlineUpdaterUnit, BlendUsesCustomWeights) {
    std::vector<Reel> reels{makeReel(0, {0.0f, 1.0f}), makeReel(1, {1.0f, 0.0f})};
    LearningConfig cfg = defaultConfig();
    cfg.sessionLambda = 0.90;
    cfg.longTermWeight = 0.5;
    cfg.sessionWeight = 0.5;
    OnlineUserStateUpdater updater(reels, cfg);

    User user = makeUser({1.0f, 0.0f});
    user.recentInteractions.push_back(makeEvent(0, 1.0f, 7));
    InteractionEvent newest = makeEvent(1, 0.0f, 7);
    user.recentInteractions.push_back(newest);

    updater.apply(user, reels[1], newest);

    EXPECT_NEAR(user.estimatedPreference[0], 0.70710678f, kTol);
    EXPECT_NEAR(user.estimatedPreference[1], 0.70710678f, kTol);
}

// --- Degenerate-normalization fallbacks -------------------------------------------------------

// Long-term target with ~zero norm (eta=0.5, u=v={1,0}, r=-1 => target {0,0}) keeps the previous
// long-term vector unchanged.
TEST(OnlineUpdaterUnit, LongTermDegenerateKeepsPreviousVector) {
    std::vector<Reel> reels{makeReel(0, {1.0f, 0.0f})};
    LearningConfig cfg = defaultConfig();
    cfg.longTermRate = 0.5;
    OnlineUserStateUpdater updater(reels, cfg);

    User user = makeUser({1.0f, 0.0f});
    InteractionEvent e = makeEvent(0, -1.0f, 7);
    user.recentInteractions.push_back(e);

    updater.apply(user, reels[0], e);

    EXPECT_NEAR(user.longTermPreference[0], 1.0f, kTol);
    EXPECT_NEAR(user.longTermPreference[1], 0.0f, kTol);
    EXPECT_TRUE(isValid(user.longTermPreference));
}

// Session sum that cancels to ~zero (collinear reels, opposing rewards) falls back to the
// (updated) long-term vector. lambda=0.5: sum = 0.5*(+1)*{1,0} + 1*(-0.5)*{1,0} = {0,0}.
TEST(OnlineUpdaterUnit, SessionDegenerateFallsBackToLongTerm) {
    std::vector<Reel> reels{makeReel(0, {1.0f, 0.0f})};
    LearningConfig cfg = defaultConfig();
    cfg.sessionLambda = 0.5;
    OnlineUserStateUpdater updater(reels, cfg);

    User user = makeUser({1.0f, 0.0f});
    user.recentInteractions.push_back(makeEvent(0, 1.0f, 7)); // older, weight 0.5
    InteractionEvent newest = makeEvent(0, -0.5f, 7);         // newest, weight 1
    user.recentInteractions.push_back(newest);

    updater.apply(user, reels[0], newest);

    // Session is a copy of the (updated) long-term vector, so they are bit-identical.
    EXPECT_EQ(user.sessionPreference, user.longTermPreference);
    EXPECT_TRUE(isValid(user.sessionPreference));
}

// Blend that cancels to ~zero (equal weights, antipodal long-term/session) keeps the previous
// estimate. longTerm={1,0} (reward 0), session={-1,0} (older event reward -1 on v={1,0}); with
// weights 0.5/0.5 the blend target is {0,0}.
TEST(OnlineUpdaterUnit, BlendDegenerateKeepsPreviousEstimate) {
    std::vector<Reel> reels{makeReel(0, {1.0f, 0.0f}), makeReel(1, {1.0f, 0.0f})};
    LearningConfig cfg = defaultConfig();
    cfg.sessionLambda = 0.5;
    cfg.longTermWeight = 0.5;
    cfg.sessionWeight = 0.5;
    OnlineUserStateUpdater updater(reels, cfg);

    User user = makeUser({1.0f, 0.0f});
    user.estimatedPreference = {0.0f, 1.0f}; // sentinel: the previous estimate to be preserved
    user.recentInteractions.push_back(makeEvent(0, -1.0f, 7)); // older, weight 0.5, v={1,0}
    InteractionEvent newest = makeEvent(1, 0.0f, 7);           // newest, reward 0
    user.recentInteractions.push_back(newest);

    updater.apply(user, reels[1], newest);

    // longTerm and session are antipodal, so the blend degenerates and the estimate is preserved.
    EXPECT_NEAR(user.estimatedPreference[0], 0.0f, kTol);
    EXPECT_NEAR(user.estimatedPreference[1], 1.0f, kTol);
}

// --- Config master switch ---------------------------------------------------------------------

// With learning disabled, apply() is a no-op: all three preference vectors are unchanged.
TEST(OnlineUpdaterUnit, DisabledConfigLeavesStateUnchanged) {
    std::vector<Reel> reels{makeReel(0, {0.0f, 1.0f})};
    LearningConfig cfg = defaultConfig();
    cfg.enabled = false;
    OnlineUserStateUpdater updater(reels, cfg);

    User user = makeUser({1.0f, 0.0f});
    const Embedding est0 = user.estimatedPreference;
    const Embedding lt0 = user.longTermPreference;
    const Embedding sess0 = user.sessionPreference;

    InteractionEvent e = makeEvent(0, 1.0f, 7);
    user.recentInteractions.push_back(e);
    updater.apply(user, reels[0], e);

    EXPECT_EQ(user.estimatedPreference, est0);
    EXPECT_EQ(user.longTermPreference, lt0);
    EXPECT_EQ(user.sessionPreference, sess0);
}
