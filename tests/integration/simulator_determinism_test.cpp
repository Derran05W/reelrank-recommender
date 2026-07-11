#include "rr/simulation/simulator.hpp"

#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/simulation/dataset_generator.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <utility>
#include <vector>

using namespace rr;

// Event-stream determinism at the SIMULATOR level (TDD 24.6 / phase-3 task 4): two independently
// constructed Simulators driven by identically forked rngs over the same (user, reel) visitation
// sequence must produce byte-identical InteractionEvent streams (and identical outcomes). A
// different seed must produce a different stream (probabilistic sanity).

namespace {

SimulationConfig smallConfig() {
    SimulationConfig c;
    c.topics = 8;
    c.creators = 12;
    c.reels = 80;
    c.users = 8;
    c.dimensions = 32;
    return c;
}

const Creator &creatorFor(const GeneratedDataset &ds, const Reel &reel) {
    for (const auto &c : ds.creators) {
        if (c.id == reel.creatorId) {
            return c;
        }
    }
    ADD_FAILURE() << "no creator for reel " << reel.id.value;
    return ds.creators.front();
}

// A fixed, seed-independent visitation schedule: many rounds over all users, each round showing a
// different reel per user. Long enough to exercise session rotation and a wide range of outcomes.
std::vector<std::pair<size_t, size_t>> visitSchedule(const GeneratedDataset &ds) {
    std::vector<std::pair<size_t, size_t>> visits;
    const size_t rounds = 30;
    for (size_t round = 0; round < rounds; ++round) {
        for (size_t u = 0; u < ds.users.size(); ++u) {
            const size_t reelIdx = (round * ds.users.size() + u) % ds.reels.size();
            visits.emplace_back(u, reelIdx);
        }
    }
    return visits;
}

struct Stream {
    std::vector<InteractionEvent> events;
    std::vector<BehaviourOutcome> outcomes;
};

// Run one full simulation. Users/reels are private mutable copies so two runs never interfere;
// hidden state and creators are read-only and shared.
Stream runStream(const GeneratedDataset &ds, uint64_t behaviourSeed,
                 const std::vector<std::pair<size_t, size_t>> &visits) {
    std::vector<User> users = ds.users; // mutable per-run copies
    std::vector<Reel> reels = ds.reels;
    Simulator sim{BehaviourConfig{}, RewardConfig{}, forkRng(behaviourSeed, "behaviour"),
                  /*recentWindow=*/20, /*trendingHalfLifeSeconds=*/21600.0};

    Stream stream;
    stream.events.reserve(visits.size());
    stream.outcomes.reserve(visits.size());
    for (const auto &[u, r] : visits) {
        StepResult res = sim.step(users[u], ds.hiddenStates[u], reels[r], creatorFor(ds, reels[r]));
        stream.events.push_back(res.event);
        stream.outcomes.push_back(res.outcome);
    }
    return stream;
}

void expectIdenticalEvent(const InteractionEvent &a, const InteractionEvent &b) {
    EXPECT_EQ(a.userId, b.userId);
    EXPECT_EQ(a.reelId, b.reelId);
    EXPECT_EQ(a.creatorId, b.creatorId);
    EXPECT_EQ(a.type, b.type);
    EXPECT_EQ(a.watchSeconds, b.watchSeconds); // identical computation -> identical bits
    EXPECT_EQ(a.watchRatio, b.watchRatio);
    EXPECT_EQ(a.reward, b.reward);
    EXPECT_EQ(a.timestamp, b.timestamp);
    EXPECT_EQ(a.sessionId, b.sessionId);
}

void expectIdenticalOutcome(const BehaviourOutcome &a, const BehaviourOutcome &b) {
    EXPECT_EQ(a.baseAffinity, b.baseAffinity);
    EXPECT_EQ(a.behaviourScore, b.behaviourScore);
    EXPECT_EQ(a.instantSkip, b.instantSkip);
    EXPECT_EQ(a.completed, b.completed);
    EXPECT_EQ(a.rewatch, b.rewatch);
    EXPECT_EQ(a.liked, b.liked);
    EXPECT_EQ(a.shared, b.shared);
    EXPECT_EQ(a.followed, b.followed);
    EXPECT_EQ(a.notInterested, b.notInterested);
    EXPECT_EQ(a.watchRatio, b.watchRatio);
    EXPECT_EQ(a.watchSeconds, b.watchSeconds);
    EXPECT_EQ(a.primaryType, b.primaryType);
}

bool eventsDiffer(const InteractionEvent &a, const InteractionEvent &b) {
    return a.type != b.type || a.watchSeconds != b.watchSeconds || a.watchRatio != b.watchRatio ||
           a.reward != b.reward || a.timestamp != b.timestamp || a.sessionId != b.sessionId;
}

} // namespace

// Identical seed => byte-identical event and outcome streams. Swept over several seeds.
TEST(SimulatorDeterminismTest, IdenticalSeedProducesIdenticalStream) {
    for (uint64_t seed : {1ull, 2ull, 3ull, 4ull, 5ull, 6ull, 7ull}) {
        GeneratedDataset ds = generateDataset(smallConfig(), seed);
        const auto visits = visitSchedule(ds);

        Stream a = runStream(ds, seed, visits);
        Stream b = runStream(ds, seed, visits);

        ASSERT_EQ(a.events.size(), visits.size());
        ASSERT_EQ(a.events.size(), b.events.size());
        ASSERT_EQ(a.outcomes.size(), b.outcomes.size());
        for (size_t i = 0; i < a.events.size(); ++i) {
            expectIdenticalEvent(a.events[i], b.events[i]);
            expectIdenticalOutcome(a.outcomes[i], b.outcomes[i]);
        }
    }
}

// A different behaviour seed yields a different event stream (the simulation is genuinely
// stochastic under the seed, not constant). Probabilistic but overwhelmingly reliable over the full
// schedule: the behaviour rng and the session-target rng both diverge.
TEST(SimulatorDeterminismTest, DifferentSeedProducesDifferentStream) {
    GeneratedDataset ds = generateDataset(smallConfig(), /*seed=*/42);
    const auto visits = visitSchedule(ds);

    Stream a = runStream(ds, /*behaviourSeed=*/42, visits);
    Stream b = runStream(ds, /*behaviourSeed=*/424242, visits);

    ASSERT_EQ(a.events.size(), b.events.size());
    size_t differing = 0;
    for (size_t i = 0; i < a.events.size(); ++i) {
        if (eventsDiffer(a.events[i], b.events[i])) {
            ++differing;
        }
    }
    EXPECT_GT(differing, 0u) << "different seeds should yield a different event stream";
}
