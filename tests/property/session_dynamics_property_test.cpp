// Phase 16 session-dynamics property tests (>= 20 seeds, V2 TDD 4.8 / 7 / D8 / D17 / D19):
//   (i)   deterministic exit sampling — same seed => identical session boundaries + records;
//   (ii)  gate-off byte-identity — the P14 (latent-reactions) constructor's stepV2 is unchanged by
//         the Phase 16 additions (the new closedSession param + gated code are inert): identical
//         events with/without the pointer, no exit ever fires, no record is emitted;
//   (iii) fixed draw counts / stream alignment — the "session-exit" and "external-interruption"
//         streams are each consumed EXACTLY once per impression, proven by replaying a reference
//         fork of the stream in stepV2 call order and matching the observed exit flags bit-for-bit.

#include "rr/simulation/simulator.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "rr/domain/creator.hpp"
#include "rr/domain/interaction.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/simulation/dataset_generator.hpp"
#include "rr/simulation/hidden/hidden_session_state.hpp"

using namespace rr;

namespace {

constexpr uint32_t kNumSeeds = 24; // >= 20 (D17 property-test requirement)

SimulationConfig smallCfg() {
    SimulationConfig c;
    c.users = 20;
    c.reels = 80;
    c.creators = 12;
    c.topics = 8;
    c.dimensions = 16;
    return c;
}

RealismConfig realismOn() {
    RealismConfig r;
    r.contentV2 = true; // required for latent_reactions + populated hidden reel states / music
    r.latentReactions = true;
    r.sessionDynamics = true;
    return r;
}

struct StepLog {
    std::vector<uint8_t> exitFlags; // observedExitAfterImpression, in stepV2 call order
    std::vector<InteractionEvent> events;
    std::vector<SessionRecord> records; // closed sessions only (exit fired)
};

// Drive stepV2 over a dataset in a FIXED call order (round, user, feed slot). `sim` and `ds` are
// mutated; callers pass a freshly generated dataset per run so runs are independent.
StepLog drive(Simulator &sim, GeneratedDataset &ds, size_t rounds, size_t feedSize,
              bool captureRecords) {
    StepLog log;
    const size_t reelCount = ds.reels.size();
    uint64_t reqId = 0;
    for (size_t round = 0; round < rounds; ++round) {
        for (size_t u = 0; u < ds.users.size(); ++u) {
            const Timestamp rt = sim.now();
            ++reqId;
            for (size_t k = 0; k < feedSize; ++k) {
                const size_t idx = (u * 101 + round * feedSize + k) % reelCount;
                Reel &reel = ds.reels[idx];
                const Creator &creator = ds.creators[reel.creatorId.value];
                StepV2Inputs v2;
                v2.hiddenReel = &ds.hiddenReelStates[idx];
                v2.positionInFeed = static_cast<uint32_t>(k);
                v2.requestId = reqId;
                v2.requestTimestamp = rt;
                LatentReaction latent;
                SessionRecord closed{};
                const StepResult sr = sim.stepV2(ds.users[u], ds.hiddenStates[u], reel, creator, v2,
                                                 latent, captureRecords ? &closed : nullptr);
                log.exitFlags.push_back(sr.event.observedExitAfterImpression ? 1U : 0U);
                log.events.push_back(sr.event);
                if (captureRecords && sr.event.observedExitAfterImpression) {
                    log.records.push_back(closed);
                }
            }
        }
    }
    return log;
}

Simulator makeP16Sim(const SessionDynamicsConfig &sdc, uint64_t seed) {
    return Simulator(BehaviourConfig{}, BehaviourV2Config{}, sdc, RewardConfig{},
                     forkRng(seed, "behaviour"), forkRng(seed, "satisfaction"),
                     forkRng(seed, "session-exit"), forkRng(seed, "external-interruption"),
                     /*recentWindow=*/20, /*trendingHalfLifeSeconds=*/3600.0);
}

bool eventsEqual(const InteractionEvent &a, const InteractionEvent &b) {
    return a.userId == b.userId && a.reelId == b.reelId && a.type == b.type &&
           a.watchSeconds == b.watchSeconds && a.watchRatio == b.watchRatio &&
           a.reward == b.reward && a.timestamp == b.timestamp && a.sessionId == b.sessionId &&
           a.startTimestamp == b.startTimestamp && a.finishTimestamp == b.finishTimestamp &&
           a.dwellSeconds == b.dwellSeconds && a.replayCount == b.replayCount &&
           a.commented == b.commented && a.saved == b.saved &&
           a.profileVisited == b.profileVisited &&
           a.observedExitAfterImpression == b.observedExitAfterImpression;
}

void expectRecordsEqual(const SessionRecord &a, const SessionRecord &b) {
    EXPECT_EQ(a.userId, b.userId);
    EXPECT_EQ(a.sessionId, b.sessionId);
    EXPECT_EQ(a.exitType, b.exitType);
    EXPECT_EQ(a.impressions, b.impressions);
    EXPECT_FLOAT_EQ(a.durationSeconds, b.durationSeconds);
    EXPECT_FLOAT_EQ(a.satisfactionSum, b.satisfactionSum);
    EXPECT_FLOAT_EQ(a.regretSum, b.regretSum);
    EXPECT_FLOAT_EQ(a.harmfulFatigue, b.harmfulFatigue);
    EXPECT_FLOAT_EQ(a.sessionUtility, b.sessionUtility);
    EXPECT_FLOAT_EQ(a.startingSatisfaction, b.startingSatisfaction);
    EXPECT_EQ(a.startTime, b.startTime);
    EXPECT_EQ(a.endTime, b.endTime);
}

} // namespace

// ============================================================================================
//  (i) DETERMINISTIC EXIT SAMPLING: same seed => identical boundaries AND identical records.
// ============================================================================================
TEST(SessionDynamicsProperty, DeterministicExitBoundariesAndRecords) {
    const SimulationConfig sc = smallCfg();
    const RealismConfig realism = realismOn();
    const SessionDynamicsConfig sdc; // default operating point -> realistic mixed exits
    for (uint64_t seed = 1; seed <= kNumSeeds; ++seed) {
        SCOPED_TRACE("seed=" + std::to_string(seed));
        GeneratedDataset dsA = generateDataset(sc, realism, seed);
        GeneratedDataset dsB = generateDataset(sc, realism, seed);
        Simulator simA = makeP16Sim(sdc, seed);
        Simulator simB = makeP16Sim(sdc, seed);
        const StepLog a = drive(simA, dsA, /*rounds=*/6, /*feedSize=*/5, /*captureRecords=*/true);
        const StepLog b = drive(simB, dsB, 6, 5, true);

        ASSERT_EQ(a.exitFlags, b.exitFlags) << "session boundaries must be seed-deterministic";
        ASSERT_EQ(a.records.size(), b.records.size());
        for (size_t i = 0; i < a.records.size(); ++i) {
            SCOPED_TRACE("record " + std::to_string(i));
            expectRecordsEqual(a.records[i], b.records[i]);
        }
        // Sanity: the default operating point actually closes some sessions over 6 rounds.
        EXPECT_GT(a.records.size(), 0U) << "no exits fired — the arm would not exercise the model";
    }
}

// ============================================================================================
//  (ii) GATE-OFF BYTE-IDENTITY: the P14 (latent_reactions) constructor's stepV2 is inert to the
//       Phase 16 additions. Passing a closedSession pointer changes nothing, no exit fires, no
//       record is emitted, and two identically-seeded runs are byte-identical.
// ============================================================================================
TEST(SessionDynamicsProperty, GateOffP14PathUnchangedByPhase16) {
    const SimulationConfig sc = smallCfg();
    const RealismConfig realism = realismOn(); // dataset identical; only the CONSTRUCTOR differs
    for (uint64_t seed = 1; seed <= kNumSeeds; ++seed) {
        SCOPED_TRACE("seed=" + std::to_string(seed));
        const auto makeP14 = [&] {
            return Simulator(BehaviourConfig{}, BehaviourV2Config{}, RewardConfig{},
                             forkRng(seed, "behaviour"), forkRng(seed, "satisfaction"),
                             /*recentWindow=*/20, /*trendingHalfLifeSeconds=*/3600.0);
        };
        // Run 1: no closedSession pointer. Run 2: a non-null pointer every call.
        GeneratedDataset ds1 = generateDataset(sc, realism, seed);
        GeneratedDataset ds2 = generateDataset(sc, realism, seed);
        Simulator s1 = makeP14();
        Simulator s2 = makeP14();
        const StepLog noPtr = drive(s1, ds1, 6, 5, /*captureRecords=*/false);
        const StepLog withPtr = drive(s2, ds2, 6, 5, /*captureRecords=*/true);

        ASSERT_EQ(noPtr.events.size(), withPtr.events.size());
        for (size_t i = 0; i < noPtr.events.size(); ++i) {
            EXPECT_TRUE(eventsEqual(noPtr.events[i], withPtr.events[i]))
                << "the closedSession pointer must not perturb the P14 event stream (i=" << i
                << ")";
        }
        for (uint8_t f : withPtr.exitFlags) {
            EXPECT_EQ(f, 0U) << "gate-off stepV2 must never fire a probabilistic exit";
        }
        EXPECT_TRUE(withPtr.records.empty()) << "gate-off stepV2 must never emit a SessionRecord";
    }
}

// ============================================================================================
//  (iii-a) SESSION-EXIT STREAM ALIGNED: with every exit-logit STATE weight zeroed, P(exit) is a
//          constant sigmoid(exitBias) each impression, so a reference fork of "session-exit"
//          drawing one bernoulli(p) per impression reproduces the exit flags EXACTLY — one draw per
//          impression, stream-aligned regardless of feed/outcome.
// ============================================================================================
TEST(SessionDynamicsProperty, SessionExitStreamDrawnExactlyOncePerImpression) {
    const SimulationConfig sc = smallCfg();
    const RealismConfig realism = realismOn();
    SessionDynamicsConfig sdc;
    sdc.exitFatigueWeight = 0.0;
    sdc.exitRegretWeight = 0.0;
    sdc.exitPoorStreakWeight = 0.0;
    sdc.exitSatisfactionWeight = 0.0;
    sdc.exitInterruptionWeight = 0.0;
    sdc.externalInterruptionHazard = 0.0; // no interruption term either
    sdc.exitBias = -1.0;                  // p = sigmoid(-1) ~ 0.269, constant every impression
    const double p = 1.0 / (1.0 + std::exp(1.0));

    for (uint64_t seed = 1; seed <= kNumSeeds; ++seed) {
        SCOPED_TRACE("seed=" + std::to_string(seed));
        GeneratedDataset ds = generateDataset(sc, realism, seed);
        Simulator sim = makeP16Sim(sdc, seed);
        const StepLog log = drive(sim, ds, 6, 5, /*captureRecords=*/false);

        Rng refExit = forkRng(seed, "session-exit"); // same fork the Simulator consumes
        size_t exits = 0;
        for (size_t i = 0; i < log.exitFlags.size(); ++i) {
            const bool predicted = refExit.bernoulli(p);
            ASSERT_EQ(log.exitFlags[i], predicted ? 1U : 0U)
                << "session-exit stream desynchronized at impression " << i;
            exits += predicted ? 1U : 0U;
        }
        EXPECT_GT(exits, 0U); // the arm actually exercises exits
    }
}

// ============================================================================================
//  (iii-b) EXTERNAL-INTERRUPTION STREAM ALIGNED: exitBias -200 (P(exit)=0 without interruption) and
//          a huge interruption weight (P(exit)=1 with interruption) make exit fire IFF interruption
//          fired, so a reference fork of "external-interruption" drawing one bernoulli(hazard) per
//          impression reproduces the exit flags EXACTLY — one interruption draw per impression.
// ============================================================================================
TEST(SessionDynamicsProperty, InterruptionStreamDrawnExactlyOncePerImpression) {
    const SimulationConfig sc = smallCfg();
    const RealismConfig realism = realismOn();
    SessionDynamicsConfig sdc;
    sdc.exitFatigueWeight = 0.0;
    sdc.exitRegretWeight = 0.0;
    sdc.exitPoorStreakWeight = 0.0;
    sdc.exitSatisfactionWeight = 0.0;
    sdc.exitBias = -200.0; // sigmoid(-200) == 0.0 exactly -> bernoulli(0)=never
    sdc.exitInterruptionWeight =
        400.0; // interruption -> sigmoid(+200) == 1.0 -> bernoulli(1)=always
    const double hazard = 0.5;
    sdc.externalInterruptionHazard = hazard;

    for (uint64_t seed = 1; seed <= kNumSeeds; ++seed) {
        SCOPED_TRACE("seed=" + std::to_string(seed));
        GeneratedDataset ds = generateDataset(sc, realism, seed);
        Simulator sim = makeP16Sim(sdc, seed);
        const StepLog log = drive(sim, ds, 6, 5, /*captureRecords=*/false);

        Rng refIntr = forkRng(seed, "external-interruption");
        size_t interruptions = 0;
        for (size_t i = 0; i < log.exitFlags.size(); ++i) {
            const bool predicted =
                refIntr.bernoulli(hazard); // exit IFF interruption under this cfg
            ASSERT_EQ(log.exitFlags[i], predicted ? 1U : 0U)
                << "external-interruption stream desynchronized at impression " << i;
            interruptions += predicted ? 1U : 0U;
        }
        EXPECT_GT(interruptions, 0U);
    }
}
