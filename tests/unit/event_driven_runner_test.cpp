// Event-driven runner UNIT tests (Phase 18, V2 TDD §4.11/4.12/4.14, D20). Package A owns the
// EventDrivenRunner; these tests exercise the pure, directly-testable pieces of it — the event-log
// digest fold, the baseline return-delay formula, and the within-timestamp processing-phase order
// (the "RequestFeed-first-within-group" contract package C's equal-timestamp tie-break test relies
// on). The full pipeline (all four §6 groups, determinism, independent timelines) is driven in the
// integration test event_runner_pipeline_test.cpp; the pinned tie-breaker golden values are
// package C's (event_queue_test.cpp).

#include "rr/evaluation/event_driven_runner.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/simulation/event_queue.hpp"

using namespace rr;

namespace {

EventLogEntry entry(Timestamp time, uint64_t tb, uint32_t user, EventType type, uint64_t seq) {
    return EventLogEntry{time, tb, user, static_cast<uint8_t>(type), seq};
}

// A small, representative event log (three users, several event kinds, ascending time).
std::vector<EventLogEntry> sampleLog() {
    return {
        entry(0, 111, 0, EventType::OpenApp, 0),     entry(0, 222, 0, EventType::RequestFeed, 1),
        entry(0, 333, 0, EventType::StartReel, 2),   entry(0, 333, 0, EventType::FinishReel, 2),
        entry(0, 333, 0, EventType::Interaction, 2), entry(5, 444, 1, EventType::OpenApp, 0),
        entry(12, 555, 0, EventType::ExitApp, 3),    entry(90, 666, 1, EventType::RequestFeed, 1),
    };
}

} // namespace

// ===== Event-log digest fold (D20 "same seed => identical event sequence" tripwire) =============

TEST(EventLogDigestTest, DeterministicForIdenticalLogs) {
    const auto a = sampleLog();
    const auto b = sampleLog();
    EXPECT_EQ(foldEventLog(a), foldEventLog(b));
}

TEST(EventLogDigestTest, EmptyLogIsTheSeedConstant) {
    // Documented init: the FNV-1a 64-bit offset basis (no entries mixed).
    EXPECT_EQ(foldEventLog({}), 14695981039346656037ULL);
}

TEST(EventLogDigestTest, OrderSensitive) {
    auto a = sampleLog();
    auto b = sampleLog();
    std::swap(b[1], b[6]); // reorder two entries -> a different stream
    EXPECT_NE(foldEventLog(a), foldEventLog(b));
}

TEST(EventLogDigestTest, FieldSensitive) {
    const auto base = sampleLog();
    // Perturb each identity field of one entry in turn; every change must flip the digest.
    {
        auto v = base;
        v[2].time += 1;
        EXPECT_NE(foldEventLog(base), foldEventLog(v));
    }
    {
        auto v = base;
        v[2].tieBreaker ^= 1ULL;
        EXPECT_NE(foldEventLog(base), foldEventLog(v));
    }
    {
        auto v = base;
        v[2].userId += 1;
        EXPECT_NE(foldEventLog(base), foldEventLog(v));
    }
    {
        auto v = base;
        v[2].type = static_cast<uint8_t>(EventType::ExitApp);
        EXPECT_NE(foldEventLog(base), foldEventLog(v));
    }
    {
        auto v = base;
        v[2].seq += 1;
        EXPECT_NE(foldEventLog(base), foldEventLog(v));
    }
}

// ===== Baseline return-delay formula (V2 §4.12, stream "scheduling") =============================

TEST(BaselineReturnDelayTest, DeterministicForSameStream) {
    SchedulingConfig sched; // defaults
    Rng a = forkRng(20260718, "scheduling");
    Rng b = forkRng(20260718, "scheduling");
    for (int i = 0; i < 200; ++i) {
        const double usage = 0.2 + 0.01 * i;
        EXPECT_EQ(baselineReturnDelay(a, sched, usage), baselineReturnDelay(b, sched, usage));
    }
}

TEST(BaselineReturnDelayTest, RespectsSixtySecondFloor) {
    SchedulingConfig sched;
    Rng rng = forkRng(1, "scheduling");
    for (int i = 0; i < 5000; ++i) {
        // Sweep usages incl. tiny (clamped) and huge (mean -> 0 => the floor engages).
        const double usage = (i % 7 == 0) ? 1e9 : (0.01 + 0.001 * (i % 500));
        EXPECT_GE(baselineReturnDelay(rng, sched, usage), 60u);
    }
}

TEST(BaselineReturnDelayTest, HugeUsageCollapsesToTheFloor) {
    SchedulingConfig sched;
    Rng rng = forkRng(7, "scheduling");
    // baselineDailyUsage this large drives mean/sd to ~0, so every draw floors at 60s exactly.
    for (int i = 0; i < 200; ++i) {
        EXPECT_EQ(baselineReturnDelay(rng, sched, 1e12), 60u);
    }
}

TEST(BaselineReturnDelayTest, HeavyUsersReturnSoonerOnAverage) {
    SchedulingConfig sched;
    const auto meanDelay = [&](double usage) {
        Rng rng = forkRng(99, "scheduling");
        double sum = 0.0;
        constexpr int kN = 4000;
        for (int i = 0; i < kN; ++i) {
            sum += static_cast<double>(baselineReturnDelay(rng, sched, usage));
        }
        return sum / kN;
    };
    // Mean scales as 1/usage, so a heavy user (usage 2.0) returns sooner than a light one (0.5).
    EXPECT_LT(meanDelay(2.0), meanDelay(0.5));
}

// ===== Within-timestamp processing phase (the equal-timestamp snapshot ordering) ================

TEST(EventProcessingPhaseTest, MapsEveryEventTypeToItsPhase) {
    EXPECT_EQ(eventProcessingPhase(EventType::OpenApp), 0);
    EXPECT_EQ(eventProcessingPhase(EventType::ReturnToApp), 0);
    EXPECT_EQ(eventProcessingPhase(EventType::RequestFeed), 1);
    EXPECT_EQ(eventProcessingPhase(EventType::StartReel), 2);
    EXPECT_EQ(eventProcessingPhase(EventType::FinishReel), 2);
    EXPECT_EQ(eventProcessingPhase(EventType::Interaction), 2);
    EXPECT_EQ(eventProcessingPhase(EventType::ExitApp), 2);
    EXPECT_EQ(eventProcessingPhase(EventType::PreferenceDrift), 2);
    EXPECT_EQ(eventProcessingPhase(EventType::ReelPublished), 2);
}

TEST(EventProcessingPhaseTest, RequestFeedBeforeConsumptionAndAfterArrivals) {
    // The core equal-timestamp contract (V2 §4.14 / D20): arrivals precede requests, requests
    // precede consumption — so every RequestFeed at a timestamp is handled before any counter-
    // mutating consumption at that timestamp.
    EXPECT_LT(eventProcessingPhase(EventType::OpenApp),
              eventProcessingPhase(EventType::RequestFeed));
    EXPECT_LT(eventProcessingPhase(EventType::ReturnToApp),
              eventProcessingPhase(EventType::RequestFeed));
    EXPECT_LT(eventProcessingPhase(EventType::RequestFeed),
              eventProcessingPhase(EventType::StartReel));
    EXPECT_LT(eventProcessingPhase(EventType::RequestFeed),
              eventProcessingPhase(EventType::ExitApp));
}

TEST(EventProcessingPhaseTest, GroupOrderingPutsAllRequestsBeforeAllConsumption) {
    // Construct a mixed batch of events sharing one timestamp (as the runner would face when
    // several users act at once) and reproduce the runner's phase-major processing order (stable
    // within a phase). Verify: no StartReel is processed before any RequestFeed, and no RequestFeed
    // before any arrival — the "RequestFeed-first-within-group" guarantee.
    std::vector<EventType> batch = {
        EventType::StartReel, EventType::RequestFeed, EventType::OpenApp, EventType::RequestFeed,
        EventType::StartReel, EventType::ReturnToApp, EventType::ExitApp, EventType::RequestFeed,
    };
    std::vector<EventType> processed;
    for (int phase = 0; phase <= 2; ++phase) {
        for (const EventType t : batch) {
            if (eventProcessingPhase(t) == phase) {
                processed.push_back(t);
            }
        }
    }
    const auto isReq = [](EventType t) { return t == EventType::RequestFeed; };
    const auto isConsumption = [](EventType t) {
        return t == EventType::StartReel || t == EventType::ExitApp;
    };
    const auto isArrival = [](EventType t) {
        return t == EventType::OpenApp || t == EventType::ReturnToApp;
    };
    const auto lastArrival = std::distance(
        processed.begin(), std::find_if(processed.rbegin(), processed.rend(), isArrival).base());
    const auto firstReq =
        std::distance(processed.begin(), std::find_if(processed.begin(), processed.end(), isReq));
    const auto lastReq = std::distance(
        processed.begin(), std::find_if(processed.rbegin(), processed.rend(), isReq).base());
    const auto firstConsumption = std::distance(
        processed.begin(), std::find_if(processed.begin(), processed.end(), isConsumption));
    EXPECT_LE(lastArrival, firstReq);     // every arrival before every request
    EXPECT_LE(lastReq, firstConsumption); // every request before every consumption
}
