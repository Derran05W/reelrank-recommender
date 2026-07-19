// Phase 18 event-queue unit tests (V2 TDD §4.11; D20 is the binding determinism contract).
//
// These are HARD assertions that hold TODAY: the EventQueue and the pinned eventTieBreaker are
// REAL in this tree (src/simulation/event_queue.cpp), independent of package A's runner. They are
// the foundation of the D20 determinism suite — the order-invariance and equal-timestamp
// acceptance criteria are discharged at this level (see tests/property/event_determinism_test.cpp
// for how the runner-level acceptance chains onto these queue-level proofs).
//
// TIE-BREAKER GOLDEN VALUES — harvesting procedure (the drift_scheduler cohortHash01 precedent):
// the pinned constants in TieBreakerGoldenValues below were produced by the print-then-pin method,
// NOT hand-derived. To (re)harvest after an intentional, reviewed change to the mixing constants
// in eventTieBreaker (a cross-phase break — see the D20 warning in event_queue.hpp):
//   1. Temporarily replace each pinned literal with a placeholder and add, in the loop,
//      `std::cerr << row.expected << " actual=" << actual << "\n";`
//   2. Build + run `--gtest_filter=EventQueueTest.TieBreakerGoldenValues`.
//   3. Copy the printed `actual=` values back into the table; the test then re-pins them.
// The values here were harvested by compiling the real src/simulation/event_queue.cpp against a
// grid over userIds {0,1,7,4242} x all 9 EventTypes x seqs {0,1,99} and selecting a representative
// spread. They are a cross-platform regression tripwire: if the SplitMix64-finalizer constants
// ever change, they break, exactly like cohortHash01's golden values.

#include "rr/simulation/event_queue.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

using namespace rr;

namespace {

// A realistic event: deterministicTieBreaker derived from identity exactly as the runner schedules
// it (eventTieBreaker(userId, type, perUserSeq)).
SimulationEvent ev(Timestamp t, uint32_t uid, EventType type, uint64_t seq) {
    SimulationEvent e;
    e.time = t;
    e.userId = UserId{uid};
    e.type = type;
    e.perUserSeq = seq;
    e.deterministicTieBreaker = eventTieBreaker(UserId{uid}, type, seq);
    return e;
}

// An event with an EXPLICITLY forced tie-breaker — used to construct (otherwise impossible)
// tie-breaker collisions so the trailing (userId, type) comparator keys can be exercised in
// isolation.
SimulationEvent forced(Timestamp t, uint64_t tb, uint32_t uid, EventType type) {
    SimulationEvent e;
    e.time = t;
    e.deterministicTieBreaker = tb;
    e.userId = UserId{uid};
    e.type = type;
    return e;
}

bool sameEvent(const SimulationEvent &a, const SimulationEvent &b) {
    return a.time == b.time && a.deterministicTieBreaker == b.deterministicTieBreaker &&
           a.userId.value == b.userId.value && a.type == b.type && a.perUserSeq == b.perUserSeq;
}

std::vector<SimulationEvent> drain(EventQueue &q) {
    std::vector<SimulationEvent> out;
    while (!q.empty()) {
        out.push_back(q.pop());
    }
    return out;
}

} // namespace

// --- Pinned tie-breaker golden values (the cross-phase tripwire) ---------------------------------

TEST(EventQueueTest, TieBreakerGoldenValues) {
    struct Row {
        uint32_t userId;
        EventType type;
        uint64_t seq;
        uint64_t expected;
    };
    // ~12 representative values spanning all four userIds, seven of the nine event types, and all
    // three seqs. Harvested from the real function (see the file header for the procedure).
    const std::vector<Row> golden = {
        {0, EventType::OpenApp, 0, 16481712997681181849ULL},
        {0, EventType::OpenApp, 1, 18348171785173989475ULL},
        {0, EventType::OpenApp, 99, 5562489710352892537ULL},
        {0, EventType::RequestFeed, 0, 392536317241979068ULL},
        {1, EventType::OpenApp, 0, 3852735613347767281ULL},
        {1, EventType::ExitApp, 0, 1454435403657136273ULL},
        {7, EventType::StartReel, 99, 106873256164440607ULL},
        {7, EventType::PreferenceDrift, 0, 18295825394675196732ULL},
        {7, EventType::ReelPublished, 1, 1730781279394552556ULL},
        {4242, EventType::OpenApp, 0, 15508678279313680078ULL},
        {4242, EventType::Interaction, 1, 18237608427515023479ULL},
        {4242, EventType::ReelPublished, 99, 2554078825271459480ULL},
    };
    for (const Row &r : golden) {
        EXPECT_EQ(eventTieBreaker(UserId{r.userId}, r.type, r.seq), r.expected)
            << "tie-breaker golden broke for userId=" << r.userId << " type=" << toString(r.type)
            << " seq=" << r.seq
            << " — an unreviewed change to the PINNED SplitMix64 mix (D20 cross-phase break), or "
               "re-harvest per the file header if the change was intentional";
    }
}

// Each of the three identity components must feed the mix: changing any one, holding the others
// fixed, must change the tie-breaker. (Cheap protection against a future edit that accidentally
// drops a component from the hash.)
TEST(EventQueueTest, TieBreakerDependsOnEveryComponent) {
    const uint64_t base = eventTieBreaker(UserId{7}, EventType::StartReel, 3);
    EXPECT_NE(base, eventTieBreaker(UserId{8}, EventType::StartReel, 3));  // userId varies
    EXPECT_NE(base, eventTieBreaker(UserId{7}, EventType::FinishReel, 3)); // type varies
    EXPECT_NE(base, eventTieBreaker(UserId{7}, EventType::StartReel, 4));  // seq varies
}

// --- Ordering: time dominates, tie-breaker orders equal times ------------------------------------

TEST(EventQueueTest, TimeDominatesOrdering) {
    EventQueue q;
    // The EARLIER event carries the LARGER tie-breaker; time must still win, proving time is the
    // primary key and the tie-breaker only ever settles EQUAL times.
    SimulationEvent early = forced(/*t=*/50, /*tb=*/0xFFFFFFFFFFFFFFFFULL, 9, EventType::OpenApp);
    SimulationEvent late = forced(/*t=*/100, /*tb=*/0x0000000000000000ULL, 0, EventType::OpenApp);
    q.push(late);
    q.push(early);
    EXPECT_EQ(q.nextTime(), 50u);
    SimulationEvent first = q.pop();
    EXPECT_EQ(first.time, 50u);
    EXPECT_TRUE(sameEvent(first, early));
    EXPECT_TRUE(sameEvent(q.pop(), late));
}

TEST(EventQueueTest, TieBreakerOrdersEqualTimes) {
    EventQueue q;
    // Same time; the smaller tie-breaker pops first regardless of push order.
    SimulationEvent small = forced(/*t=*/100, /*tb=*/10, 5, EventType::OpenApp);
    SimulationEvent big = forced(/*t=*/100, /*tb=*/20, 5, EventType::OpenApp);
    q.push(big);
    q.push(small);
    EXPECT_EQ(q.pop().deterministicTieBreaker, 10u);
    EXPECT_EQ(q.pop().deterministicTieBreaker, 20u);
}

// --- Insertion order NEVER matters: the queue-level order-invariance proof (D20) ---

TEST(EventQueueTest, InsertionOrderNeverMatters) {
    // A fixed event set with deliberate equal-time collisions (three at t=100, two at t=50) so the
    // tie-breaker path is exercised, plus distinct times so the time path is too. Every (time,
    // userId) pair is unique, so the (time, tieBreaker, userId, type) order is TOTAL over the set —
    // meaning the pop sequence is a pure function of the SET, never of insertion order. We prove it
    // by pushing every one of the 6! permutations and asserting an identical drain each time. This
    // is the queue-level discharge of the Tier 3 order-invariance acceptance (V2 §4.14 "changing
    // user iteration order must not materially change results").
    std::vector<SimulationEvent> base = {
        ev(100, 0, EventType::OpenApp, 0),     ev(100, 1, EventType::OpenApp, 0),
        ev(100, 7, EventType::RequestFeed, 2), ev(50, 4242, EventType::StartReel, 5),
        ev(200, 0, EventType::FinishReel, 9),  ev(50, 3, EventType::ExitApp, 1),
    };

    std::vector<int> idx = {0, 1, 2, 3, 4, 5};
    std::vector<SimulationEvent> canonical;
    {
        EventQueue q;
        for (int i : idx) {
            q.push(base[static_cast<std::size_t>(i)]);
        }
        canonical = drain(q);
    }
    ASSERT_EQ(canonical.size(), base.size());
    // The canonical drain must itself be non-decreasing in time (sanity on the comparator).
    for (std::size_t i = 1; i < canonical.size(); ++i) {
        EXPECT_LE(canonical[i - 1].time, canonical[i].time);
    }

    std::size_t permutations = 0;
    do {
        EventQueue q;
        for (int i : idx) {
            q.push(base[static_cast<std::size_t>(i)]);
        }
        const std::vector<SimulationEvent> got = drain(q);
        ASSERT_EQ(got.size(), canonical.size());
        for (std::size_t i = 0; i < got.size(); ++i) {
            EXPECT_TRUE(sameEvent(got[i], canonical[i]))
                << "permutation " << permutations << " diverged at pop " << i;
        }
        ++permutations;
    } while (std::next_permutation(idx.begin(), idx.end()));
    EXPECT_EQ(permutations, 720u); // 6! — every insertion order visited
}

// --- Trailing comparator keys break (hypothetical) tie-breaker collisions ------------------------

// The tie-breaker cannot collide for two real events (perUserSeq makes it injective per user), but
// the comparator still carries userId then type as trailing keys so the order is TOTAL even under a
// forced collision — determinism is unconditional, never probabilistic. Force equal (time,
// tieBreaker) and vary userId: the smaller userId pops first.
TEST(EventQueueTest, UserIdBreaksTieBreakerCollision) {
    EventQueue q;
    const uint64_t tb = 0x00C0FFEE00C0FFEEULL;
    q.push(forced(100, tb, 5, EventType::OpenApp));
    q.push(forced(100, tb, 2, EventType::OpenApp));
    q.push(forced(100, tb, 9, EventType::OpenApp));
    EXPECT_EQ(q.pop().userId.value, 2u);
    EXPECT_EQ(q.pop().userId.value, 5u);
    EXPECT_EQ(q.pop().userId.value, 9u);
}

// Force equal (time, tieBreaker, userId) and vary type: the smaller EventType enumerator pops first
// (OpenApp=0 < StartReel=2 < FinishReel=3).
TEST(EventQueueTest, TypeBreaksUserIdAndTieBreakerCollision) {
    EventQueue q;
    const uint64_t tb = 0x00C0FFEE00C0FFEEULL;
    q.push(forced(100, tb, 5, EventType::FinishReel));
    q.push(forced(100, tb, 5, EventType::OpenApp));
    q.push(forced(100, tb, 5, EventType::StartReel));
    EXPECT_EQ(q.pop().type, EventType::OpenApp);
    EXPECT_EQ(q.pop().type, EventType::StartReel);
    EXPECT_EQ(q.pop().type, EventType::FinishReel);
}

// --- pop / empty / size / nextTime semantics -----------------------------------------------------

TEST(EventQueueTest, PopEmptySizeSemantics) {
    EventQueue q;
    EXPECT_TRUE(q.empty());
    EXPECT_EQ(q.size(), 0u);

    q.push(ev(300, 1, EventType::OpenApp, 0));
    q.push(ev(100, 2, EventType::OpenApp, 0));
    q.push(ev(200, 3, EventType::OpenApp, 0));
    EXPECT_FALSE(q.empty());
    EXPECT_EQ(q.size(), 3u);

    // Pops come out in ascending time; size shrinks by one each pop.
    EXPECT_EQ(q.pop().time, 100u);
    EXPECT_EQ(q.size(), 2u);
    EXPECT_EQ(q.pop().time, 200u);
    EXPECT_EQ(q.size(), 1u);
    EXPECT_EQ(q.pop().time, 300u);
    EXPECT_EQ(q.size(), 0u);
    EXPECT_TRUE(q.empty());
}

TEST(EventQueueTest, NextTimePeeksWithoutPopping) {
    EventQueue q;
    q.push(ev(500, 1, EventType::OpenApp, 0));
    q.push(ev(250, 2, EventType::OpenApp, 0));
    EXPECT_EQ(q.nextTime(), 250u);
    EXPECT_EQ(q.size(), 2u); // nextTime is a peek — it does not consume
    EXPECT_EQ(q.nextTime(), 250u);
    EXPECT_EQ(q.pop().time, 250u);
    EXPECT_EQ(q.nextTime(), 500u); // now the later event is the front
    EXPECT_EQ(q.size(), 1u);
}
