#pragma once

#include <cstdint>
#include <deque>
#include <queue>
#include <vector>

#include "rr/domain/ids.hpp"
#include "rr/domain/recommendation.hpp"
#include "rr/infrastructure/clock.hpp"

namespace rr {

// Event-driven simulation core (V2 TDD 4.11, Phase 18; D20 is the binding determinism
// contract). The queue is a strict priority queue ordered by (time ascending,
// deterministicTieBreaker ascending): the tie-breaker is a PINNED SplitMix64-finalizer hash of
// (userId, eventType, perUserSeq) — golden-value tripwire-tested like cohortHash01 — so
// equal-timestamp pop order is a pure function of event identity, never of insertion order
// (the Tier 3 order-invariance acceptance). Single-threaded simulated concurrency (D13
// unchanged): event timestamps REPRESENT concurrency; processing is sequential.

enum class EventType : uint8_t {
    OpenApp,
    RequestFeed,
    StartReel,
    FinishReel,
    Interaction,
    ExitApp,
    ReturnToApp,
    PreferenceDrift,
    ReelPublished,
};

// Stable snake_case name for logs/digests (D6 conventions).
const char *toString(EventType t);

struct SimulationEvent {
    Timestamp time = 0;
    uint64_t deterministicTieBreaker = 0;
    UserId userId{};
    EventType type = EventType::OpenApp;
    // perUserSeq the tie-breaker was derived from — carried for event-log digests/debugging.
    uint64_t perUserSeq = 0;
};

// PINNED tie-breaker (D20): SplitMix64 finalizer over (userId.value, eventType, perUserSeq)
// mixed with fixed odd constants. DO NOT change the mixing — equal-timestamp semantics are a
// cross-phase contract with golden-value tests (event_queue_test.cpp).
uint64_t eventTieBreaker(UserId userId, EventType type, uint64_t perUserSeq);

// Per-user timeline state (V2 TDD 4.11).
struct UserTimeline {
    Timestamp nextEventAt = 0;
    bool online = false;
    SessionId activeSession{0};
    std::deque<RankedReel> prefetchedFeed{};
    // Monotone per-user event sequence number: incremented on every event SCHEDULED for this
    // user; feeds the tie-breaker so two events for one user can never tie.
    uint64_t nextSeq = 0;
};

// Deterministic priority queue over SimulationEvents. push() computes nothing — callers build
// events via schedule() helpers on the runner side; the queue only orders. Strict-weak order:
// (time, tieBreaker, userId, type) — the trailing keys make the order total even under a
// (vanishingly unlikely) tie-breaker collision, keeping determinism unconditional.
class EventQueue {
  public:
    void push(const SimulationEvent &e);
    bool empty() const;
    std::size_t size() const;
    // Pop the earliest event (asserts non-empty in Debug).
    SimulationEvent pop();
    // Earliest event time without popping (asserts non-empty in Debug).
    Timestamp nextTime() const;

  private:
    struct Later {
        bool operator()(const SimulationEvent &a, const SimulationEvent &b) const;
    };
    std::priority_queue<SimulationEvent, std::vector<SimulationEvent>, Later> heap_;
};

} // namespace rr
