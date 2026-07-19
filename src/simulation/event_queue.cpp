#include "rr/simulation/event_queue.hpp"

#include <cassert>

namespace rr {

const char *toString(EventType t) {
    switch (t) {
    case EventType::OpenApp:
        return "open_app";
    case EventType::RequestFeed:
        return "request_feed";
    case EventType::StartReel:
        return "start_reel";
    case EventType::FinishReel:
        return "finish_reel";
    case EventType::Interaction:
        return "interaction";
    case EventType::ExitApp:
        return "exit_app";
    case EventType::ReturnToApp:
        return "return_to_app";
    case EventType::PreferenceDrift:
        return "preference_drift";
    case EventType::ReelPublished:
        return "reel_published";
    }
    return "unknown";
}

// PINNED (D20): the same SplitMix64 finalizer family as rr::cohortHash01 / the drift cohort
// hash, over a fixed odd-constant mix of the three identity components. Golden values in
// tests/unit/event_queue_test.cpp are the tripwire — a change here is a cross-phase break.
uint64_t eventTieBreaker(UserId userId, EventType type, uint64_t perUserSeq) {
    uint64_t x = static_cast<uint64_t>(userId.value) * 0x9E3779B97F4A7C15ULL;
    x ^= (static_cast<uint64_t>(type) + 1) * 0xBF58476D1CE4E5B9ULL;
    x ^= perUserSeq * 0x94D049BB133111EBULL;
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

bool EventQueue::Later::operator()(const SimulationEvent &a, const SimulationEvent &b) const {
    // std::priority_queue pops the LARGEST under the comparator, so "Later" orders the earliest
    // event to the top: a is "later" than b when it should pop AFTER b.
    if (a.time != b.time) {
        return a.time > b.time;
    }
    if (a.deterministicTieBreaker != b.deterministicTieBreaker) {
        return a.deterministicTieBreaker > b.deterministicTieBreaker;
    }
    if (a.userId.value != b.userId.value) {
        return a.userId.value > b.userId.value;
    }
    return static_cast<uint8_t>(a.type) > static_cast<uint8_t>(b.type);
}

void EventQueue::push(const SimulationEvent &e) { heap_.push(e); }

bool EventQueue::empty() const { return heap_.empty(); }

std::size_t EventQueue::size() const { return heap_.size(); }

SimulationEvent EventQueue::pop() {
    assert(!heap_.empty());
    SimulationEvent e = heap_.top();
    heap_.pop();
    return e;
}

Timestamp EventQueue::nextTime() const {
    assert(!heap_.empty());
    return heap_.top().time;
}

} // namespace rr
