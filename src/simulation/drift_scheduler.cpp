#include "rr/simulation/drift_scheduler.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>

namespace rr {

namespace {

// hash01: deterministic UserId -> [0, 1) map, PINNED to the SplitMix64 finalizer so cohort
// membership is portable across compilers/platforms and every work package can reproduce it
// byte-for-byte. The body below is exactly rr::splitmix64(userId.value) (the same finalizer
// rr::Rng seeds with) followed by the canonical 53-bit-mantissa scaling; it is inlined here rather
// than calling infrastructure/random so the golden values are self-contained and cannot drift if
// that helper is ever refactored. DO NOT change these constants — cohort membership is a
// cross-package contract (the adaptation metrics split "drifted vs control" on it).
//
//   uint64_t x = static_cast<uint64_t>(userId.value) + 0x9E3779B97F4A7C15ULL;
//   x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
//   x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
//   x ^= x >> 31;
//   double u = static_cast<double>(x >> 11) * 0x1.0p-53;   // in [0, 1)
double hash01(UserId userId) {
    uint64_t x = static_cast<uint64_t>(userId.value) + 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return static_cast<double>(x >> 11) * 0x1.0p-53; // [0, 1)
}

} // namespace

DriftScheduler::DriftScheduler(const DriftConfig &config, const std::vector<Topic> &topics)
    : events_(config.events) {
    // Index the generated topic set by TopicId value for O(1) centre lookup during validation.
    std::unordered_map<uint32_t, const Embedding *> centreById;
    centreById.reserve(topics.size());
    for (const Topic &t : topics) {
        centreById.emplace(t.id.value, &t.centre);
    }

    targets_.reserve(events_.size());
    targetTopics_.reserve(events_.size());

    for (const DriftEvent &e : events_) {
        // --- Setup-error validation (D10: fail fast on bad config)
        // --------------------------------
        if (e.topicMix.empty()) {
            throw std::invalid_argument("rr::DriftScheduler: drift event has an empty topicMix");
        }
        if (!(e.cohortLo >= 0.0 && e.cohortLo <= 1.0) ||
            !(e.cohortHi >= 0.0 && e.cohortHi <= 1.0) || e.cohortLo >= e.cohortHi) {
            throw std::invalid_argument(
                "rr::DriftScheduler: cohort range must satisfy 0 <= cohortLo < cohortHi <= 1");
        }

        // --- Precompute the normalized target preference: normalize(sum_i weight_i * centre_i)
        // ---- Accumulated in float over the topic-centre floats so the result is bit-reproducible
        // and the unit test can recompute the expected vector by the identical operations.
        Embedding target;
        std::vector<TopicId> mixTopics;
        mixTopics.reserve(e.topicMix.size());

        for (const DriftTopicWeight &w : e.topicMix) {
            if (!std::isfinite(w.weight) || w.weight <= 0.0) {
                throw std::invalid_argument(
                    "rr::DriftScheduler: topic weight must be finite and > 0");
            }
            auto it = centreById.find(w.topic);
            if (it == centreById.end()) {
                throw std::invalid_argument(
                    "rr::DriftScheduler: topic id in mix is not present in the topic set");
            }
            const Embedding &centre = *it->second;
            if (target.empty()) {
                target.assign(centre.size(), 0.0f);
            }
            const float weight = static_cast<float>(w.weight);
            for (std::size_t d = 0; d < target.size(); ++d) {
                target[d] += weight * centre[d];
            }
            mixTopics.push_back(TopicId{w.topic});
        }

        // normalize throws std::invalid_argument if the mix summed to a ~zero vector (a degenerate
        // config); that surfaces as the same setup error, which is the correct fail-fast behaviour.
        normalize(target);

        targets_.push_back(std::move(target));
        targetTopics_.push_back(std::move(mixTopics));
    }
}

bool DriftScheduler::inCohort(UserId userId, double cohortLo, double cohortHi) {
    const double h = hash01(userId);
    return h >= cohortLo && h < cohortHi; // [lo, hi): lo inclusive, hi exclusive
}

bool DriftScheduler::everApplies(UserId userId) const {
    for (const DriftEvent &e : events_) {
        if (inCohort(userId, e.cohortLo, e.cohortHi)) {
            return true;
        }
    }
    return false;
}

uint32_t DriftScheduler::firstDriftInteraction() const {
    uint32_t first = 0;
    bool any = false;
    for (const DriftEvent &e : events_) {
        if (!any || e.atInteraction < first) {
            first = e.atInteraction;
            any = true;
        }
    }
    return first;
}

bool DriftScheduler::maybeApply(HiddenUserState &hidden, uint32_t totalInteractions) const {
    bool applied = false;
    // Config order with last-wins on collision: a later event firing at the same interaction for
    // the same user overwrites an earlier one (documented contract).
    for (std::size_t i = 0; i < events_.size(); ++i) {
        const DriftEvent &e = events_[i];
        if (e.atInteraction == totalInteractions &&
            inCohort(hidden.userId, e.cohortLo, e.cohortHi)) {
            hidden.hiddenPreference = targets_[i];     // precomputed, no recompute per call
            hidden.preferredTopics = targetTopics_[i]; // ground truth stays honest for inspection
            applied = true;
        }
    }
    return applied;
}

} // namespace rr
