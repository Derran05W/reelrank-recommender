#include "rr/learning/tolerance_estimator.hpp"

#include <algorithm>
#include <cstddef>

#include "rr/domain/ids.hpp"

namespace rr {

namespace {

// --- EMA rates and decay constants (D24 named constants; the tuning surface is documented here,
// NOT on the config — no planned experiment varies them). Chosen so a consistent repetition-heavy
// or novelty-heavy stream CONVERGES the estimate within ~recentWindow (default 20) interactions:
// a per-event step of ~0.15 of the gap closes ~96% of it over 20 saturated events. ------------
constexpr double kRepetitionToleranceRate = 0.15; // EMA rate, repetition tolerance <- sat
constexpr double kNoveltyToleranceRate = 0.15;    // EMA rate, novelty tolerance <- sat
constexpr double kTopicFatigueRate = 0.30;        // EMA rate, topic fatigue <- fatigue signal
constexpr double kCreatorFatigueRate = 0.30;      // EMA rate, creator fatigue <- fatigue signal

// Per-event multiplicative decay of EVERY live fatigue entry ("decays otherwise"): an unrepeated
// topic/creator recovers by ~5% per subsequent interaction, so a saturated fatigue halves in ~13
// interactions of no recurrence. Entries that decay below the prune epsilon are erased, keeping
// both maps bounded to the small set of RECENTLY-fatigued topics/creators (so apply() stays cheap).
constexpr double kFatigueDecayPerEvent = 0.95;
constexpr double kFatiguePruneEpsilon = 0.02;

// Exit-after-repetition (plan task (b)): an observed session exit whose recent window was
// repetitive (repTopic at/above this threshold) is a strong "too much of the same" signal, so `sat`
// is forced to the negative extreme for the whole unified update path (lowers repetition tolerance,
// raises topic fatigue).
constexpr double kExitRepetitionThreshold = 0.30;

double clamp01(double v) { return std::clamp(v, 0.0, 1.0); }

// Per-event OBSERVABLE satisfaction proxy in [0,1]: the single signal every update EMAs toward.
// complete/rewatch/like/share/follow => 1 (positive); instant-skip / not-interested => 0
// (negative); partial-watch / impression => graded by the observed watchRatio. Deep-engagement
// cadence (comment / save / profile-visit) lifts it to 1 — mild positive evidence on the current
// topic (plan task (d)). An exit after a repetitive window forces it to 0 (plan "exit-after-
// repetition"); the caller passes repTopic so this stays a pure function of observables.
double computeSat(const InteractionEvent &e, double repTopic) {
    // Defensive init: the switch below covers every InteractionType enumerator, but GCC's
    // -Wmaybe-uninitialized cannot prove enum exhaustiveness (CI ubuntu arms build with GCC).
    double sat = 0.0;
    switch (e.type) {
    case InteractionType::CompleteWatch:
    case InteractionType::Rewatch:
    case InteractionType::Like:
    case InteractionType::Share:
    case InteractionType::FollowCreator:
        sat = 1.0;
        break;
    case InteractionType::InstantSkip:
    case InteractionType::NotInterested:
        sat = 0.0;
        break;
    case InteractionType::PartialWatch:
    case InteractionType::Impression:
        sat = clamp01(static_cast<double>(e.watchRatio));
        break;
    }
    // Deep-engagement cadence: an explicit comment/save/profile-visit is unambiguous positive
    // engagement, so treat the impression as fully satisfying regardless of the coarse type.
    if (e.commented || e.saved || e.profileVisited) {
        sat = 1.0;
    }
    // Exit after a repetitive window => strong intolerance evidence (drive sat negative).
    if (e.observedExitAfterImpression && repTopic >= kExitRepetitionThreshold) {
        sat = 0.0;
    }
    return sat;
}

// Decay every live fatigue entry toward 0 by kFatigueDecayPerEvent and drop negligible ones. Keeps
// the map to the recently-fatigued set (bounded work) and realizes the documented "decays
// otherwise" recovery for topics/creators the user has stopped seeing.
template <class Key> void decayFatigue(std::unordered_map<Key, float> &m) {
    for (auto it = m.begin(); it != m.end();) {
        const double decayed = static_cast<double>(it->second) * kFatigueDecayPerEvent;
        if (decayed < kFatiguePruneEpsilon) {
            it = m.erase(it);
        } else {
            it->second = static_cast<float>(decayed);
            ++it;
        }
    }
}

// EMA one fatigue entry toward `signal` at `rate`, then clamp/store (already decayed by
// decayFatigue this event). Creating the entry on demand starts it at 0 (neutral, unfatigued).
template <class Key>
void emaFatigue(std::unordered_map<Key, float> &m, Key key, double signal, double rate) {
    float &v = m[key];
    const double next = clamp01(static_cast<double>(v) + rate * (signal - static_cast<double>(v)));
    v = static_cast<float>(next);
}

} // namespace

ToleranceEstimator::ToleranceEstimator(const std::vector<Reel> &reels,
                                       const DiversityConfig &config)
    : reels_(reels), config_(config) {}

void ToleranceEstimator::apply(User &user, const Reel &reel,
                               const InteractionEvent &interaction) const {
    // --- Prior-window repetition context (observables only). The current interaction is the newest
    // entry of recentInteractions (call-site contract), so the prior window is every entry EXCEPT
    // the last. repTopic / repCreator = fraction of that window sharing the current reel's primary
    // topic / creator, each in [0,1]. Prior creator comes off the event; prior topic is a dense-id
    // reel lookup (range-guarded, mirroring FeatureExtractor's "a bad lookup never throws"). An
    // empty prior window (first-ever interaction) => 0 context => no tolerance move, fatigue stays.
    const TopicId curTopic = reel.primaryTopic;
    const CreatorId curCreator = reel.creatorId;
    std::size_t priorCount = 0;
    std::size_t sameTopic = 0;
    std::size_t sameCreator = 0;
    if (user.recentInteractions.size() >= 2) {
        const std::size_t n = user.recentInteractions.size();
        for (std::size_t i = 0; i + 1 < n; ++i) { // all but the last (== current) entry
            const InteractionEvent &e = user.recentInteractions[i];
            ++priorCount;
            if (e.creatorId == curCreator) {
                ++sameCreator;
            }
            if (e.reelId.value < reels_.size() && reels_[e.reelId.value].primaryTopic == curTopic) {
                ++sameTopic;
            }
        }
    }
    // No prior window (the first-ever interaction) => no observable evidence about repetition OR
    // novelty yet (we can't tell whether this topic is fresh or repeated), so leave every estimate
    // at its neutral default. From the second interaction on the window gives real context.
    if (priorCount == 0) {
        return;
    }
    const double repTopic = static_cast<double>(sameTopic) / static_cast<double>(priorCount);
    const double repCreator = static_cast<double>(sameCreator) / static_cast<double>(priorCount);

    const double sat = computeSat(interaction, repTopic);

    // (b) Global repetition tolerance: EMA toward sat, STEP SCALED by repTopic. Under repetition,
    //     completions/deep-engagement pull it up and skips/not-interested/exit-after-repeat pull it
    //     down; a purely novel event (repTopic 0) leaves it exactly where it was. The scaled step
    //     rate*repTopic is in [0,1], so the convex move stays in [0,1] with no external clamp.
    {
        const double step = kRepetitionToleranceRate * repTopic;
        const double cur = static_cast<double>(user.estimatedRepetitionTolerance);
        user.estimatedRepetitionTolerance = static_cast<float>(clamp01(cur + step * (sat - cur)));
    }

    // (c) Global novelty tolerance: EMA toward sat, STEP SCALED by (1 - repTopic).
    // Completing/liking
    //     a novel-topic item raises it; skipping one lowers it; repeated events barely move it.
    {
        const double step = kNoveltyToleranceRate * (1.0 - repTopic);
        const double cur = static_cast<double>(user.estimatedNoveltyTolerance);
        user.estimatedNoveltyTolerance = static_cast<float>(clamp01(cur + step * (sat - cur)));
    }

    // (a) Per-topic / per-creator fatigue. Decay ALL live entries first (recovery for anything the
    //     user has stopped seeing — the documented decay), then EMA the CURRENT topic/creator
    //     toward the fatigue signal repChannel * (1 - sat): a repeated channel met with a negative
    //     reaction rises toward fatigued, a completion relaxes it toward 0, a novel item
    //     (repChannel ~ 0) barely moves it. Bounded [0,1] by emaFatigue's clamp.
    decayFatigue(user.estimatedTopicFatigue);
    decayFatigue(user.estimatedCreatorFatigue);
    emaFatigue(user.estimatedTopicFatigue, curTopic, repTopic * (1.0 - sat), kTopicFatigueRate);
    emaFatigue(user.estimatedCreatorFatigue, curCreator, repCreator * (1.0 - sat),
               kCreatorFatigueRate);
}

} // namespace rr
