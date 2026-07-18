#include "rr/evaluation/session_health_metrics.hpp"

#include <algorithm>
#include <cassert>

namespace rr {

namespace {
constexpr double kSecondsPerMinute = 60.0;
} // namespace

double sessionUtility(const SessionRecord &rec, const SessionDynamicsConfig &lambdas) {
    // V2 TDD §4.9: U_s = satSum − λ1·regretSum − λ2·harmfulFatigue − λ3·[exitType == Failure].
    const double failureIndicator = (rec.exitType == SessionExitType::Failure) ? 1.0 : 0.0;
    return static_cast<double>(rec.satisfactionSum) -
           lambdas.regretLambda * static_cast<double>(rec.regretSum) -
           lambdas.fatigueLambda * static_cast<double>(rec.harmfulFatigue) -
           lambdas.failureExitLambda * failureIndicator;
}

double perSessionMinute(double sum, double durationSeconds) {
    const double minutes = durationSeconds / kSecondsPerMinute;
    return minutes > 0.0 ? sum / minutes : 0.0;
}

double medianOf(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    if (n % 2 == 1) {
        return values[n / 2];
    }
    return (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

void ExitTypeCounts::tally(SessionExitType t) {
    switch (t) {
    case SessionExitType::Failure:
        ++failure;
        break;
    case SessionExitType::Satisfied:
        ++satisfied;
        break;
    case SessionExitType::Fatigue:
        ++fatigue;
        break;
    case SessionExitType::External:
        ++external;
        break;
    case SessionExitType::Regret:
        ++regret;
        break;
    case SessionExitType::RunEnded:
        ++runEnded;
        break;
    }
}

double SessionHealthReport::exitShare(SessionExitType t) const {
    const std::size_t closed = exits.closed();
    if (closed == 0) {
        return 0.0;
    }
    const double denom = static_cast<double>(closed);
    switch (t) {
    case SessionExitType::Failure:
        return static_cast<double>(exits.failure) / denom;
    case SessionExitType::Satisfied:
        return static_cast<double>(exits.satisfied) / denom;
    case SessionExitType::Fatigue:
        return static_cast<double>(exits.fatigue) / denom;
    case SessionExitType::External:
        return static_cast<double>(exits.external) / denom;
    case SessionExitType::Regret:
        return static_cast<double>(exits.regret) / denom;
    case SessionExitType::RunEnded:
        return 0.0; // RunEnded is not a closed type; use openShare().
    }
    return 0.0;
}

double SessionHealthReport::openShare() const {
    const std::size_t total = exits.total();
    return total > 0 ? static_cast<double>(exits.runEnded) / static_cast<double>(total) : 0.0;
}

SessionHealthMetrics::SessionHealthMetrics(std::size_t rounds, SessionDynamicsConfig lambdas)
    : lambdas_(lambdas), rounds_(rounds) {}

void SessionHealthMetrics::add(std::size_t round, const SessionRecord &rec) {
    assert(round < rounds_.size() && "session-health round index out of range");
    RoundAcc &r = rounds_[round];
    r.exits.tally(rec.exitType);
    exits_.tally(rec.exitType);

    // Next-session linkage (§4.9): a record whose user has already been seen follows an earlier
    // session, so its startingSatisfaction is a "next-session" starting satisfaction. Accumulated
    // online in add() (chronological) order so the sum is deterministic across same-seed runs.
    // Applies to every collected record (closed or, once a drain feeds them, RunEnded).
    const uint64_t userKey = rec.userId.value;
    auto [it, inserted] = userHasPrior_.try_emplace(userKey, uint8_t{1});
    if (!inserted) {
        r.nextStartSum += static_cast<double>(rec.startingSatisfaction);
        ++r.linked;
        nextStartSum_ += static_cast<double>(rec.startingSatisfaction);
        ++linked_;
    }

    // RunEnded is an OPEN session: counted above, but excluded from every aggregate below.
    if (rec.exitType == SessionExitType::RunEnded) {
        return;
    }

    const double duration = static_cast<double>(rec.durationSeconds);
    const double util = sessionUtility(rec, lambdas_);

    r.durationSum += duration;
    r.impressionsSum += static_cast<double>(rec.impressions);
    r.satisfactionSum += static_cast<double>(rec.satisfactionSum);
    r.regretSum += static_cast<double>(rec.regretSum);
    r.utilitySum += util;
    r.harmfulFatigueSum += static_cast<double>(rec.harmfulFatigue);
    r.durations.push_back(duration);

    durationSum_ += duration;
    impressionsSum_ += static_cast<double>(rec.impressions);
    satisfactionSum_ += static_cast<double>(rec.satisfactionSum);
    regretSum_ += static_cast<double>(rec.regretSum);
    utilitySum_ += util;
    harmfulFatigueSum_ += static_cast<double>(rec.harmfulFatigue);
    durations_.push_back(duration);
    ++closedSessions_;
}

SessionHealthRoundPoint SessionHealthMetrics::reduceRound(std::size_t round, const RoundAcc &acc) {
    SessionHealthRoundPoint p;
    p.round = round;
    p.exits = acc.exits;
    p.sessions = acc.exits.closed();
    p.openSessions = acc.exits.runEnded;

    const std::size_t closed = p.sessions;
    const double denom = closed > 0 ? static_cast<double>(closed) : 1.0;
    if (closed > 0) {
        p.meanDurationSeconds = acc.durationSum / denom;
        p.meanImpressions = acc.impressionsSum / denom;
        p.meanSessionUtility = acc.utilitySum / denom;
        p.harmfulFatigueMean = acc.harmfulFatigueSum / denom;
        p.earlyFailureExitRate = static_cast<double>(acc.exits.failure) / denom;
        p.naturalCompletionRate = static_cast<double>(acc.exits.satisfied) / denom;
    }
    p.medianDurationSeconds = medianOf(acc.durations);
    p.durationMinutes = acc.durationSum / kSecondsPerMinute;
    p.satisfactionPerMinute = perSessionMinute(acc.satisfactionSum, acc.durationSum);
    p.regretPerMinute = perSessionMinute(acc.regretSum, acc.durationSum);

    p.linkedSessions = acc.linked;
    p.nextSessionStartingSatisfaction =
        acc.linked > 0 ? acc.nextStartSum / static_cast<double>(acc.linked) : 0.0;
    return p;
}

SessionHealthReport SessionHealthMetrics::reduce() const {
    SessionHealthReport w;
    w.exits = exits_;
    w.sessions = exits_.closed();
    w.openSessions = exits_.runEnded;

    const std::size_t closed = w.sessions;
    const double denom = closed > 0 ? static_cast<double>(closed) : 1.0;
    if (closed > 0) {
        w.meanDurationSeconds = durationSum_ / denom;
        w.meanImpressions = impressionsSum_ / denom;
        w.meanSessionUtility = utilitySum_ / denom;
        w.harmfulFatigueMean = harmfulFatigueSum_ / denom;
        w.earlyFailureExitRate = static_cast<double>(exits_.failure) / denom;
        w.naturalCompletionRate = static_cast<double>(exits_.satisfied) / denom;
    }
    w.medianDurationSeconds = medianOf(durations_);
    w.durationMinutes = durationSum_ / kSecondsPerMinute;
    w.satisfactionPerMinute = perSessionMinute(satisfactionSum_, durationSum_);
    w.regretPerMinute = perSessionMinute(regretSum_, durationSum_);

    w.linkedSessions = linked_;
    w.nextSessionStartingSatisfaction =
        linked_ > 0 ? nextStartSum_ / static_cast<double>(linked_) : 0.0;

    w.byRound.reserve(rounds_.size());
    for (std::size_t r = 0; r < rounds_.size(); ++r) {
        w.byRound.push_back(reduceRound(r, rounds_[r]));
    }
    return w;
}

} // namespace rr
