// Phase 18 event-driven determinism suite (D20; V2 TDD §4.11/§4.14/§7 "same seed produces
// identical event sequence" + "equal-timestamp events have deterministic ordering"; Tier 3
// order-invariance acceptance minus the batch experiment).
//
// PACKAGE SPLIT / STATUS LEGEND. The event queue + pinned tie-breaker are REAL in this tree, so the
// queue-level proofs below assert HARD today. Package A owns EventDrivenRunner (a throwing stub
// here); every test that needs a completed event-mode RUN follows the house pending-integration
// idiom (P15/P16/P17): attempt the run, catch the stub's throw, and GTEST_SKIP with the exact
// post-merge assertions + any observed reference numbers. Auto-activates when A lands (the skip
// disappears the moment run() stops throwing). Per-test status:
//   SameSeedIdenticalDigest ....... SKIP-pending-A (needs a run + A's summary.json digest field)
//   OrderInvarianceDischarge ...... HARD now (queue-level) + SKIP-pending-A reinforcement
//   EqualTimestampSemantics ....... HARD now (queue-level) + SKIP-pending-A rerun-stability
//   EventModeRequiresGateStack .... HARD now (config validation throws; no runner needed)
//   CrossRunnerComparability ...... SKIP-pending-A (round-robin reference runs today; nums shown)
//
// DIGEST READ IS DEFENSIVE. A exposes an additive eventLogDigest/eventCount pair on
// ExperimentResult and (D22) in summary.json. This file NEVER references the not-yet-existing
// struct field (it would not compile against the scaffold); it reads summary.json by candidate key
// path so it compiles now and activates post-merge. If A's chosen key differs from the candidates
// in eventDigestFromSummary, the post-A assertion fails LOUDLY with an instruction to sync the path
// (see also scripts/check_event_digest.py and tests/golden/event-digest/README.md).

#include "rr/evaluation/experiment_runner.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/simulation/event_queue.hpp"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace rr;

namespace {

namespace fs = std::filesystem;

// --- Comparability margins (D20). NOT byte-identity: the event runner is explicitly NOT required
// to reproduce round-robin output — a documented statistical-comparability band replaces it. Named
// here so the integrator can recalibrate against real post-A numbers without hunting literals.
constexpr double kMaxMeanSatisfactionDelta = 0.15;
constexpr double kMaxCompletionRateDelta = 0.15;

std::string readFile(const fs::path &p) {
    std::ifstream in(p, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

nlohmann::json readJson(const fs::path &p) {
    std::ifstream in(p);
    return nlohmann::json::parse(in);
}

// The tiny full-gate stack every event-mode run needs: content_v2 -> latent_reactions ->
// session_dynamics (exits/returns schedule against the P16 machinery), hnsw_ranker. Round-robin by
// default; toEventMode() flips the scheduler on.
ExperimentConfig fullGateBase(uint64_t seed, uint32_t users, uint32_t reels, uint32_t dims) {
    ExperimentConfig c;
    c.simulation.seed = seed;
    c.simulation.users = users;
    c.simulation.reels = reels;
    c.simulation.creators = 40;
    c.simulation.topics = 8;
    c.simulation.dimensions = dims;
    c.simulation.interactionsPerUser = 20; // round-robin rounds; an OUTCOME under event mode
    c.recommendation.feedSize = 5;
    c.recommendation.vectorCandidates = 100;
    c.evaluation.oracleSampleRate = 0.1;
    c.evaluation.retrievalSampleRate = 0.0;
    c.algorithm = RecommendationAlgorithm::HnswRanker;
    c.realism.contentV2 = true;
    c.realism.latentReactions = true;
    c.realism.sessionDynamics = true;
    return c;
}

// Flip a full-gate config into event mode. horizonSeconds > 0 is mandatory (config validation);
// staggerSeconds controls the initial-OpenApp spread — stagger == 1 forces every open to t=0
// because Timestamp is integer seconds and the stagger draw is uniform([0, stagger)).
ExperimentConfig toEventMode(ExperimentConfig c, double horizonSeconds, double staggerSeconds) {
    c.simulation.scheduler = "event_queue";
    c.simulation.horizonSeconds = horizonSeconds;
    c.scheduling.openStaggerSeconds = staggerSeconds;
    return c;
}

// Run an experiment into its own temp root; returns the leaf result. Throws (stub) pre-A.
ExperimentResult runInto(const ExperimentConfig &cfg, const fs::path &root) {
    fs::remove_all(root);
    ExperimentRunner runner(cfg, root);
    return runner.run();
}

struct EventDigest {
    bool present = false;
    std::string digest; // normalized to text (A may emit hex string or integer)
    uint64_t eventCount = 0;
    bool countPresent = false;
};

// Read the event-log digest + event count from summary.json by CANDIDATE key path (A's exact key is
// unknown pre-merge). Ordered most-specific first; the README documents these for the integrator.
EventDigest eventDigestFromSummary(const fs::path &summaryPath) {
    EventDigest d;
    if (!fs::exists(summaryPath)) {
        return d;
    }
    const nlohmann::json j = readJson(summaryPath);

    auto asText = [](const nlohmann::json &v) -> std::optional<std::string> {
        if (v.is_string()) {
            return v.get<std::string>();
        }
        if (v.is_number_unsigned()) {
            return std::to_string(v.get<uint64_t>());
        }
        if (v.is_number_integer()) {
            return std::to_string(v.get<int64_t>());
        }
        return std::nullopt;
    };

    // Candidate summary.json key paths, most-specific first (A's exact key is unknown pre-merge;
    // kept in sync by hand with DIGEST_CANDIDATES in scripts/check_event_digest.py). An empty block
    // means top-level.
    struct Candidate {
        const char *block;
        const char *digestKey;
        const char *countKey;
    };
    const std::vector<Candidate> candidates = {
        {"", "event_log_digest", "event_count"},
        {"event_log", "digest", "event_count"},
        {"event_log", "digest", "count"},
        {"event_log", "digest", "events"},
        {"determinism", "event_log_digest", "event_count"},
        {"event_mode", "event_log_digest", "event_count"},
        {"event_mode", "digest", "event_count"},
    };
    for (const Candidate &c : candidates) {
        const nlohmann::json *node = &j;
        if (c.block[0] != '\0') {
            if (!j.contains(c.block)) {
                continue;
            }
            node = &j.at(c.block);
        }
        if (!node->is_object() || !node->contains(c.digestKey)) {
            continue;
        }
        const auto text = asText(node->at(c.digestKey));
        if (!text) {
            continue;
        }
        d.present = true;
        d.digest = *text;
        if (node->contains(c.countKey) && node->at(c.countKey).is_number()) {
            d.eventCount = node->at(c.countKey).get<uint64_t>();
            d.countPresent = true;
        }
        break;
    }
    return d;
}

// Defensive double read for the comparability means (stable keys, verified against results_writer).
std::optional<double> completionRate(const nlohmann::json &summary) {
    if (summary.contains("metrics") && summary.at("metrics").contains("completion_rate")) {
        return summary.at("metrics").at("completion_rate").get<double>();
    }
    return std::nullopt;
}
std::optional<double> meanSatisfaction(const nlohmann::json &summary) {
    if (summary.contains("welfare") &&
        summary.at("welfare").contains("mean_immediate_satisfaction")) {
        return summary.at("welfare").at("mean_immediate_satisfaction").get<double>();
    }
    return std::nullopt;
}

// A realistic scheduled event (tie-breaker derived exactly as the runner schedules it).
SimulationEvent ev(Timestamp t, uint32_t uid, EventType type, uint64_t seq) {
    SimulationEvent e;
    e.time = t;
    e.userId = UserId{uid};
    e.type = type;
    e.perUserSeq = seq;
    e.deterministicTieBreaker = eventTieBreaker(UserId{uid}, type, seq);
    return e;
}

bool sameEvent(const SimulationEvent &a, const SimulationEvent &b) {
    return a.time == b.time && a.deterministicTieBreaker == b.deterministicTieBreaker &&
           a.userId.value == b.userId.value && a.type == b.type;
}

std::vector<SimulationEvent> drain(EventQueue &q) {
    std::vector<SimulationEvent> out;
    while (!q.empty()) {
        out.push_back(q.pop());
    }
    return out;
}

} // namespace

// =================================================================================================
// (a) Same seed => identical event-log digest + event count + byte-identical welfare/session CSVs.
// V2 §7 "same seed produces identical event sequence". SKIP-pending-A.
// =================================================================================================
TEST(EventDeterminismTest, SameSeedIdenticalDigest) {
    const ExperimentConfig cfg =
        toEventMode(fullGateBase(/*seed=*/424242, /*users=*/200, /*reels=*/2000, /*dims=*/32),
                    /*horizonSeconds=*/6 * 3600.0, /*staggerSeconds=*/43200.0);

    ExperimentResult r1, r2;
    try {
        r1 = runInto(cfg, fs::path(::testing::TempDir()) / "rr_p18_digest_a");
        r2 = runInto(cfg, fs::path(::testing::TempDir()) / "rr_p18_digest_b");
    } catch (const std::exception &e) {
        GTEST_SKIP() << "PENDING PACKAGE A (EventDrivenRunner stub throws: " << e.what()
                     << "). Post-merge "
                        "asserts two same-seed event-mode runs yield EQUAL event_log_digest, EQUAL "
                        "event_count, and byte-identical welfare_metrics.csv + session_health.csv.";
    }

    const EventDigest d1 = eventDigestFromSummary(r1.directory / "summary.json");
    const EventDigest d2 = eventDigestFromSummary(r2.directory / "summary.json");
    ASSERT_TRUE(d1.present)
        << "event-log digest not found in summary.json under any candidate key — integrator must "
           "sync eventDigestFromSummary()/check_event_digest.py to A's actual key (see "
           "tests/golden/event-digest/README.md)";
    EXPECT_EQ(d1.digest, d2.digest) << "same seed must produce an identical event-log digest";
    EXPECT_EQ(d1.countPresent, d2.countPresent);
    EXPECT_EQ(d1.eventCount, d2.eventCount) << "same seed must produce an identical event count";

    // The two event-mode-only metric CSVs must be byte-identical across seeds (D20/D22).
    for (const char *csv : {"welfare_metrics.csv", "session_health.csv"}) {
        const fs::path p1 = r1.directory / csv;
        const fs::path p2 = r2.directory / csv;
        ASSERT_TRUE(fs::exists(p1)) << csv << " must be written under the full event gate stack";
        EXPECT_EQ(readFile(p1), readFile(p2))
            << csv << " must be byte-identical across same-seed runs";
    }
}

// =================================================================================================
// (b) Order invariance (D20 "permuting user initialization order => identical event log").
//
// DISCHARGE CHAIN. EventDrivenRunner exposes NO user-iteration/insertion-order knob: A seeds users
// in pinned ascending-UserId order and the ONLY structure where insertion order could otherwise
// leak into results is the priority queue. So the acceptance decomposes into two provable facts:
//   (1) the queue's pop order is a pure function of the event SET, not of push order — proven HARD
//       here (and exhaustively in event_queue_test.InsertionOrderNeverMatters); and
//   (2) A's seeding order is pinned (ascending UserId) — A's frozen contract.
// (1) AND (2) => permuting user initialization order cannot change the event log. The skip-guarded
// tail reinforces (1)+(2) end-to-end once A lands: rerunning the same seed reproduces the digest.
// =================================================================================================
TEST(EventDeterminismTest, OrderInvarianceDischarge) {
    // Model the runner's initial seeding: one OpenApp per user at its staggered open time. Push in
    // EVERY permutation and assert the drained order is invariant — i.e. whichever order A visits
    // users in, the queue emits the same sequence.
    std::vector<SimulationEvent> opens = {
        ev(0, 0, EventType::OpenApp, 0),  ev(0, 1, EventType::OpenApp, 0),
        ev(30, 2, EventType::OpenApp, 0), ev(0, 3, EventType::OpenApp, 0),
        ev(30, 4, EventType::OpenApp, 0),
    };
    std::vector<int> idx = {0, 1, 2, 3, 4};

    std::vector<SimulationEvent> canonical;
    {
        EventQueue q;
        for (int i : idx) {
            q.push(opens[static_cast<std::size_t>(i)]);
        }
        canonical = drain(q);
    }
    ASSERT_EQ(canonical.size(), opens.size());

    std::size_t perms = 0;
    do {
        EventQueue q;
        for (int i : idx) {
            q.push(opens[static_cast<std::size_t>(i)]);
        }
        const std::vector<SimulationEvent> got = drain(q);
        ASSERT_EQ(got.size(), canonical.size());
        for (std::size_t k = 0; k < got.size(); ++k) {
            EXPECT_TRUE(sameEvent(got[k], canonical[k]))
                << "user-seeding permutation " << perms << " diverged at pop " << k;
        }
        ++perms;
    } while (std::next_permutation(idx.begin(), idx.end()));
    EXPECT_EQ(perms, 120u); // 5! seeding orders

    // Reinforcement (SKIP-pending-A): with no order knob, "different order" is only observable at
    // the queue level (proven above); end-to-end, the same seed must reproduce the digest.
    const ExperimentConfig cfg = toEventMode(fullGateBase(7, 120, 1200, 16), 3 * 3600.0, 43200.0);
    ExperimentResult r1, r2;
    try {
        r1 = runInto(cfg, fs::path(::testing::TempDir()) / "rr_p18_order_a");
        r2 = runInto(cfg, fs::path(::testing::TempDir()) / "rr_p18_order_b");
    } catch (const std::exception &e) {
        GTEST_SKIP() << "queue-level order invariance PROVEN above (120 permutations). "
                        "End-to-end reinforcement PENDING PACKAGE A (stub throws: "
                     << e.what()
                     << "): post-merge asserts a same-seed rerun reproduces the event-log digest.";
    }
    const EventDigest d1 = eventDigestFromSummary(r1.directory / "summary.json");
    const EventDigest d2 = eventDigestFromSummary(r2.directory / "summary.json");
    ASSERT_TRUE(d1.present) << "integrator: sync the digest key path (see README)";
    EXPECT_EQ(d1.digest, d2.digest);
}

// =================================================================================================
// (c) Equal-timestamp semantics (D20 / V2 §4.14 / §7 "equal-timestamp events have deterministic
// ordering"). HARD queue-level proof + SKIP-pending-A whole-run rerun stability under forced
// simultaneity.
// =================================================================================================
TEST(EventDeterminismTest, EqualTimestampSemantics) {
    // Many events sharing time T=0 (the degenerate stagger==1 world where every user opens at t=0),
    // including two RequestFeeds at the same instant. Their pop order is a pure function of the
    // tie-breaker (then userId/type) — never of push order — so equal-time feed requests resolve
    // identically regardless of arrival order (the precondition for the T-epsilon snapshot: both
    // observe the same prior global state). Prove via full-permutation invariance at T=0.
    std::vector<SimulationEvent> simultaneous = {
        ev(0, 10, EventType::OpenApp, 0),     ev(0, 11, EventType::OpenApp, 0),
        ev(0, 10, EventType::RequestFeed, 1), ev(0, 11, EventType::RequestFeed, 1),
        ev(0, 12, EventType::OpenApp, 0),
    };
    std::vector<int> idx = {0, 1, 2, 3, 4};
    std::vector<SimulationEvent> canonical;
    {
        EventQueue q;
        for (int i : idx) {
            q.push(simultaneous[static_cast<std::size_t>(i)]);
        }
        canonical = drain(q);
    }
    for (const SimulationEvent &e : canonical) {
        EXPECT_EQ(e.time, 0u); // all at the same instant
    }
    std::size_t perms = 0;
    do {
        EventQueue q;
        for (int i : idx) {
            q.push(simultaneous[static_cast<std::size_t>(i)]);
        }
        const std::vector<SimulationEvent> got = drain(q);
        for (std::size_t k = 0; k < got.size(); ++k) {
            EXPECT_TRUE(sameEvent(got[k], canonical[k]))
                << "equal-timestamp permutation " << perms << " diverged at pop " << k;
        }
        ++perms;
    } while (std::next_permutation(idx.begin(), idx.end()));
    EXPECT_EQ(perms, 120u);

    // Whole-run stability under MASSIVE forced simultaneity (SKIP-pending-A): stagger == 1
    // collapses every initial open to t=0, so the equal-timestamp ordering path is exercised across
    // the whole population. Determinism => rerunning the same seed reproduces the digest
    // bit-for-bit.
    const ExperimentConfig cfg =
        toEventMode(fullGateBase(99, 150, 1500, 16), 6 * 3600.0, /*staggerSeconds=*/1.0);
    ExperimentResult r1, r2;
    try {
        r1 = runInto(cfg, fs::path(::testing::TempDir()) / "rr_p18_eqts_a");
        r2 = runInto(cfg, fs::path(::testing::TempDir()) / "rr_p18_eqts_b");
    } catch (const std::exception &e) {
        GTEST_SKIP() << "equal-timestamp ORDER determinism PROVEN above (120 permutations at T=0). "
                        "Whole-run stability under stagger==1 PENDING PACKAGE A (stub throws: "
                     << e.what()
                     << "): post-merge asserts the digest is stable across reruns when the entire "
                        "population opens simultaneously at t=0. If A exposes a per-timestamp "
                        "snapshot count, assert equal-T feed requests read one snapshot here.";
    }
    const EventDigest d1 = eventDigestFromSummary(r1.directory / "summary.json");
    const EventDigest d2 = eventDigestFromSummary(r2.directory / "summary.json");
    ASSERT_TRUE(d1.present) << "integrator: sync the digest key path (see README)";
    EXPECT_EQ(d1.digest, d2.digest) << "stagger==1 (all opens at t=0) must still be reproducible";
    EXPECT_EQ(d1.eventCount, d2.eventCount);
}

// =================================================================================================
// (d) Event mode requires the full gate stack + a positive horizon. HARD now — config validation
// throws at load, no runner needed. This is a DETERMINISM prerequisite: exits/returns schedule
// against the P16 session machinery, so event mode without session_dynamics has no deterministic
// lifecycle to reproduce. (The config-surface angle is also covered by config_test.cpp; this states
// the invariant from the D20 determinism contract's side and pins the exact configs the suite
// uses.)
// =================================================================================================
TEST(EventDeterminismTest, EventModeRequiresGateStack) {
    using nlohmann::json;

    // event_queue with NO session_dynamics (nor its prerequisites) => reject.
    json noGates = {{"simulation", {{"scheduler", "event_queue"}, {"horizon_seconds", 3600.0}}}};
    EXPECT_THROW(noGates.get<ExperimentConfig>(), std::invalid_argument);

    // event_queue with a partial stack (content_v2 + latent, but session_dynamics off) => reject.
    json partial = {{"simulation", {{"scheduler", "event_queue"}, {"horizon_seconds", 3600.0}}},
                    {"realism", {{"content_v2", true}, {"latent_reactions", true}}}};
    EXPECT_THROW(partial.get<ExperimentConfig>(), std::invalid_argument);

    // event_queue with the full stack but a non-positive horizon => reject.
    json noHorizon = {
        {"simulation", {{"scheduler", "event_queue"}, {"horizon_seconds", 0.0}}},
        {"realism",
         {{"content_v2", true}, {"latent_reactions", true}, {"session_dynamics", true}}}};
    EXPECT_THROW(noHorizon.get<ExperimentConfig>(), std::invalid_argument);

    // Unknown scheduler name => reject (only round_robin / event_queue are deterministic paths).
    json bogus = {{"simulation", {{"scheduler", "priority"}}}};
    EXPECT_THROW(bogus.get<ExperimentConfig>(), std::invalid_argument);

    // The full stack + positive horizon is ACCEPTED — this is exactly the shape the suite's event
    // configs use, so this pins that they load cleanly.
    json ok = {{"algorithm", "hnsw_ranker"},
               {"simulation", {{"scheduler", "event_queue"}, {"horizon_seconds", 21600.0}}},
               {"realism",
                {{"content_v2", true}, {"latent_reactions", true}, {"session_dynamics", true}}}};
    ExperimentConfig c;
    EXPECT_NO_THROW(c = ok.get<ExperimentConfig>());
    EXPECT_EQ(c.simulation.scheduler, "event_queue");
    EXPECT_GT(c.simulation.horizonSeconds, 0.0);
    EXPECT_TRUE(c.realism.sessionDynamics);
}

// =================================================================================================
// (e) Cross-runner comparability (D20): a DEGENERATE event run (all users open ~t=0, generous
// horizon) is STATISTICALLY comparable to round-robin on engagement/welfare means — NOT byte-
// identical. Margins are named constants for integrator calibration. The round-robin reference runs
// TODAY, so its observed means are reported in the skip message; the event arm is SKIP-pending-A.
// =================================================================================================
TEST(EventDeterminismTest, CrossRunnerComparability) {
    const ExperimentConfig base =
        fullGateBase(/*seed=*/2024, /*users=*/120, /*reels=*/1200, /*dims=*/16);

    // Round-robin reference arm (runs today).
    const nlohmann::json rrSummary = [&] {
        const ExperimentResult r = runInto(base, fs::path(::testing::TempDir()) / "rr_p18_xrun_rr");
        return readJson(r.directory / "summary.json");
    }();
    const std::optional<double> rrCompletion = completionRate(rrSummary);
    const std::optional<double> rrSatisfaction = meanSatisfaction(rrSummary);
    ASSERT_TRUE(rrCompletion.has_value())
        << "round-robin summary.json missing metrics.completion_rate";
    ASSERT_TRUE(rrSatisfaction.has_value())
        << "round-robin summary.json missing welfare.mean_immediate_satisfaction (latent gate on?)";

    // Degenerate event arm: stagger==1 (all open ~t=0), generous horizon so per-user consumption is
    // comparable to the round-robin interaction budget. SKIP-pending-A.
    const ExperimentConfig eventCfg =
        toEventMode(base, /*horizonSeconds=*/24 * 3600.0, /*staggerSeconds=*/1.0);
    nlohmann::json evSummary;
    try {
        const ExperimentResult r =
            runInto(eventCfg, fs::path(::testing::TempDir()) / "rr_p18_xrun_ev");
        evSummary = readJson(r.directory / "summary.json");
    } catch (const std::exception &e) {
        GTEST_SKIP()
            << "PENDING PACKAGE A (event arm stub throws: " << e.what()
            << "). Round-robin reference (runs today) observed: completion_rate=" << *rrCompletion
            << ", mean_immediate_satisfaction=" << *rrSatisfaction
            << ". Post-merge asserts the degenerate event run's means fall within "
               "|Δ completion| < "
            << kMaxCompletionRateDelta << " and |Δ satisfaction| < " << kMaxMeanSatisfactionDelta
            << " of these (D20 comparability band; recalibrate the margins/horizon here if "
               "the integration numbers warrant).";
    }

    const std::optional<double> evCompletion = completionRate(evSummary);
    const std::optional<double> evSatisfaction = meanSatisfaction(evSummary);
    ASSERT_TRUE(evCompletion.has_value()) << "event summary.json missing metrics.completion_rate";
    ASSERT_TRUE(evSatisfaction.has_value())
        << "event summary.json missing welfare.mean_immediate_satisfaction";

    EXPECT_NEAR(*evCompletion, *rrCompletion, kMaxCompletionRateDelta)
        << "event vs round-robin completion rate outside the comparability band";
    EXPECT_NEAR(*evSatisfaction, *rrSatisfaction, kMaxMeanSatisfactionDelta)
        << "event vs round-robin mean satisfaction outside the comparability band";
}
