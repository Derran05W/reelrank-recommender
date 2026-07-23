// Phase 22 TrainingLogger UNIT test (package A). Drives the logger directly with a synthetic
// RankingCapture + shown feed + outcomes — no simulator — so the pool/shown/position join, the
// pinned sampling predicate, the frozen headers, and part-file rotation are checked precisely and
// fast. The end-to-end emitted-file audit on a real sim lives in the integration purity test.

#include "rr/learning_v2/training_logger.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "rr/learning_v2/training_log_schema.hpp"

using namespace rr;

namespace {

namespace fs = std::filesystem;

std::vector<std::string> readLines(const fs::path &p) {
    std::ifstream in(p);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        lines.push_back(line);
    }
    return lines;
}

std::vector<std::string> splitCsv(const std::string &line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
        out.push_back(cell);
    }
    return out;
}

// A synthetic full-pool capture: `n` candidates, best-first, with distinct reel ids 10, 11, ...
RankingCapture makePool(std::size_t n) {
    RankingCapture cap;
    for (std::size_t i = 0; i < n; ++i) {
        RankingCaptureRow row;
        row.reelId = ReelId{static_cast<std::uint32_t>(10 + i)};
        row.poolRank = i;
        row.servedScore = 0.9f - 0.01f * static_cast<float>(i);
        row.explorationLabeled = (i == 2); // one exploration-labeled pool item
        row.retrievalSimilarity = 0.5f + 0.01f * static_cast<float>(i);
        row.sources = {CandidateSource::VectorHNSW};
        if (i == 2) {
            row.sources = {CandidateSource::VectorHNSW, CandidateSource::Exploration};
        }
        row.features = FeatureVector{};
        row.features.similarity = 0.5f + 0.001f * static_cast<float>(i);
        cap.rows.push_back(row);
    }
    return cap;
}

// A shown feed of the first `shownCount` pool reels, positions 0..shownCount-1.
std::vector<RankedReel> makeFeed(const RankingCapture &cap, std::size_t shownCount) {
    std::vector<RankedReel> feed;
    for (std::size_t i = 0; i < shownCount && i < cap.rows.size(); ++i) {
        RankedReel r{};
        r.reelId = cap.rows[i].reelId;
        r.score = cap.rows[i].servedScore;
        r.rank = i;
        r.sources = cap.rows[i].sources;
        feed.push_back(r);
    }
    return feed;
}

RecommendationRequest makeRequest() {
    RecommendationRequest req{};
    req.userId = UserId{7};
    req.sessionId = SessionId{3};
    req.feedSize = 2;
    req.requestTime = 100000;
    return req;
}

LearningV2Config baseConfig() {
    LearningV2Config cfg;
    cfg.trainingLog = true;
    cfg.logSampleRate = 1.0;     // log every request's shown impressions
    cfg.logPoolSampleRate = 1.0; // log every request's full pool
    return cfg;
}

} // namespace

// --- Frozen headers exactly match the schema allowlists (single source of truth) ----------------
TEST(TrainingLoggerTest, EmitsFrozenHeaders) {
    const fs::path dir = fs::path(::testing::TempDir()) / "rr_p22_hdr";
    fs::remove_all(dir);
    {
        TrainingLogger logger(baseConfig(), dir);
        logger.finish(); // header-only run: every table must still exist
    }
    const fs::path log = dir / "training_log";
    ASSERT_TRUE(fs::exists(log / "requests.csv"));
    ASSERT_TRUE(fs::exists(log / "candidates-part0000.csv"));
    ASSERT_TRUE(fs::exists(log / "outcomes-part0000.csv"));

    // requests header == kRequestsColumns, in order.
    const auto reqHdr = splitCsv(readLines(log / "requests.csv").at(0));
    ASSERT_EQ(reqHdr.size(), learning_v2::kRequestsColumns.size());
    for (std::size_t i = 0; i < reqHdr.size(); ++i) {
        EXPECT_EQ(reqHdr[i], learning_v2::kRequestsColumns[i]);
    }
    // candidates header == prefix ++ features, in order.
    const auto candHdr = splitCsv(readLines(log / "candidates-part0000.csv").at(0));
    ASSERT_EQ(candHdr.size(),
              learning_v2::kCandidatesPrefixColumns.size() + learning_v2::kFeatureColumns.size());
    for (std::size_t i = 0; i < learning_v2::kCandidatesPrefixColumns.size(); ++i) {
        EXPECT_EQ(candHdr[i], learning_v2::kCandidatesPrefixColumns[i]);
    }
    for (std::size_t i = 0; i < learning_v2::kFeatureColumns.size(); ++i) {
        EXPECT_EQ(candHdr[learning_v2::kCandidatesPrefixColumns.size() + i],
                  learning_v2::kFeatureColumns[i]);
    }
    // outcomes header == kOutcomesColumns, in order.
    const auto outHdr = splitCsv(readLines(log / "outcomes-part0000.csv").at(0));
    ASSERT_EQ(outHdr.size(), learning_v2::kOutcomesColumns.size());
    for (std::size_t i = 0; i < outHdr.size(); ++i) {
        EXPECT_EQ(outHdr[i], learning_v2::kOutcomesColumns[i]);
    }
}

// --- Pool-sampled request: the FULL pool is logged with shown flags + feed positions (-1 for
// pool-only), and the outcome joins on (request_id, reel_id). Exit-condition-4 evidence. ----------
TEST(TrainingLoggerTest, PoolSampledLogsFullPoolWithPositions) {
    const fs::path dir = fs::path(::testing::TempDir()) / "rr_p22_pool";
    fs::remove_all(dir);

    const RankingCapture cap = makePool(4);
    const std::vector<RankedReel> feed = makeFeed(cap, 2); // reels 10,11 shown at positions 0,1
    User user{};
    user.id = UserId{7}; // requests.csv user_id comes from the observable User (== request.userId)
    {
        TrainingLogger logger(baseConfig(), dir);
        logger.onRequestRanked(1, makeRequest(), user, 0.05, cap, feed);
        // Outcome for the shown reel 10 (completed + liked at once — the lossy-type case).
        InteractionEvent ev{};
        ev.requestId = 1;
        ev.reelId = ReelId{10};
        ev.positionInFeed = 0;
        ev.watchSeconds = 5.0f;
        ev.watchRatio = 0.5f;
        BehaviourOutcome oc{};
        oc.completed = true;
        oc.liked = true;
        logger.onImpressionOutcome(ev, oc);
        logger.finish();
    }
    const fs::path log = dir / "training_log";

    // requests.csv: one row, pool_size=4, shown_count=2, pool_logged=1.
    const auto reqLines = readLines(log / "requests.csv");
    ASSERT_EQ(reqLines.size(), 2u); // header + 1
    const auto req = splitCsv(reqLines[1]);
    EXPECT_EQ(req[0], "1"); // request_id
    EXPECT_EQ(req[1], "7"); // user_id
    EXPECT_EQ(req[2], "3"); // session_id
    EXPECT_EQ(req[6], "4"); // pool_size
    EXPECT_EQ(req[7], "2"); // shown_count
    EXPECT_EQ(req[8], "1"); // pool_logged

    // candidates.csv: all 4 pool rows. Map reel_id -> (shown, position).
    const auto candLines = readLines(log / "candidates-part0000.csv");
    ASSERT_EQ(candLines.size(), 5u); // header + 4
    int shownRows = 0;
    int poolOnlyRows = 0;
    bool sawExplorationFlag = false;
    for (std::size_t i = 1; i < candLines.size(); ++i) {
        const auto c = splitCsv(candLines[i]);
        const std::string reelId = c[1];
        const std::string shown = c[3];
        const std::string position = c[4];
        const std::string explorationFlag = c[6];
        if (reelId == "10") {
            EXPECT_EQ(shown, "1");
            EXPECT_EQ(position, "0");
        } else if (reelId == "11") {
            EXPECT_EQ(shown, "1");
            EXPECT_EQ(position, "1");
        } else { // 12, 13 pool-only
            EXPECT_EQ(shown, "0");
            EXPECT_EQ(position, "-1");
        }
        if (shown == "1") {
            ++shownRows;
        } else {
            ++poolOnlyRows;
        }
        if (reelId == "12") {
            EXPECT_EQ(explorationFlag, "1");            // pool item 2 was exploration-labeled
            EXPECT_EQ(c[7], "vector_hnsw|exploration"); // retrieval_sources union preserved
            sawExplorationFlag = true;
        }
    }
    EXPECT_EQ(shownRows, 2);
    EXPECT_EQ(poolOnlyRows, 2);
    EXPECT_TRUE(sawExplorationFlag);

    // outcomes.csv: one joined row, completed=1 AND liked=1 (both, from step.outcome).
    const auto outLines = readLines(log / "outcomes-part0000.csv");
    ASSERT_EQ(outLines.size(), 2u); // header + 1
    const auto out = splitCsv(outLines[1]);
    EXPECT_EQ(out[0], "1");  // request_id
    EXPECT_EQ(out[1], "10"); // reel_id
    EXPECT_EQ(out[2], "0");  // position
    EXPECT_EQ(out[5], "1");  // completed
    EXPECT_EQ(out[6], "1");  // liked
    EXPECT_EQ(out[7], "0");  // shared
}

// --- Shown-only sample (pool rate 0): only the shown impressions are logged, pool-only dropped ---
TEST(TrainingLoggerTest, ShownOnlySampleLogsOnlyShownRows) {
    const fs::path dir = fs::path(::testing::TempDir()) / "rr_p22_shown";
    fs::remove_all(dir);
    LearningV2Config cfg = baseConfig();
    cfg.logPoolSampleRate = 0.0; // shown-only

    const RankingCapture cap = makePool(4);
    const std::vector<RankedReel> feed = makeFeed(cap, 2);
    {
        TrainingLogger logger(cfg, dir);
        logger.onRequestRanked(1, makeRequest(), User{}, 0.0, cap, feed);
        logger.finish();
    }
    const fs::path log = dir / "training_log";
    const auto candLines = readLines(log / "candidates-part0000.csv");
    ASSERT_EQ(candLines.size(), 3u); // header + 2 shown rows only
    for (std::size_t i = 1; i < candLines.size(); ++i) {
        EXPECT_EQ(splitCsv(candLines[i])[3], "1"); // every emitted row is shown
    }
    // requests.csv still records the true pool size + pool_logged=0.
    const auto req = splitCsv(readLines(log / "requests.csv").at(1));
    EXPECT_EQ(req[6], "4"); // pool_size (the full pool, even though not all logged)
    EXPECT_EQ(req[8], "0"); // pool_logged
}

// --- Rotation: candidates/outcomes split into contiguous part files at log_max_rows_per_file -----
TEST(TrainingLoggerTest, RotatesPartFilesAtRowLimit) {
    const fs::path dir = fs::path(::testing::TempDir()) / "rr_p22_rot";
    fs::remove_all(dir);
    LearningV2Config cfg = baseConfig();
    cfg.logMaxRowsPerFile = 2; // tiny threshold

    const RankingCapture cap = makePool(5); // 5 pool rows => parts 0000(2),0001(2),0002(1)
    const std::vector<RankedReel> feed = makeFeed(cap, 1);
    {
        TrainingLogger logger(cfg, dir);
        logger.onRequestRanked(1, makeRequest(), User{}, 0.0, cap, feed);
        logger.finish();
    }
    const fs::path log = dir / "training_log";
    ASSERT_TRUE(fs::exists(log / "candidates-part0000.csv"));
    ASSERT_TRUE(fs::exists(log / "candidates-part0001.csv"));
    ASSERT_TRUE(fs::exists(log / "candidates-part0002.csv"));
    EXPECT_FALSE(fs::exists(log / "candidates-part0003.csv"));
    // Each part: header + at most 2 data rows.
    EXPECT_EQ(readLines(log / "candidates-part0000.csv").size(), 3u); // header + 2
    EXPECT_EQ(readLines(log / "candidates-part0001.csv").size(), 3u); // header + 2
    EXPECT_EQ(readLines(log / "candidates-part0002.csv").size(), 2u); // header + 1
}

// --- Unsampled request (both rates 0) writes no data rows anywhere
// ---------------------------------
TEST(TrainingLoggerTest, UnsampledRequestEmitsNoRows) {
    const fs::path dir = fs::path(::testing::TempDir()) / "rr_p22_unsampled";
    fs::remove_all(dir);
    LearningV2Config cfg = baseConfig();
    cfg.logSampleRate = 0.0;
    cfg.logPoolSampleRate = 0.0;

    const RankingCapture cap = makePool(4);
    const std::vector<RankedReel> feed = makeFeed(cap, 2);
    {
        TrainingLogger logger(cfg, dir);
        logger.onRequestRanked(1, makeRequest(), User{}, 0.0, cap, feed);
        InteractionEvent ev{};
        ev.requestId = 1;
        ev.reelId = ReelId{10};
        logger.onImpressionOutcome(ev, BehaviourOutcome{});
        logger.finish();
    }
    const fs::path log = dir / "training_log";
    // Files exist (header-only) but carry zero data rows.
    EXPECT_EQ(readLines(log / "requests.csv").size(), 1u);
    EXPECT_EQ(readLines(log / "candidates-part0000.csv").size(), 1u);
    EXPECT_EQ(readLines(log / "outcomes-part0000.csv").size(), 1u);
}
