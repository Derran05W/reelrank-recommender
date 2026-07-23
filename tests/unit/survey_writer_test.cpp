// Phase 22 SurveyWriter UNIT test (package A). Verifies the two-draw "explicit-feedback" contract
// (exactly two draws per shown impression, in a fixed order, regardless of outcome), the documented
// Likert quantization of the noisy immediateSatisfaction, and the header-only file when nothing is
// surveyed. Stream discipline vs the V1 streams (survey OFF => zero draws) is proven end-to-end by
// the integration determinism test's digest-identity check.

#include "rr/evaluation/survey_writer.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "rr/infrastructure/random.hpp"

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

LearningV2Config surveyConfig(double sampleRate, double noiseSd) {
    LearningV2Config cfg;
    cfg.survey.enabled = true;
    cfg.survey.sampleRate = sampleRate;
    cfg.survey.noiseSd = noiseSd;
    return cfg;
}

int lastCell(const std::string &line) {
    const auto pos = line.find_last_of(',');
    return std::stoi(line.substr(pos + 1));
}

} // namespace

// --- Exactly TWO draws per call, in order (bernoulli then gaussian), whatever the outcome
// ---------
TEST(SurveyWriterTest, MakesExactlyTwoDrawsPerImpression) {
    const fs::path dir = fs::path(::testing::TempDir()) / "rr_p22_survey_draws";
    fs::remove_all(dir);
    const LearningV2Config cfg = surveyConfig(0.5, 0.35);
    SurveyWriter writer(cfg, dir);

    Rng a(20260722);
    Rng b(20260722); // reference: manually draws bernoulli then gaussian per impression

    for (std::uint64_t i = 0; i < 64; ++i) {
        writer.maybeSurvey(UserId{static_cast<std::uint32_t>(i)},
                           ReelId{static_cast<std::uint32_t>(i)}, i, 1000 + i, 0.25f, a);
        (void)b.bernoulli(cfg.survey.sampleRate);
        (void)b.gaussian();
    }
    writer.finish();

    // If the writer consumed exactly two draws per call, the two streams are still aligned.
    EXPECT_EQ(a.nextU64(), b.nextU64());
}

// --- sample_rate=0 => no rows written, but the two draws STILL happen (stream stays aligned)
// ------
TEST(SurveyWriterTest, ZeroRateStillDrawsTwiceAndWritesNoRows) {
    const fs::path dir = fs::path(::testing::TempDir()) / "rr_p22_survey_zero";
    fs::remove_all(dir);
    const LearningV2Config cfg = surveyConfig(0.0, 0.35);
    SurveyWriter writer(cfg, dir);

    Rng a(7);
    Rng b(7);
    for (std::uint64_t i = 0; i < 40; ++i) {
        writer.maybeSurvey(UserId{1}, ReelId{1}, i, 500 + i, 0.1f, a);
        (void)b.bernoulli(0.0);
        (void)b.gaussian();
    }
    writer.finish();

    EXPECT_EQ(a.nextU64(), b.nextU64()); // two draws per call even when nothing is surveyed
    // survey.csv exists but is header-only.
    EXPECT_EQ(readLines(dir / "training_log" / "survey.csv").size(), 1u);
}

// --- Likert quantization: documented mapping of clamp(sat + noise) in [-1,1] onto 1..5
// ------------
TEST(SurveyWriterTest, LikertMappingAndRange) {
    const fs::path dir = fs::path(::testing::TempDir()) / "rr_p22_survey_likert";
    fs::remove_all(dir);
    const LearningV2Config cfg =
        surveyConfig(1.0, 0.0); // always surveyed, no noise => deterministic
    SurveyWriter writer(cfg, dir);

    Rng rng(99);
    writer.maybeSurvey(UserId{1}, ReelId{1}, 0, 100, 1.0f, rng);  // sat=+1 => likert 5
    writer.maybeSurvey(UserId{2}, ReelId{2}, 1, 101, -1.0f, rng); // sat=-1 => likert 1
    writer.maybeSurvey(UserId{3}, ReelId{3}, 2, 102, 0.0f, rng);  // sat= 0 => likert 3
    writer.maybeSurvey(UserId{4}, ReelId{4}, 3, 103, 0.5f, rng);  // sat=+.5 => likert 4
    writer.finish();

    const auto lines = readLines(dir / "training_log" / "survey.csv");
    ASSERT_EQ(lines.size(), 5u); // header + 4
    EXPECT_EQ(lastCell(lines[1]), 5);
    EXPECT_EQ(lastCell(lines[2]), 1);
    EXPECT_EQ(lastCell(lines[3]), 3);
    EXPECT_EQ(lastCell(lines[4]), 4);
}

// --- Noisy draws always land in [1,5]
// --------------------------------------------------------------
TEST(SurveyWriterTest, NoisyLikertStaysInRange) {
    const fs::path dir = fs::path(::testing::TempDir()) / "rr_p22_survey_range";
    fs::remove_all(dir);
    const LearningV2Config cfg = surveyConfig(1.0, 1.5); // large noise to stress the clamp
    SurveyWriter writer(cfg, dir);

    Rng rng(2024);
    for (std::uint64_t i = 0; i < 300; ++i) {
        const float sat = -1.0f + 2.0f * (static_cast<float>(i % 11) / 10.0f);
        writer.maybeSurvey(UserId{1}, ReelId{1}, i, 100 + i, sat, rng);
    }
    writer.finish();

    const auto lines = readLines(dir / "training_log" / "survey.csv");
    ASSERT_GT(lines.size(), 1u);
    for (std::size_t i = 1; i < lines.size(); ++i) {
        const int likert = lastCell(lines[i]);
        EXPECT_GE(likert, 1);
        EXPECT_LE(likert, 5);
    }
}
