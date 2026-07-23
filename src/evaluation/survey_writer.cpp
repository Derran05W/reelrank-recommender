#include "rr/evaluation/survey_writer.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "rr/learning_v2/training_log_schema.hpp"

namespace rr {

namespace {

// Comma-join the frozen survey column allowlist into the CSV header (single source of truth: the
// schema header, which the purity audit checks against — the two can never drift).
std::string surveyHeaderRow() {
    std::string out;
    for (std::size_t i = 0; i < learning_v2::kSurveyColumns.size(); ++i) {
        if (i != 0) {
            out += ',';
        }
        out += std::string(learning_v2::kSurveyColumns[i]);
    }
    return out;
}

} // namespace

SurveyWriter::SurveyWriter(const LearningV2Config &config, std::filesystem::path runDir)
    : config_(config), trainingLogDir_(std::move(runDir) / "training_log") {}

void SurveyWriter::ensureOpen() {
    if (opened_) {
        return;
    }
    std::filesystem::create_directories(trainingLogDir_);
    surveyOut_.open(trainingLogDir_ / "survey.csv");
    surveyOut_ << surveyHeaderRow() << '\n';
    opened_ = true;
}

// Two-draw "explicit-feedback" contract (contracts §1, header). BOTH draws happen unconditionally,
// in a fixed order (bernoulli THEN gaussian), so the per-impression draw count is exactly 2 and the
// stream stays aligned regardless of the sample outcome.
//
// LIKERT MAPPING (documented here per contracts §2): immediateSatisfaction is roughly [-1, 1]
// (the simulator clamps LatentReaction::immediateSatisfaction to [-1, 1]). The noisy value
//   v = clamp(immediateSatisfaction + gaussian()*noise_sd, -1, 1)
// is mapped onto the 5-point Likert scale by the affine bucketing likert = 1 + round((v + 1) * 2),
// i.e. five equal-width bins over [-1, 1]:  v=-1 -> 1,  v=0 -> 3,  v=+1 -> 5. Deterministic given
// the stream (round-half-away-from-zero via std::lround).
std::optional<int> SurveyWriter::maybeSurvey(UserId userId, ReelId reelId, std::uint64_t requestId,
                                             Timestamp timestamp, float immediateSatisfaction,
                                             Rng &explicitFeedback) {
    const bool surveyed = explicitFeedback.bernoulli(config_.survey.sampleRate); // DRAW 1
    const double noise = explicitFeedback.gaussian() * config_.survey.noiseSd;   // DRAW 2
    if (!surveyed) {
        return std::nullopt;
    }
    const double v = std::clamp(static_cast<double>(immediateSatisfaction) + noise, -1.0, 1.0);
    const long bucket = std::lround((v + 1.0) * 2.0); // (v+1)/2 * 4, in [0, 4]
    const int likert = static_cast<int>(std::clamp(bucket, 0L, 4L)) + 1;

    ensureOpen();
    surveyOut_ << userId.value << ',' << reelId.value << ',' << requestId << ',' << timestamp << ','
               << likert << '\n';
    return likert; // Phase 23: the observable label for the in-memory training matrix.
}

void SurveyWriter::finish() {
    ensureOpen(); // guarantee survey.csv exists (header-only if nothing was surveyed).
    surveyOut_.flush();
}

} // namespace rr
