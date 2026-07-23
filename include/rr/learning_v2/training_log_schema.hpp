#pragma once

#include <array>
#include <cstdint>
#include <string_view>

#include "rr/core/pinned_hash.hpp"

// ================================================================================================
// Phase 22 FROZEN training-log schema (contracts docs/design/P22-CONTRACTS.md §2). This header is
// the SINGLE SOURCE OF TRUTH for every training-log column name, the schema version, and the two
// pinned request-sampling salts. It is consumed by BOTH the writer (package A's TrainingLogger)
// AND the emitted-file purity audit (package A's §4 test: every CSV header column must be a member
// of the exact allowlists below), so the two can never drift apart.
//
// D18 purity: this header lives under include/rr/learning_v2/ (a D18-scanned root as of Phase 22)
// and pulls in NOTHING from simulation/hidden/ — only pinned_hash (core) and the standard library.
// ================================================================================================

namespace rr::learning_v2 {

// Bumped only when a column set changes; echoed into schema.json and audited against it (§4c).
inline constexpr int kSchemaVersion = 1;

// ---- requests.csv — one row per SAMPLED request (shown-sample UNION pool-sample) ----------------
// contracts §2. `pool_logged` is 1 iff the request won the pool-sample draw (full ranked pool
// emitted), else 0 (only its shown impressions were logged).
inline constexpr std::array<std::string_view, 9> kRequestsColumns = {
    "request_id",        "user_id",   "session_id",  "timestamp",   "feed_size",
    "effective_epsilon", "pool_size", "shown_count", "pool_logged",
};

// ---- candidates.csv fixed prefix (contracts §2). The FULL header is this prefix followed IN ORDER
// by kFeatureColumns below; the writer and the purity-audit allowlist both use prefix UNION
// features. `position` = feed slot for shown rows, -1 for pool-only rows. -----------------------
inline constexpr std::array<std::string_view, 9> kCandidatesPrefixColumns = {
    "request_id",
    "reel_id",
    "pool_rank",
    "shown",
    "position",
    "served_score",
    "exploration_flag",
    "retrieval_sources",
    "retrieval_similarity",
};

// ---- FEATURE COLUMNS — every field of rr::FeatureVector in DECLARATION ORDER, snake_case
// (contracts §2). Transcribed VERBATIM from include/rr/recommendation/feature_extractor.hpp; the
// STRUCT is authoritative. THESE ARE SERVED-TIME VALUES captured at the ranking call.
//
// NOTE (scaffold flag, see report): the struct's DECLARATION ORDER and two of its field names
// differ from the §2 PROSE list. This array follows the struct (authoritative):
//   * order: struct declares session_topic 2nd and popularity 5th (§2 prose lists them 10th/11th),
//     and the V2 tail order differs (struct: clickbait before emotional_intensity/usefulness/...).
//   * names: struct fields are `repetition` and `impression_count` (the §2 prose calls these
//     `repetition_penalty` / `impression_penalty`). The struct comment documents both as the raw
//     [0,1] penalty MAGNITUDES; the column names here are the snake_cased struct identifiers.
// If a field is ever added/removed/renamed in FeatureVector, update THIS array in the same commit
// and bump kSchemaVersion — the purity audit fails on any mismatch.
inline constexpr std::array<std::string_view, 21> kFeatureColumns = {
    "similarity",          "session_topic",    "quality",
    "freshness",           "popularity",       "trending",
    "creator_affinity",    "exploration",      "duration_match",
    "repetition",          "impression_count", "visual_match",
    "music_match",         "emotional_match",  "clickbait",
    "emotional_intensity", "usefulness",       "production_quality",
    "information_density", "language_match",   "save_popularity",
};

// ---- outcomes.csv — SEPARATE label table (purity: evaluation-side observables only). Joined from
// InteractionEvents for SHOWN sampled impressions. contracts §2. ---------------------------------
inline constexpr std::array<std::string_view, 14> kOutcomesColumns = {
    "request_id",      "reel_id",
    "position",        "watch_seconds",
    "watch_ratio",     "completed",
    "liked",           "shared",
    "followed",        "not_interested",
    "commented",       "saved",
    "profile_visited", "observed_exit_after_impression",
};

// ---- survey.csv — the ONLY hidden-derived table (gate survey.enabled only, clearly labeled in
// schema.json). `likert` in 1..5 quantized from immediateSatisfaction + gaussian(noise_sd).
// contracts §2. ---------------------------------------------------------------------------------
inline constexpr std::array<std::string_view, 5> kSurveyColumns = {
    "user_id", "reel_id", "request_id", "timestamp", "likert",
};

// ---- Pinned request-sampling salts (contracts §1). Two DISTINCT frozen uint64 constants — one per
// sample rate — XOR'd with request_id before the pinned SplitMix64 finalizer, so the shown-sample
// and pool-sample decisions are independent and neither draws simulation rng. The literals are
// ASCII mnemonics ("REELSHWN" / "REELPOOL"), pinned like cohortHash01's constants; changing them
// re-shuffles which requests get logged and must break a golden tripwire, so DO NOT change them.
inline constexpr uint64_t kLogSampleSalt = 0x5245454C5348574EULL; // "REELSHWN" — shown sample
inline constexpr uint64_t kLogPoolSampleSalt =
    0x5245454C504F4F4CULL; // "REELPOOL" — full-pool sample

// The pinned, rng-free request-sampling predicate (contracts §1): a request is logged iff
// pinnedHash01(requestId ^ salt) < rate. Callers pass kLogSampleSalt with log_sample_rate for the
// shown-impression sample and kLogPoolSampleSalt with log_pool_sample_rate for the full-pool
// sample. Header-inline so the writer AND its determinism/purity tests share one pinned decision.
inline bool logSampleSelected(uint64_t requestId, uint64_t salt, double rate) {
    return pinnedHash01(requestId ^ salt) < rate;
}

} // namespace rr::learning_v2
