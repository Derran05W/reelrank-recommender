#pragma once

#include <cstdint>
#include <vector>

namespace rr {

// Skewed global language distribution over realism.languages ids (V2 TDD 4.1: "small
// config-driven language set with a skewed global distribution"). weight_i is proportional to
// 1 / (i + 1) — a Zipf(s=1) profile, the classic model for language/popularity skew — and the
// returned weights are normalized to sum to 1, so language 0 is the dominant global language and
// the tail thins harmonically. Pure function of the count, no rng: reel draws ("reels-v2") and
// user primary-language draws ("users-v2") both sample from the SAME distribution via
// Rng::uniform against the cumulative weights, keeping the two sides' language mixes aligned by
// construction. Throws std::invalid_argument when languages == 0 (a setup error, D10).
std::vector<double> languageWeights(uint32_t languages);

} // namespace rr
