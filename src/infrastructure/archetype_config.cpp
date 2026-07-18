#include "rr/infrastructure/archetype_config.hpp"

#include <array>
#include <stdexcept>
#include <string_view>

#include <nlohmann/json.hpp>

namespace rr {
namespace {

// Every JSON key ArchetypeSpec accepts (snake_case per D6). from_json rejects anything else so a
// typo is a config error (D10 fail-fast), exactly as the V1 config blocks do. Kept in field order.
constexpr std::array<std::string_view, 20> kKnownKeys = {
    "name",
    "weight",
    "usefulness_mean",
    "humour_mean",
    "novelty_mean",
    "controversy_mean",
    "clickbait_strength_mean",
    "information_density_mean",
    "emotional_intensity_mean",
    "production_quality_shift",
    "scalar_spread",
    "visual_coherence",
    "music_coherence",
    "emotional_coherence",
    "satisfaction_bias",
    "regret_bias",
    "opening_hook",
    "retention_decay",
    "niche_cohort_width",
    "comfort_return_bonus",
};

bool isKnownKey(const std::string &key) {
    for (std::string_view k : kKnownKeys) {
        if (key == k) {
            return true;
        }
    }
    return false;
}

// Read an optional double key, leaving the (already-defaulted) member untouched when absent — so a
// config may specify only the parameters it wants to override.
void readOptional(const nlohmann::json &j, const char *key, double &out) {
    if (j.contains(key)) {
        out = j.at(key).get<double>();
    }
}

} // namespace

void to_json(nlohmann::json &j, const ArchetypeSpec &a) {
    // Emit every field so a to_json -> from_json round-trip is the identity (the ConfigTest
    // round-trip cases depend on this).
    j = nlohmann::json{
        {"name", a.name},
        {"weight", a.weight},
        {"usefulness_mean", a.usefulnessMean},
        {"humour_mean", a.humourMean},
        {"novelty_mean", a.noveltyMean},
        {"controversy_mean", a.controversyMean},
        {"clickbait_strength_mean", a.clickbaitStrengthMean},
        {"information_density_mean", a.informationDensityMean},
        {"emotional_intensity_mean", a.emotionalIntensityMean},
        {"production_quality_shift", a.productionQualityShift},
        {"scalar_spread", a.scalarSpread},
        {"visual_coherence", a.visualCoherence},
        {"music_coherence", a.musicCoherence},
        {"emotional_coherence", a.emotionalCoherence},
        {"satisfaction_bias", a.satisfactionBias},
        {"regret_bias", a.regretBias},
        {"opening_hook", a.openingHook},
        {"retention_decay", a.retentionDecay},
        {"niche_cohort_width", a.nicheCohortWidth},
        {"comfort_return_bonus", a.comfortReturnBonus},
    };
}

void from_json(const nlohmann::json &j, ArchetypeSpec &a) {
    for (const auto &[key, value] : j.items()) {
        (void)value;
        if (!isKnownKey(key)) {
            throw std::invalid_argument("unknown archetype key: " + key);
        }
    }
    // `name` is required and non-empty (frozen contract; existing config tests assert this).
    if (!j.contains("name") || !j.at("name").is_string() ||
        j.at("name").get<std::string>().empty()) {
        throw std::invalid_argument("archetype entry requires a non-empty name");
    }
    a.name = j.at("name").get<std::string>();

    readOptional(j, "weight", a.weight);
    readOptional(j, "usefulness_mean", a.usefulnessMean);
    readOptional(j, "humour_mean", a.humourMean);
    readOptional(j, "novelty_mean", a.noveltyMean);
    readOptional(j, "controversy_mean", a.controversyMean);
    readOptional(j, "clickbait_strength_mean", a.clickbaitStrengthMean);
    readOptional(j, "information_density_mean", a.informationDensityMean);
    readOptional(j, "emotional_intensity_mean", a.emotionalIntensityMean);
    readOptional(j, "production_quality_shift", a.productionQualityShift);
    readOptional(j, "scalar_spread", a.scalarSpread);
    readOptional(j, "visual_coherence", a.visualCoherence);
    readOptional(j, "music_coherence", a.musicCoherence);
    readOptional(j, "emotional_coherence", a.emotionalCoherence);
    readOptional(j, "satisfaction_bias", a.satisfactionBias);
    readOptional(j, "regret_bias", a.regretBias);
    readOptional(j, "opening_hook", a.openingHook);
    readOptional(j, "retention_decay", a.retentionDecay);
    readOptional(j, "niche_cohort_width", a.nicheCohortWidth);
    readOptional(j, "comfort_return_bonus", a.comfortReturnBonus);

    // Validation (frozen contract + D10 fail-fast for the parameters that feed generation math).
    if (!(a.weight > 0.0)) {
        throw std::invalid_argument("archetype weight must be > 0 for " + a.name);
    }
    if (a.scalarSpread < 0.0) {
        throw std::invalid_argument("archetype scalar_spread must be >= 0 for " + a.name);
    }
    const auto requireUnitRange = [&a](const char *label, double coherence) {
        if (coherence < 0.0 || coherence > 1.0) {
            throw std::invalid_argument(std::string("archetype ") + label +
                                        " must be in [0, 1] for " + a.name);
        }
    };
    requireUnitRange("visual_coherence", a.visualCoherence);
    requireUnitRange("music_coherence", a.musicCoherence);
    requireUnitRange("emotional_coherence", a.emotionalCoherence);
    if (a.nicheCohortWidth < 0.0) {
        throw std::invalid_argument("archetype niche_cohort_width must be >= 0 for " + a.name);
    }
}

std::vector<ArchetypeSpec> defaultArchetypeCatalog() {
    // The eight V2 TDD 4.4 archetypes transliterated to distribution parameters with DISTINCT
    // signatures (D24). Weights are relative and normalized at sampling; they sum to 1.0 here
    // purely for readability. Every archetype is a probabilistic feature combination, never a
    // label the ranker sees (D18). Field order matches the struct so this reads as a table.
    std::vector<ArchetypeSpec> catalog;
    catalog.reserve(8);

    auto add = [&catalog](ArchetypeSpec spec) { catalog.push_back(std::move(spec)); };

    // Genuinely satisfying: positive satisfaction, solid production, low controversy.
    add({.name = "genuinely_satisfying",
         .weight = 0.22,
         .usefulnessMean = 0.60,
         .humourMean = 0.55,
         .noveltyMean = 0.50,
         .controversyMean = 0.15,
         .clickbaitStrengthMean = 0.05,
         .informationDensityMean = 0.55,
         .emotionalIntensityMean = 0.55,
         .productionQualityShift = 0.10,
         .scalarSpread = 0.15,
         .visualCoherence = 0.60,
         .musicCoherence = 0.55,
         .emotionalCoherence = 0.60,
         .satisfactionBias = 0.35,
         .regretBias = -0.05});

    // Useful: high usefulness + information density, muted humour, positive satisfaction.
    add({.name = "useful",
         .weight = 0.14,
         .usefulnessMean = 0.85,
         .humourMean = 0.25,
         .noveltyMean = 0.45,
         .controversyMean = 0.10,
         .clickbaitStrengthMean = 0.05,
         .informationDensityMean = 0.85,
         .emotionalIntensityMean = 0.30,
         .productionQualityShift = 0.0,
         .scalarSpread = 0.12,
         .visualCoherence = 0.50,
         .musicCoherence = 0.45,
         .emotionalCoherence = 0.45,
         .satisfactionBias = 0.30,
         .regretBias = -0.10});

    // Ragebait: high controversy + emotional intensity, NEGATIVE satisfaction, POSITIVE regret.
    add({.name = "ragebait",
         .weight = 0.10,
         .usefulnessMean = 0.15,
         .humourMean = 0.20,
         .noveltyMean = 0.40,
         .controversyMean = 0.85,
         .clickbaitStrengthMean = 0.55,
         .informationDensityMean = 0.20,
         .emotionalIntensityMean = 0.85,
         .productionQualityShift = -0.05,
         .scalarSpread = 0.15,
         .visualCoherence = 0.55,
         .musicCoherence = 0.55,
         .emotionalCoherence = 0.60,
         .satisfactionBias = -0.35,
         .regretBias = 0.35,
         .openingHook = 0.10,
         .retentionDecay = 0.10});

    // Clickbait: very high clickbait strength, strong opening hook + retention decay, POSITIVE
    // regret (early abandonment after the hook).
    add({.name = "clickbait",
         .weight = 0.12,
         .usefulnessMean = 0.20,
         .humourMean = 0.35,
         .noveltyMean = 0.50,
         .controversyMean = 0.45,
         .clickbaitStrengthMean = 0.90,
         .informationDensityMean = 0.25,
         .emotionalIntensityMean = 0.60,
         .productionQualityShift = 0.05,
         .scalarSpread = 0.15,
         .visualCoherence = 0.60,
         .musicCoherence = 0.55,
         .emotionalCoherence = 0.55,
         .satisfactionBias = -0.15,
         .regretBias = 0.25,
         .openingHook = 0.60,
         .retentionDecay = 0.60});

    // Comfort content: familiar (low novelty), mild positive satisfaction, POSITIVE return bonus.
    add({.name = "comfort",
         .weight = 0.14,
         .usefulnessMean = 0.40,
         .humourMean = 0.50,
         .noveltyMean = 0.20,
         .controversyMean = 0.10,
         .clickbaitStrengthMean = 0.05,
         .informationDensityMean = 0.35,
         .emotionalIntensityMean = 0.35,
         .productionQualityShift = 0.0,
         .scalarSpread = 0.13,
         .visualCoherence = 0.55,
         .musicCoherence = 0.60,
         .emotionalCoherence = 0.55,
         .satisfactionBias = 0.15,
         .regretBias = -0.05,
         .comfortReturnBonus = 0.40});

    // Highly polished but irrelevant: VERY high production shift, low usefulness/novelty, catches
    // attention without lasting value (slightly negative satisfaction, mild regret).
    add({.name = "polished_irrelevant",
         .weight = 0.10,
         .usefulnessMean = 0.15,
         .humourMean = 0.35,
         .noveltyMean = 0.20,
         .controversyMean = 0.15,
         .clickbaitStrengthMean = 0.20,
         .informationDensityMean = 0.20,
         .emotionalIntensityMean = 0.40,
         .productionQualityShift = 0.35,
         .scalarSpread = 0.13,
         .visualCoherence = 0.70,
         .musicCoherence = 0.55,
         .emotionalCoherence = 0.55,
         .satisfactionBias = -0.05,
         .regretBias = 0.10});

    // Niche treasure: highly satisfying to a small cohort (niche_cohort_width > 0 => centre drawn
    // per reel), novel, high satisfaction bias.
    add({.name = "niche_treasure",
         .weight = 0.08,
         .usefulnessMean = 0.60,
         .humourMean = 0.45,
         .noveltyMean = 0.70,
         .controversyMean = 0.20,
         .clickbaitStrengthMean = 0.05,
         .informationDensityMean = 0.55,
         .emotionalIntensityMean = 0.50,
         .productionQualityShift = -0.05,
         .scalarSpread = 0.15,
         .visualCoherence = 0.55,
         .musicCoherence = 0.55,
         .emotionalCoherence = 0.55,
         .satisfactionBias = 0.45,
         .regretBias = -0.10,
         .nicheCohortWidth = 0.15});

    // Background music reel: STRONG music coherence, weak topic relevance (low usefulness/info).
    add({.name = "background_music",
         .weight = 0.10,
         .usefulnessMean = 0.20,
         .humourMean = 0.35,
         .noveltyMean = 0.40,
         .controversyMean = 0.10,
         .clickbaitStrengthMean = 0.05,
         .informationDensityMean = 0.20,
         .emotionalIntensityMean = 0.45,
         .productionQualityShift = 0.0,
         .scalarSpread = 0.13,
         .visualCoherence = 0.50,
         .musicCoherence = 0.90,
         .emotionalCoherence = 0.50,
         .satisfactionBias = 0.10,
         .regretBias = -0.05});

    return catalog;
}

} // namespace rr
