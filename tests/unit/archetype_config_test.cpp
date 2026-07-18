#include "rr/infrastructure/archetype_config.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

using nlohmann::json;
using rr::ArchetypeSpec;
using rr::defaultArchetypeCatalog;

namespace {

// Look up an archetype by name (the default catalog is small; linear scan is fine).
const ArchetypeSpec &byName(const std::vector<ArchetypeSpec> &catalog, const std::string &name) {
    for (const ArchetypeSpec &a : catalog) {
        if (a.name == name) {
            return a;
        }
    }
    ADD_FAILURE() << "archetype not found: " << name;
    static const ArchetypeSpec kEmpty{};
    return kEmpty;
}

} // namespace

// --- Default catalog: the eight V2 TDD 4.4 archetypes ---------------------------------------

TEST(ArchetypeConfigTest, DefaultCatalogHasEightNamedArchetypes) {
    const auto catalog = defaultArchetypeCatalog();
    ASSERT_EQ(catalog.size(), 8u);
    for (const char *name : {"genuinely_satisfying", "useful", "ragebait", "clickbait", "comfort",
                             "polished_irrelevant", "niche_treasure", "background_music"}) {
        bool found = false;
        for (const ArchetypeSpec &a : catalog) {
            found = found || a.name == name;
        }
        EXPECT_TRUE(found) << "missing archetype " << name;
    }
    for (const ArchetypeSpec &a : catalog) {
        EXPECT_GT(a.weight, 0.0) << a.name << " must have a positive mixture weight";
    }
}

// The eight archetypes must carry DISTINCT signatures (D24) — asserted here on the catalog itself
// (the statistical suite then confirms they survive sampling). Each check pins one designed
// contrast from the task's archetype table.
TEST(ArchetypeConfigTest, DefaultCatalogSignaturesAreDistinct) {
    const auto c = defaultArchetypeCatalog();

    // ragebait: high controversy + emotional intensity, negative satisfaction, positive regret.
    EXPECT_GT(byName(c, "ragebait").controversyMean, 0.7);
    EXPECT_GT(byName(c, "ragebait").emotionalIntensityMean, 0.7);
    EXPECT_LT(byName(c, "ragebait").satisfactionBias, 0.0);
    EXPECT_GT(byName(c, "ragebait").regretBias, 0.0);

    // clickbait: very high clickbait strength, opening hook > 0, retention decay > 0, regret > 0.
    EXPECT_GT(byName(c, "clickbait").clickbaitStrengthMean, 0.7);
    EXPECT_GT(byName(c, "clickbait").openingHook, 0.0);
    EXPECT_GT(byName(c, "clickbait").retentionDecay, 0.0);
    EXPECT_GT(byName(c, "clickbait").regretBias, 0.0);

    // useful: high usefulness + information density, positive satisfaction.
    EXPECT_GT(byName(c, "useful").usefulnessMean, 0.7);
    EXPECT_GT(byName(c, "useful").informationDensityMean, 0.7);
    EXPECT_GT(byName(c, "useful").satisfactionBias, 0.0);

    // genuinely_satisfying: positive satisfaction, solid production polish.
    EXPECT_GT(byName(c, "genuinely_satisfying").satisfactionBias, 0.0);
    EXPECT_GT(byName(c, "genuinely_satisfying").productionQualityShift, 0.0);

    // comfort: positive return bonus.
    EXPECT_GT(byName(c, "comfort").comfortReturnBonus, 0.0);

    // polished_irrelevant: very high production shift, low usefulness/novelty.
    EXPECT_GT(byName(c, "polished_irrelevant").productionQualityShift, 0.25);
    EXPECT_LT(byName(c, "polished_irrelevant").usefulnessMean, 0.3);
    EXPECT_LT(byName(c, "polished_irrelevant").noveltyMean, 0.3);

    // niche_treasure: cohort width > 0 (centre sampled per reel), high satisfaction.
    EXPECT_GT(byName(c, "niche_treasure").nicheCohortWidth, 0.0);
    EXPECT_GT(byName(c, "niche_treasure").satisfactionBias, 0.0);

    // background_music: strong music coherence, distinctly above the other channels.
    EXPECT_GT(byName(c, "background_music").musicCoherence, 0.8);
    EXPECT_GT(byName(c, "background_music").musicCoherence,
              byName(c, "background_music").visualCoherence + 0.2);
}

// --- Serialization: unknown-key rejection, validation, defaults, round-trip ------------------

TEST(ArchetypeConfigTest, FromJsonRejectsUnknownKey) {
    json j = {{"name", "x"}, {"weight", 1.0}, {"bogus_key", 3}};
    try {
        j.get<ArchetypeSpec>();
        FAIL() << "expected throw on unknown key";
    } catch (const std::invalid_argument &e) {
        EXPECT_NE(std::string(e.what()).find("bogus_key"), std::string::npos);
    }
}

TEST(ArchetypeConfigTest, FromJsonRequiresNonEmptyName) {
    EXPECT_THROW((json{{"weight", 1.0}}).get<ArchetypeSpec>(), std::invalid_argument);
    EXPECT_THROW((json{{"name", ""}}).get<ArchetypeSpec>(), std::invalid_argument);
}

TEST(ArchetypeConfigTest, FromJsonRejectsNonPositiveWeight) {
    EXPECT_THROW((json{{"name", "x"}, {"weight", 0.0}}).get<ArchetypeSpec>(),
                 std::invalid_argument);
    EXPECT_THROW((json{{"name", "x"}, {"weight", -1.0}}).get<ArchetypeSpec>(),
                 std::invalid_argument);
}

TEST(ArchetypeConfigTest, FromJsonValidatesCoherenceAndSpread) {
    EXPECT_THROW((json{{"name", "x"}, {"visual_coherence", 1.5}}).get<ArchetypeSpec>(),
                 std::invalid_argument);
    EXPECT_THROW((json{{"name", "x"}, {"music_coherence", -0.1}}).get<ArchetypeSpec>(),
                 std::invalid_argument);
    EXPECT_THROW((json{{"name", "x"}, {"scalar_spread", -0.01}}).get<ArchetypeSpec>(),
                 std::invalid_argument);
    EXPECT_THROW((json{{"name", "x"}, {"niche_cohort_width", -0.2}}).get<ArchetypeSpec>(),
                 std::invalid_argument);
}

TEST(ArchetypeConfigTest, FromJsonKeepsDefaultsForMissingParams) {
    // A name-only entry (the minimal valid config archetype) keeps every distribution default.
    auto a = json{{"name", "minimal"}}.get<ArchetypeSpec>();
    const ArchetypeSpec def{};
    EXPECT_EQ(a.name, "minimal");
    EXPECT_DOUBLE_EQ(a.weight, def.weight);
    EXPECT_DOUBLE_EQ(a.usefulnessMean, def.usefulnessMean);
    EXPECT_DOUBLE_EQ(a.controversyMean, def.controversyMean);
    EXPECT_DOUBLE_EQ(a.scalarSpread, def.scalarSpread);
    EXPECT_DOUBLE_EQ(a.musicCoherence, def.musicCoherence);
    EXPECT_DOUBLE_EQ(a.satisfactionBias, def.satisfactionBias);
    EXPECT_DOUBLE_EQ(a.nicheCohortWidth, def.nicheCohortWidth);
}

TEST(ArchetypeConfigTest, FromJsonReadsEveryParameter) {
    json j = {{"name", "custom"},
              {"weight", 3.5},
              {"usefulness_mean", 0.11},
              {"humour_mean", 0.22},
              {"novelty_mean", 0.33},
              {"controversy_mean", 0.44},
              {"clickbait_strength_mean", 0.55},
              {"information_density_mean", 0.66},
              {"emotional_intensity_mean", 0.77},
              {"production_quality_shift", -0.12},
              {"scalar_spread", 0.09},
              {"visual_coherence", 0.81},
              {"music_coherence", 0.82},
              {"emotional_coherence", 0.83},
              {"satisfaction_bias", -0.4},
              {"regret_bias", 0.3},
              {"opening_hook", 0.7},
              {"retention_decay", 0.6},
              {"niche_cohort_width", 0.2},
              {"comfort_return_bonus", 0.5}};
    auto a = j.get<ArchetypeSpec>();
    EXPECT_EQ(a.name, "custom");
    EXPECT_DOUBLE_EQ(a.weight, 3.5);
    EXPECT_DOUBLE_EQ(a.usefulnessMean, 0.11);
    EXPECT_DOUBLE_EQ(a.humourMean, 0.22);
    EXPECT_DOUBLE_EQ(a.noveltyMean, 0.33);
    EXPECT_DOUBLE_EQ(a.controversyMean, 0.44);
    EXPECT_DOUBLE_EQ(a.clickbaitStrengthMean, 0.55);
    EXPECT_DOUBLE_EQ(a.informationDensityMean, 0.66);
    EXPECT_DOUBLE_EQ(a.emotionalIntensityMean, 0.77);
    EXPECT_DOUBLE_EQ(a.productionQualityShift, -0.12);
    EXPECT_DOUBLE_EQ(a.scalarSpread, 0.09);
    EXPECT_DOUBLE_EQ(a.visualCoherence, 0.81);
    EXPECT_DOUBLE_EQ(a.musicCoherence, 0.82);
    EXPECT_DOUBLE_EQ(a.emotionalCoherence, 0.83);
    EXPECT_DOUBLE_EQ(a.satisfactionBias, -0.4);
    EXPECT_DOUBLE_EQ(a.regretBias, 0.3);
    EXPECT_DOUBLE_EQ(a.openingHook, 0.7);
    EXPECT_DOUBLE_EQ(a.retentionDecay, 0.6);
    EXPECT_DOUBLE_EQ(a.nicheCohortWidth, 0.2);
    EXPECT_DOUBLE_EQ(a.comfortReturnBonus, 0.5);
}

TEST(ArchetypeConfigTest, ToJsonFromJsonRoundTripsEveryField) {
    // Every archetype in the default catalog survives a to_json -> from_json round-trip unchanged
    // (this is what keeps the ExperimentConfig round-trip tests green after the struct grew).
    for (const ArchetypeSpec &original : defaultArchetypeCatalog()) {
        json j = original;
        auto back = j.get<ArchetypeSpec>();
        EXPECT_EQ(original, back) << "round-trip changed " << original.name;
    }
}

TEST(ArchetypeConfigTest, DefaultConstructibleAndEquality) {
    ArchetypeSpec a;
    ArchetypeSpec b;
    EXPECT_EQ(a, b);
    b.controversyMean += 0.1;
    EXPECT_NE(a, b);
}
